"""Unity と通信する WebSocket サーバーモジュール。

公開関数:
    start_server(host="0.0.0.0", port=8765)
    end_server()
    send_client(client_ip, data)
    receive_client(client_ip)

追加の補助関数:
    get_connected_clients()
    receive_any()
    is_server_running()

送信形式:
- JSON ファイルのパス、dict、list: JSON 内容をそのまま送信
- 通常の文字列: {"text": "内容"} に変換して送信
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_MAX_SIZE = 64 * 1024 * 1024  # 64 MiB


class WebSocketServerManager:
    """asyncio の WebSocket サーバーを別スレッドで管理する。"""

    def __init__(self) -> None:
        self._host = DEFAULT_HOST
        self._port = DEFAULT_PORT
        self._max_size = DEFAULT_MAX_SIZE

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server = None
        self._stop_event: asyncio.Event | None = None

        self._started_event = threading.Event()
        self._start_error: BaseException | None = None
        self._state_lock = threading.RLock()

        # 次の3つは asyncio イベントループ内でのみ更新する。
        self._clients: dict[str, dict[str, ServerConnection]] = defaultdict(dict)
        self._send_locks: dict[str, asyncio.Lock] = {}

        # 外部スレッドから参照する接続IPのスナップショット。
        self._connected_ips: set[str] = set()
        self._connected_ips_lock = threading.RLock()

        # 受信キューは queue.Queue のため、GUIスレッドから安全に読める。
        self._received_by_ip: dict[str, queue.Queue[Any]] = defaultdict(queue.Queue)
        self._received_any: queue.Queue[dict[str, Any]] = queue.Queue()
        self._server_events: queue.Queue[dict[str, Any]] = queue.Queue()

    def start(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        max_size: int = DEFAULT_MAX_SIZE,
        startup_timeout: float = 5.0,
    ) -> bool:
        """サーバーを起動する。起動済みなら何もせず False を返す。"""
        with self._state_lock:
            if self.is_running():
                return False

            if not isinstance(port, int) or not 1 <= port <= 65535:
                raise ValueError("port は 1～65535 の整数で指定してください。")
            if max_size <= 0:
                raise ValueError("max_size は正の整数で指定してください。")

            self._host = host
            self._port = port
            self._max_size = max_size
            self._start_error = None
            self._started_event.clear()

            self._thread = threading.Thread(
                target=self._thread_main,
                name="UnityWebSocketServer",
                daemon=True,
            )
            self._thread.start()

        if not self._started_event.wait(timeout=startup_timeout):
            self.stop()
            raise TimeoutError("WebSocketサーバーの起動がタイムアウトしました。")

        if self._start_error is not None:
            error = self._start_error
            self.stop()
            raise RuntimeError(f"WebSocketサーバーを起動できませんでした: {error}") from error

        return True

    def stop(self, timeout: float = 5.0) -> bool:
        """サーバーと接続中のクライアントを終了する。"""
        with self._state_lock:
            thread = self._thread
            loop = self._loop
            stop_event = self._stop_event

            if thread is None:
                return False

            if loop is not None and stop_event is not None and loop.is_running():
                loop.call_soon_threadsafe(stop_event.set)

        if thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise TimeoutError("WebSocketサーバーを時間内に停止できませんでした。")

        with self._state_lock:
            self._thread = None
            self._loop = None
            self._server = None
            self._stop_event = None

        with self._connected_ips_lock:
            self._connected_ips.clear()

        return True

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive() and self._start_error is None)

    def get_connected_clients(self) -> list[str]:
        """現在接続中のクライアントIP一覧を返す。"""
        with self._connected_ips_lock:
            return sorted(self._connected_ips)

    def send(self, client_ip: str, data: Any, timeout: float = 10.0) -> int:
        """指定IPの全WebSocket接続へJSONメッセージを送る。

        戻り値は送信に成功した接続数。
        """
        if not client_ip or not client_ip.strip():
            raise ValueError("client_ip を指定してください。")

        payload = self._encode_payload(data)

        loop = self._loop
        if not self.is_running() or loop is None or not loop.is_running():
            raise RuntimeError("WebSocketサーバーが起動していません。")

        future = asyncio.run_coroutine_threadsafe(
            self._send_to_ip(client_ip.strip(), payload),
            loop,
        )
        return future.result(timeout=timeout)

    def receive(self, client_ip: str) -> Any | None:
        """指定IPから届いた次のJSONデータを非ブロッキングで返す。

        未受信なら None。JSONでない文字列を受けた場合は {"text": "..."} を返す。
        """
        if not client_ip:
            raise ValueError("client_ip を指定してください。")

        try:
            return self._received_by_ip[client_ip].get_nowait()
        except queue.Empty:
            return None

    def receive_any(self) -> dict[str, Any] | None:
        """送信元IPを問わず、次の受信レコードを非ブロッキングで返す。"""
        try:
            return self._received_any.get_nowait()
        except queue.Empty:
            return None

    def get_server_event(self) -> dict[str, Any] | None:
        """接続・切断イベントを1件返す。なければ None。"""
        try:
            return self._server_events.get_nowait()
        except queue.Empty:
            return None

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        with self._state_lock:
            self._loop = loop

        try:
            loop.run_until_complete(self._run_server())
        except BaseException as exc:  # 起動エラーも呼び出し元へ通知する。
            self._start_error = exc
            self._started_event.set()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def _run_server(self) -> None:
        self._stop_event = asyncio.Event()

        try:
            self._server = await serve(
                self._handle_client,
                self._host,
                self._port,
                max_size=self._max_size,
                ping_interval=20,
                ping_timeout=20,
            )
        except BaseException as exc:
            self._start_error = exc
            self._started_event.set()
            return

        self._server_events.put(
            {
                "event": "server_started",
                "host": self._host,
                "port": self._port,
                "time": self._now_iso(),
            }
        )
        self._started_event.set()

        await self._stop_event.wait()

        self._server.close(close_connections=True, reason="Server stopped")
        await self._server.wait_closed()

        self._clients.clear()
        self._send_locks.clear()
        with self._connected_ips_lock:
            self._connected_ips.clear()

        self._server_events.put(
            {
                "event": "server_stopped",
                "host": self._host,
                "port": self._port,
                "time": self._now_iso(),
            }
        )

    async def _handle_client(self, websocket: ServerConnection) -> None:
        remote = websocket.remote_address
        client_ip = str(remote[0]) if isinstance(remote, tuple) and remote else str(remote)
        connection_id = str(websocket.id)

        self._clients[client_ip][connection_id] = websocket
        self._send_locks[connection_id] = asyncio.Lock()
        self._refresh_connected_ip_snapshot()

        self._server_events.put(
            {
                "event": "client_connected",
                "client_ip": client_ip,
                "connection_id": connection_id,
                "time": self._now_iso(),
            }
        )

        try:
            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    raw_text = raw_message.decode("utf-8", errors="replace")
                else:
                    raw_text = raw_message

                try:
                    parsed: Any = json.loads(raw_text)
                except json.JSONDecodeError:
                    parsed = {"text": raw_text}

                self._received_by_ip[client_ip].put(parsed)
                self._received_any.put(
                    {
                        "client_ip": client_ip,
                        "connection_id": connection_id,
                        "received_at": self._now_iso(),
                        "data": parsed,
                        "raw": raw_text,
                    }
                )
        except ConnectionClosed:
            pass
        finally:
            clients_for_ip = self._clients.get(client_ip)
            if clients_for_ip is not None:
                clients_for_ip.pop(connection_id, None)
                if not clients_for_ip:
                    self._clients.pop(client_ip, None)

            self._send_locks.pop(connection_id, None)
            self._refresh_connected_ip_snapshot()

            self._server_events.put(
                {
                    "event": "client_disconnected",
                    "client_ip": client_ip,
                    "connection_id": connection_id,
                    "time": self._now_iso(),
                }
            )

    async def _send_to_ip(self, client_ip: str, payload: str) -> int:
        connections = list(self._clients.get(client_ip, {}).items())
        if not connections:
            raise KeyError(f"IP {client_ip} の接続中クライアントが見つかりません。")

        success_count = 0
        failed_ids: list[str] = []

        for connection_id, websocket in connections:
            lock = self._send_locks.get(connection_id)
            if lock is None:
                failed_ids.append(connection_id)
                continue

            try:
                async with lock:
                    await websocket.send(payload)
                success_count += 1
            except ConnectionClosed:
                failed_ids.append(connection_id)

        if success_count == 0:
            raise ConnectionError(
                f"IP {client_ip} への送信に失敗しました。切断済み接続: {failed_ids}"
            )

        return success_count

    def _refresh_connected_ip_snapshot(self) -> None:
        with self._connected_ips_lock:
            self._connected_ips = {
                ip for ip, connections in self._clients.items() if connections
            }

    @staticmethod
    def _encode_payload(data: Any) -> str:
        # Path オブジェクトは JSON ファイルとして扱う。
        if isinstance(data, Path):
            return WebSocketServerManager._load_json_file(data)

        # 既存の *.json ファイルを指す文字列もファイルとして扱う。
        if isinstance(data, str):
            possible_path = Path(data).expanduser()
            if possible_path.suffix.lower() == ".json" and possible_path.is_file():
                return WebSocketServerManager._load_json_file(possible_path)
            return json.dumps({"text": data}, ensure_ascii=False)

        # dict / list / 数値 / bool / None はJSONとして送れる。
        try:
            return json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "data にはJSONファイルのパス、文字列、またはJSON化可能な値を指定してください。"
            ) from exc

    @staticmethod
    def _load_json_file(path: Path) -> str:
        if path.suffix.lower() != ".json":
            raise ValueError("送信ファイルは .json を指定してください。")

        with path.open("r", encoding="utf-8-sig") as file:
            parsed = json.load(file)

        # 読み込んでから再エンコードし、妥当なJSONだけを送信する。
        return json.dumps(parsed, ensure_ascii=False)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


_manager = WebSocketServerManager()


def start_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    max_size: int = DEFAULT_MAX_SIZE,
) -> bool:
    """WebSocketサーバーを起動する。"""
    return _manager.start(host=host, port=port, max_size=max_size)


def end_server() -> bool:
    """WebSocketサーバーを終了する。"""
    return _manager.stop()


def send_client(client_ip: str, data: Any) -> int:
    """指定IPへJSONファイルまたはテキストを送信する。"""
    return _manager.send(client_ip=client_ip, data=data)


def receive_client(client_ip: str) -> Any | None:
    """指定IPから受信した次のデータを返す。未受信なら None。"""
    return _manager.receive(client_ip=client_ip)


def get_connected_clients() -> list[str]:
    """接続中IPの一覧を返す。"""
    return _manager.get_connected_clients()


def receive_any() -> dict[str, Any] | None:
    """送信元を問わず次の受信レコードを返す。"""
    return _manager.receive_any()


def get_server_event() -> dict[str, Any] | None:
    """サーバーイベントを1件返す。"""
    return _manager.get_server_event()


def is_server_running() -> bool:
    """サーバー起動中なら True。"""
    return _manager.is_running()
