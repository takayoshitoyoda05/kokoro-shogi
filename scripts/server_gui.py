"""kokoro_shogi.server.server を操作するTkinter管理GUI。"""

from __future__ import annotations

import json
import queue
import socket
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

from kokoro_shogi.server import server as ws_server


class ServerGUI(tk.Tk):
    POLL_INTERVAL_MS = 150

    def __init__(self) -> None:
        super().__init__()
        self.title("Python ↔ Unity WebSocket Server")
        self.geometry("960x800")
        self.minsize(800, 650)

        self._ui_queue: queue.Queue[tuple[Callable[..., Any], tuple[Any, ...]]] = queue.Queue()
        self._last_client_ips: list[str] = []

        # サーバーはすべてのネットワークインターフェースで待ち受ける。
        # Unityへ案内する接続先には、実際のローカルIPv4アドレスを表示する。
        self.host_var = tk.StringVar(value="0.0.0.0")
        self.port_var = tk.StringVar(value="8765")
        self.status_var = tk.StringVar(value="停止中")
        self.server_address_var = tk.StringVar(value="接続先: サーバー未起動")
        self.send_mode_var = tk.StringVar(value="text")
        self.file_path_var = tk.StringVar()

        self._build_ui()
        self._set_running_ui(False)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(self.POLL_INTERVAL_MS, self._poll)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)
        root.rowconfigure(3, weight=2)

        # サーバー設定
        server_frame = ttk.LabelFrame(root, text="サーバー設定", padding=10)
        server_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        server_frame.columnconfigure(5, weight=1)

        ttk.Label(server_frame, text="待受IP").grid(row=0, column=0, padx=(0, 5))
        self.host_entry = ttk.Entry(server_frame, textvariable=self.host_var, width=16)
        self.host_entry.grid(row=0, column=1, padx=(0, 12))

        ttk.Label(server_frame, text="ポート").grid(row=0, column=2, padx=(0, 5))
        self.port_entry = ttk.Entry(server_frame, textvariable=self.port_var, width=8)
        self.port_entry.grid(row=0, column=3, padx=(0, 12))

        self.start_button = ttk.Button(server_frame, text="サーバー起動", command=self._start_server)
        self.start_button.grid(row=0, column=4, padx=(0, 6))

        self.stop_button = ttk.Button(server_frame, text="サーバー終了", command=self._stop_server)
        self.stop_button.grid(row=0, column=5, sticky="w")

        ttk.Label(server_frame, textvariable=self.status_var).grid(
            row=0, column=6, sticky="e", padx=(16, 0)
        )

        ttk.Label(
            server_frame,
            textvariable=self.server_address_var,
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=1, column=0, columnspan=7, sticky="w", pady=(10, 0))

        # 接続クライアント
        clients_frame = ttk.LabelFrame(root, text="接続中クライアントIP", padding=10)
        clients_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        clients_frame.rowconfigure(0, weight=1)
        clients_frame.columnconfigure(0, weight=1)

        self.client_list = tk.Listbox(clients_frame, exportselection=False)
        self.client_list.grid(row=0, column=0, sticky="nsew")
        client_scroll = ttk.Scrollbar(clients_frame, orient=tk.VERTICAL, command=self.client_list.yview)
        client_scroll.grid(row=0, column=1, sticky="ns")
        self.client_list.configure(yscrollcommand=client_scroll.set)

        # 送信領域
        send_frame = ttk.LabelFrame(root, text="選択したIPへ送信", padding=10)
        send_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
        send_frame.columnconfigure(0, weight=1)
        send_frame.rowconfigure(2, weight=1)

        mode_frame = ttk.Frame(send_frame)
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(
            mode_frame,
            text="テキスト",
            variable=self.send_mode_var,
            value="text",
            command=self._update_send_mode,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_frame,
            text="JSONファイル",
            variable=self.send_mode_var,
            value="file",
            command=self._update_send_mode,
        ).pack(side=tk.LEFT, padx=(15, 0))

        file_frame = ttk.Frame(send_frame)
        file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(0, weight=1)
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var)
        self.file_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.browse_button = ttk.Button(file_frame, text="参照", command=self._browse_json)
        self.browse_button.grid(row=0, column=1)

        self.text_input = scrolledtext.ScrolledText(send_frame, height=8, wrap=tk.WORD)
        self.text_input.grid(row=2, column=0, sticky="nsew", pady=(0, 8))

        self.send_button = ttk.Button(send_frame, text="送信", command=self._send_selected)
        self.send_button.grid(row=3, column=0, sticky="e")

        # クライアントから受信したテキスト
        received_frame = ttk.LabelFrame(root, text="クライアント受信テキスト", padding=10)
        received_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        received_frame.rowconfigure(0, weight=1)
        received_frame.columnconfigure(0, weight=1)

        self.received_text = scrolledtext.ScrolledText(
            received_frame, state=tk.DISABLED, wrap=tk.WORD, height=18
        )
        self.received_text.grid(row=0, column=0, sticky="nsew")

        clear_received_button = ttk.Button(
            received_frame, text="受信テキストを消去", command=self._clear_received_text
        )
        clear_received_button.grid(row=1, column=0, sticky="e", pady=(8, 0))

        # 全ログ
        log_frame = ttk.LabelFrame(root, text="受信・接続ログ", padding=10)
        log_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, wrap=tk.WORD)
        self.log.grid(row=0, column=0, sticky="nsew")

        clear_button = ttk.Button(log_frame, text="ログを消去", command=self._clear_log)
        clear_button.grid(row=1, column=0, sticky="e", pady=(8, 0))

        self._update_send_mode()

    def _start_server(self) -> None:
        try:
            host = self.host_var.get().strip() or "0.0.0.0"
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "ポート番号は整数で入力してください。")
            return

        self.status_var.set("起動処理中...")
        self.start_button.configure(state=tk.DISABLED)

        def worker() -> None:
            try:
                started = ws_server.start_server(host=host, port=port)
                self._post_ui(self._server_started, host, port, started)
            except Exception as exc:
                self._post_ui(self._operation_failed, "サーバー起動", exc)

        threading.Thread(target=worker, daemon=True).start()

    def _server_started(self, host: str, port: int, started: bool) -> None:
        self._set_running_ui(True)

        connection_ip = self._resolve_connection_ip(host)
        connection_url = f"ws://{connection_ip}:{port}"
        self.status_var.set("起動OK")

        if connection_ip != "127.0.0.1":
            self.server_address_var.set(
                f"接続先: {connection_url}    同一PCから: ws://127.0.0.1:{port}"
            )
        else:
            self.server_address_var.set(f"接続先: {connection_url}")

        if started:
            self._append_log(
                f"[SERVER] 待受を開始しました。Unityの接続先: {connection_url}"
            )
        else:
            self._append_log("[SERVER] サーバーはすでに起動しています。")

    def _stop_server(self) -> None:
        self.status_var.set("終了処理中...")
        self.stop_button.configure(state=tk.DISABLED)

        def worker() -> None:
            try:
                stopped = ws_server.end_server()
                self._post_ui(self._server_stopped, stopped)
            except Exception as exc:
                self._post_ui(self._operation_failed, "サーバー終了", exc)

        threading.Thread(target=worker, daemon=True).start()

    def _server_stopped(self, stopped: bool) -> None:
        self._set_running_ui(False)
        self.status_var.set("停止中")
        self.server_address_var.set("接続先: サーバー未起動")
        self._append_log("[SERVER] サーバーを終了しました。" if stopped else "[SERVER] すでに停止中です。")

    def _browse_json(self) -> None:
        path = filedialog.askopenfilename(
            title="送信するJSONファイルを選択",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.file_path_var.set(path)

    def _send_selected(self) -> None:
        selection = self.client_list.curselection()
        if not selection:
            messagebox.showwarning("送信先未選択", "接続中クライアントIPを選択してください。")
            return

        client_ip = self.client_list.get(selection[0])

        if self.send_mode_var.get() == "file":
            raw_path = self.file_path_var.get().strip()
            if not raw_path:
                messagebox.showwarning("ファイル未選択", "送信するJSONファイルを選択してください。")
                return
            data: Any = Path(raw_path)
        else:
            data = self.text_input.get("1.0", tk.END).rstrip("\n")
            if not data:
                messagebox.showwarning("テキスト未入力", "送信するテキストを入力してください。")
                return

        self.send_button.configure(state=tk.DISABLED)

        def worker() -> None:
            try:
                count = ws_server.send_client(client_ip, data)
                self._post_ui(self._send_completed, client_ip, count, data)
            except Exception as exc:
                self._post_ui(self._operation_failed, "送信", exc)

        threading.Thread(target=worker, daemon=True).start()

    def _send_completed(self, client_ip: str, count: int, data: Any) -> None:
        self.send_button.configure(state=tk.NORMAL)
        kind = "JSONファイル" if isinstance(data, Path) else "テキスト"
        self._append_log(f"[SEND] {client_ip} の {count} 接続へ{kind}を送信しました。")

    def _operation_failed(self, operation: str, exc: Exception) -> None:
        if operation == "サーバー起動":
            self._set_running_ui(False)
            self.status_var.set("起動失敗")
        elif operation == "サーバー終了":
            self.stop_button.configure(state=tk.NORMAL)
            self.status_var.set("終了失敗")
        elif operation == "送信":
            self.send_button.configure(state=tk.NORMAL)

        self._append_log(f"[ERROR] {operation}: {exc}")
        messagebox.showerror(f"{operation}エラー", str(exc))

    def _poll(self) -> None:
        self._drain_ui_queue()
        self._refresh_clients()
        self._drain_server_events()
        self._drain_received_messages()
        self.after(self.POLL_INTERVAL_MS, self._poll)

    def _refresh_clients(self) -> None:
        client_ips = ws_server.get_connected_clients()
        if client_ips == self._last_client_ips:
            return

        selected_ip = None
        selection = self.client_list.curselection()
        if selection:
            selected_ip = self.client_list.get(selection[0])

        self.client_list.delete(0, tk.END)
        for ip in client_ips:
            self.client_list.insert(tk.END, ip)

        if selected_ip in client_ips:
            index = client_ips.index(selected_ip)
            self.client_list.selection_set(index)
        elif client_ips:
            self.client_list.selection_set(0)

        self._last_client_ips = client_ips

    def _drain_server_events(self) -> None:
        while True:
            event = ws_server.get_server_event()
            if event is None:
                break

            event_type = event.get("event")
            if event_type == "client_connected":
                self._append_log(
                    f"[CONNECT] {event['client_ip']} が接続しました。"
                )
            elif event_type == "client_disconnected":
                self._append_log(
                    f"[DISCONNECT] {event['client_ip']} が切断しました。"
                )

    def _drain_received_messages(self) -> None:
        while True:
            record = ws_server.receive_any()
            if record is None:
                break

            data = record["data"]
            formatted = json.dumps(data, ensure_ascii=False, indent=2)
            self._append_log(
                f"[RECEIVE] {record['client_ip']}  {record['received_at']}\n{formatted}"
            )

            # Unityの Send("...") から届く {"text": "..."} を専用欄へ表示する。
            if isinstance(data, dict) and "text" in data:
                text_value = data["text"]
                if not isinstance(text_value, str):
                    text_value = json.dumps(text_value, ensure_ascii=False)
                self._append_received_text(
                    record["client_ip"], record["received_at"], text_value
                )

    def _update_send_mode(self) -> None:
        file_mode = self.send_mode_var.get() == "file"
        file_state = tk.NORMAL if file_mode else tk.DISABLED
        text_state = tk.DISABLED if file_mode else tk.NORMAL

        self.file_entry.configure(state=file_state)
        self.browse_button.configure(state=file_state)
        self.text_input.configure(state=text_state)

    def _set_running_ui(self, running: bool) -> None:
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.host_entry.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.port_entry.configure(state=tk.DISABLED if running else tk.NORMAL)

    def _post_ui(self, function: Callable[..., Any], *args: Any) -> None:
        self._ui_queue.put((function, args))

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                function, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            function(*args)

    @staticmethod
    def _get_local_ipv4() -> str:
        """このPCで通常使用されるローカルIPv4アドレスを返す。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # UDPの接続先は経路選択にだけ使われ、データは送信しない。
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            if local_ip and not local_ip.startswith("127."):
                return local_ip
        except OSError:
            pass
        finally:
            sock.close()

        try:
            for address in socket.gethostbyname_ex(socket.gethostname())[2]:
                if address and not address.startswith("127."):
                    return address
        except OSError:
            pass

        return "127.0.0.1"

    @classmethod
    def _resolve_connection_ip(cls, bind_host: str) -> str:
        """Unityから接続するときに使うIPアドレスを返す。"""
        normalized = bind_host.strip()
        if normalized and normalized not in {"0.0.0.0", "::"}:
            if normalized == "localhost":
                return "127.0.0.1"
            return normalized

        return cls._get_local_ipv4()

    def _append_received_text(self, client_ip: str, received_at: str, text: str) -> None:
        self.received_text.configure(state=tk.NORMAL)
        self.received_text.insert(
            tk.END, f"[{client_ip}  {received_at}]\n{text}\n\n"
        )
        self.received_text.see(tk.END)
        self.received_text.configure(state=tk.DISABLED)

    def _append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _clear_received_text(self) -> None:
        self.received_text.configure(state=tk.NORMAL)
        self.received_text.delete("1.0", tk.END)
        self.received_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        try:
            if ws_server.is_server_running():
                ws_server.end_server()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = ServerGUI()
    app.mainloop()