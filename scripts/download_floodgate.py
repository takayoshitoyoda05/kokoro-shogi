"""floodgate (wdoor) の棋譜をダウンロードして data/ へ展開する。

取得元は2通り:

- ``--year 2024``: 年次アーカイブ ``archive/wdoorYYYY.7z`` (約350MB/年)。
  本番の学習コーパスはこちら。サーバへのリクエストは1回で済む。
- ``--date 2025-01-15``: その日のディレクトリから .csa を個別取得 (数百局)。
  パイプラインの動作確認用。**日付を大量に指定して年次分を集めないこと**
  (1局1リクエストになりサーバに負荷をかける)。

出力先は既定で ``data/floodgate/`` (.gitignore 済み。棋譜はコミットしない)。
すでにあるファイルは再取得しない。

使い方::

    uv run python scripts/download_floodgate.py --date 2025-01-15
    uv run python scripts/download_floodgate.py --year 2024
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import requests

from kokoro_shogi.config import REPO_ROOT

BASE_URL = "http://wdoor.c.u-tokyo.ac.jp/shogi"
ARCHIVE_URL = BASE_URL + "/archive/wdoor{year}.7z"
DAY_URL = BASE_URL + "/x/{year:04d}/{month:02d}/{day:02d}/"

DEFAULT_OUT = REPO_ROOT / "data" / "floodgate"
CHUNK = 1 << 20  # 1MiB
TIMEOUT = 60

_CSA_LINK = re.compile(r'href="([^"]+\.csa)"')


def download_file(url: str, destination: Path, *, force: bool = False) -> bool:
    """URLをファイルへ保存する。すでにあれば何もしない。戻り値は取得したか。"""
    if destination.exists() and not force:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    # 進捗の上書き表示は端末のときだけ。ログへリダイレクトすると \r が溜まって読めなくなる
    show_progress = sys.stderr.isatty()

    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with partial.open("wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK):
                file.write(chunk)
                written += len(chunk)
                if total and show_progress:
                    print(
                        f"\r  {destination.name}: {written / 1e6:.0f}/{total / 1e6:.0f} MB "
                        f"({100 * written / total:.0f}%)",
                        end="",
                        file=sys.stderr,
                    )
        if total and show_progress:
            print(file=sys.stderr)

    partial.replace(destination)
    return True


def download_year(year: int, out_dir: Path, *, force: bool = False) -> Path:
    """年次アーカイブを取得して展開する。展開先ディレクトリを返す。"""
    import py7zr

    archive_dir = out_dir / "archives"
    archive_path = archive_dir / f"wdoor{year}.7z"
    extract_dir = out_dir / "csa" / str(year)

    if download_file(ARCHIVE_URL.format(year=year), archive_path, force=force):
        print(f"取得: {archive_path}")
    else:
        print(f"取得済み (スキップ): {archive_path}")

    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        print(f"展開済み (スキップ): {extract_dir}")
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"展開中 (数分かかります): {archive_path} → {extract_dir}")
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=extract_dir)

    return extract_dir


def download_day(target: date, out_dir: Path, *, force: bool = False) -> Path:
    """1日ぶんの .csa を個別に取得する (動作確認用)。"""
    index_url = DAY_URL.format(year=target.year, month=target.month, day=target.day)
    destination = out_dir / "csa" / f"{target:%Y-%m-%d}"

    response = requests.get(index_url, timeout=TIMEOUT)
    response.raise_for_status()
    names = sorted(set(_CSA_LINK.findall(response.text)))
    if not names:
        print(f"{target} の棋譜は見つかりませんでした。")
        return destination

    fetched = 0
    for name in names:
        if download_file(index_url + name, destination / name, force=force):
            fetched += 1

    print(f"{target}: {len(names)} 局中 {fetched} 局を新規取得 → {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="年次アーカイブを取得する (例: 2024)")
    group.add_argument("--date", type=date.fromisoformat, help="1日分だけ取得する (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="出力先")
    parser.add_argument("--force", action="store_true", help="既存ファイルも取り直す")
    args = parser.parse_args()

    if args.year is not None:
        target_dir = download_year(args.year, args.out_dir, force=args.force)
    else:
        target_dir = download_day(args.date, args.out_dir, force=args.force)

    count = sum(1 for _ in target_dir.rglob("*.csa"))
    print(f"{target_dir} に .csa が {count} 件あります。")


if __name__ == "__main__":
    main()
