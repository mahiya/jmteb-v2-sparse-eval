"""JSON ファイルの読み書きユーティリティ (utf-8 固定)."""

from __future__ import annotations

import json
from typing import Any

_RAISE = object()  # default 省略を表すセンチネル


def read_json(path: str, default: Any = _RAISE) -> Any:
    """JSON を読み込む.

    default を省略すると, 読み込み失敗時 (ファイル無し / 壊れている等) に例外を送出する (strict).
    default を渡すと, 失敗時にその値を返す (lenient, resume 用途).
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        if default is _RAISE:
            raise
        return default


def write_json(path: str, obj: Any, *, indent: int = 4, ensure_ascii: bool = False) -> None:
    """JSON を書き出す (utf-8, 既定で indent=4 / ensure_ascii=False)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)
