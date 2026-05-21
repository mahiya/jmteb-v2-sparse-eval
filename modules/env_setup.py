"""JMTEB 評価環境のセットアップ (clone / pip install / 環境変数)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


JMTEB_REPO_URL = "https://github.com/sbintuitions/JMTEB.git"
# クローン先 / src は相対指定. 実行時にカレントディレクトリ基準で絶対パスへ解決する
# (例: cwd=/content なら /content/JMTEB). 汎用化のため Colab 固有の絶対パスは持たない.
JMTEB_CLONE_DIR = "JMTEB"
JMTEB_SRC_DIR = "JMTEB/src"


def _run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print(f"[env_setup] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def clone_or_update_jmteb(repo_url: str = JMTEB_REPO_URL, dst: str = JMTEB_CLONE_DIR) -> str:
    """JMTEB を clone (既存なら git pull) して src パス (絶対パス) を返す.

    dst は相対パス可. カレントディレクトリ基準で絶対パスに解決される.
    """
    dst_path = Path(dst).resolve()
    if dst_path.exists() and (dst_path / ".git").exists():
        _run(["git", "-C", str(dst_path), "pull", "--ff-only"], check=False)
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", repo_url, str(dst_path)])
    return str(dst_path / "src")


def add_jmteb_to_path(jmteb_src: str = JMTEB_SRC_DIR) -> None:
    """sys.path と PYTHONPATH に JMTEB の src を追加 (相対パスは絶対パス化)."""
    jmteb_src = str(Path(jmteb_src).resolve())
    if jmteb_src not in sys.path:
        sys.path.insert(0, jmteb_src)
    cur = os.environ.get("PYTHONPATH", "")
    if jmteb_src not in cur.split(os.pathsep):
        os.environ["PYTHONPATH"] = jmteb_src + (os.pathsep + cur if cur else "")


def ensure_packages(
    packages: list[str] | None = None,
    remove_packages: list[str] | None = None,
) -> None:
    """必要パッケージをインストール (mteb は 2.4.2 固定)."""
    if packages is None:
        packages = [
            "mteb==2.4.2",
            "sentence-transformers>=5.4.1",
            "transformers>=5.0",
            "loguru",
            "jsonargparse[jsonnet]",
            "fugashi",
            "unidic_lite",
        ]
    if remove_packages is None:
        remove_packages = ["torchao"]

    # まず競合パッケージを除去
    for pkg in remove_packages:
        _run([sys.executable, "-m", "pip", "uninstall", "-y", pkg], check=False)

    # 必要パッケージのインストール
    _run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def set_cuda_alloc_env() -> None:
    """フラグメント回避用の PyTorch アロケータ環境変数を設定."""
    val = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = val
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = val


def enable_progress_logging() -> None:
    """mteb / sentence_transformers のロガーを INFO に上げて進捗を可視化.

    これで mteb の `Encoding Batch N/M` や Computing Similarities, sentence_transformers の
    "Batches" tqdm bar (logger 連動) などが stdout/stderr に流れて, 長時間タスクの
    進捗を確認できるようになる.
    """
    for name in ("mteb", "sentence_transformers", "jmteb"):
        logging.getLogger(name).setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def setup_environment(
    *,
    jmteb_repo_url: str = JMTEB_REPO_URL,
    jmteb_clone_dir: str = JMTEB_CLONE_DIR,
    install_packages: bool = True,
    packages: list[str] | None = None,
    remove_packages: list[str] | None = None,
) -> None:
    """全セットアップを一気通貫で実行する高レベル関数.

    Drive はマウント済み・環境変数 (HF_TOKEN / WANDB_* 等) は設定済みである前提.
    JMTEB はカレントディレクトリ直下 (<cwd>/JMTEB) に clone される (jmteb_clone_dir で上書き可).
    """
    # 1) PyTorch アロケータ環境変数 (importより前)
    set_cuda_alloc_env()

    # 2) JMTEB の clone / pull
    src = clone_or_update_jmteb(jmteb_repo_url, jmteb_clone_dir)

    # 3) sys.path / PYTHONPATH に追加
    add_jmteb_to_path(src)

    # 4) 必要パッケージのインストール
    if install_packages:
        ensure_packages(packages=packages, remove_packages=remove_packages)

    # 5) 進捗ロガー (mteb / sentence_transformers) を INFO に
    enable_progress_logging()

    print("[env_setup] setup complete")
