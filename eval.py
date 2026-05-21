"""JMTEB v2 Retrieval 評価実行スクリプト.

実行時パラメータは config.json で指定する:
  - eval.py と同じディレクトリの config.json を読む
    (第1引数でパス指定も可: python eval.py path/to/config.json)
評価ロジックは modules/ 配下に分離してある.
"""

from __future__ import annotations

import gc
import os
import sys

# eval.py 自身のあるディレクトリ. modules/ をここから import するため sys.path に通す.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from modules import (
    setup_environment,
    build_model,
    run_evaluation,
)
from modules.jsonio import read_json

# モデル間レジューム用の中間キャッシュ (カレントディレクトリ直下に固定)
CACHE_PATH = "./cached_results"


def load_config() -> dict:
    """config.json を読み込む.

    パスは 第1引数 > <eval.py と同じディレクトリ>/config.json の順で決定する.
    """
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "config.json")
    cfg = read_json(config_path)
    print(f"[eval] loaded config: {config_path} -> {cfg}")
    return cfg


def resolve_target_models(cfg: dict) -> list[str]:
    """評価対象モデル ID のリストを返す.

    `target_models` (list[str]) を優先する. 後方互換として旧 `target_model` (str)
    も受け付け, その場合は 1 要素リストに包んで返す.
    """
    models = cfg.get("target_models")
    if models is None:
        single = cfg.get("target_model")
        models = [single] if single is not None else []
    if isinstance(models, str):  # target_models に文字列が来た場合の保険
        models = [models]
    if not models:
        raise ValueError("config に target_models (list[str]) または target_model (str) が必要です")
    return list(models)


def main() -> None:
    cfg = load_config()
    target_models = resolve_target_models(cfg)
    target_tasks = cfg["target_tasks"]

    # per-task max_seq_length を解決. config の task_max_seq_length をそのまま使う.
    # null のタスクは override せず, モデルの素の最大コンテキスト長で評価する.
    seq_overrides = cfg.get("task_max_seq_length") or {}
    task_max_seq_lengths = {t: seq_overrides.get(t) for t in target_tasks}

    # per-task バッチサイズを解決. config の task_batch_size が唯一の指定元で,
    # 全タスクに非 null の値が必須 (グローバルデフォルトは廃止).
    batch_overrides = cfg.get("task_batch_size") or {}
    task_batch_sizes = {}
    for t in target_tasks:
        v = batch_overrides.get(t)
        if v is None:
            raise ValueError(f"config の task_batch_size に {t} の値が必要です (per-task で必須)")
        task_batch_sizes[t] = int(v)

    # 1) 環境構築 (モデルに依存しないので 1 回だけ)
    setup_environment()

    # setup_environment() がパッケージを入れた後でないと import できないので遅延 import.
    import torch

    # 3) モデルごとに順次評価. run_evaluation も内部 finally で GPU を解放するが,
    #    そこで del されるのは引数の参照だけで, ここ (呼び出し側) の model 変数は残る.
    #    複数モデルを直列で回すため, 各モデル評価後に呼び出し側でも明示的に解放する.
    print(f"[eval] target_models ({len(target_models)}): {target_models}")
    for i, model_id in enumerate(target_models, start=1):
        print(f"[eval] === [{i}/{len(target_models)}] model: {model_id} ===")
        # max_seq_length は run_evaluation 側で per-task に適用
        model = build_model(model_id, is_sparse=cfg["is_sparse"])
        try:
            run_evaluation(
                model=model,
                model_id=model_id,
                task_names=target_tasks,
                output_root_dir=cfg["output_root_dir"],
                cache_path=CACHE_PATH,
                task_batch_sizes=task_batch_sizes,
                task_max_seq_lengths=task_max_seq_lengths,
            )
        finally:
            # 呼び出し側に残る参照を切ってから GPU メモリを解放する.
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[eval] gpu released (caller-side) after {model_id}")

    print("done.")


if __name__ == "__main__":
    main()
