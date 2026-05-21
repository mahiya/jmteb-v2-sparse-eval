"""モデル単位/タスク単位の 2 段 try で囲んだ評価ループ."""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
import traceback
from glob import glob
from typing import Any, Iterable, Optional

from .jsonio import read_json, write_json
from .model_loader import get_max_seq_length, get_model_max_context, set_max_seq_length
from .tasks import get_target_tasks


def _model_subdir(model_id: str) -> str:
    return model_id.replace("/", "__")


def _task_already_done(output_dir: str, task_name: str) -> bool:
    """summary.json と details/<TASK>.json の両方を持っていれば skip."""
    summary_path = os.path.join(output_dir, f"{task_name}.json")
    details_path = os.path.join(output_dir, "details", f"{task_name}.json")
    return os.path.exists(summary_path) and os.path.exists(details_path)


def _copy_details(cache_path: str, model_id: str, output_details_dir: str) -> int:
    pattern = os.path.join(cache_path, "results", _model_subdir(model_id), "**", "*.json")
    paths = glob(pattern, recursive=True)
    os.makedirs(output_details_dir, exist_ok=True)
    count = 0
    for p in paths:
        # model_meta.json は details ではなく兄弟の model_meta/ へ振り分けるためここでは除外
        if os.path.basename(p) == "model_meta.json":
            continue
        dst = os.path.join(output_details_dir, os.path.basename(p))
        try:
            shutil.copy2(p, dst)
            count += 1
        except Exception as e:
            print(f"[evaluator] failed to copy details {p}: {e}")
    return count


def _find_source_model_meta(cache_path: str, model_id: str) -> Optional[str]:
    """mteb が results 配下に出力する単一の model_meta.json のパスを返す (無ければ None)."""
    pattern = os.path.join(cache_path, "results", _model_subdir(model_id), "**", "model_meta.json")
    paths = glob(pattern, recursive=True)
    return paths[0] if paths else None


def _copy_model_meta(
    cache_path: str, model_id: str, output_dir: str, task_names: Iterable[str]
) -> int:
    """results 配下の単一の model_meta.json を, details と兄弟の model_meta/ ディレクトリへ
    タスク名ごとに配置する (model_meta/<TASK>.json). details/<TASK>.json と 1 対 1 に
    対応させるため, details が存在するタスクについてのみ複製する.
    """
    src = _find_source_model_meta(cache_path, model_id)
    if src is None:
        return 0
    details_dir = os.path.join(output_dir, "details")
    meta_dir = os.path.join(output_dir, "model_meta")
    os.makedirs(meta_dir, exist_ok=True)
    count = 0
    for t in task_names:
        if not os.path.exists(os.path.join(details_dir, f"{t}.json")):
            continue
        dst = os.path.join(meta_dir, f"{t}.json")
        try:
            shutil.copy2(src, dst)
            count += 1
        except Exception as e:
            print(f"[evaluator] failed to copy model_meta for {t}: {e}")
    return count


# JMTEB v2 の _get_task_key 相当. summary 復元時に使う.
# JMTEB の `jmteb/v2/utils.py::_get_task_key` と整合させる. 不整合だと summary 復元時に
# 別キーで書かれて下流の集計と齟齬が出るので注意.
_TASK_KEY_MAP = {
    "NLPJournalTitleAbsRetrieval.V2": "nlp_journal_title_abs",
    "NLPJournalTitleIntroRetrieval.V2": "nlp_journal_title_intro",
    "NLPJournalAbsIntroRetrieval.V2": "nlp_journal_abs_intro",
    "NLPJournalAbsArticleRetrieval.V2": "nlp_journal_abs_article",
    "MintakaRetrieval": "mintaka_retrieval",
    "JaGovFaqsRetrieval": "jagovfaqs_22k",
    "JaqketRetrieval": "jaqket",
    "JaCWIRRetrieval": "jacwir_retrieval",
    "MultiLongDocRetrieval": "mldr_retrieval",
    "MIRACLRetrieval": "miracl_retrieval",
    "MrTidyRetrieval": "mrtydi",
}


def _restore_summary_from_details(
    summary_path: str,
    details_path: str,
    task_name: str,
) -> bool:
    """JMTEB の `_extract_main_score` が split 不一致 (例: MIRACL は scores.dev のみ)
    で main_score=None を返して summary.json が `{}` になるケースを救済する.

    details JSON (mteb 出力形式) を読み, scores.dev / scores.test から main_score を抜き出して
    JMTEB の summary 形式に書き戻す.
    """
    current = read_json(summary_path, default=None)
    # 読めない (None) か, 既に正常な summary がある (truthy) なら復元しない
    if current is None or current:
        return False

    details = read_json(details_path, default=None)
    if details is None:
        print(f"[evaluator] details load failed for {task_name}")
        return False

    scores = details.get("scores") or {}
    item = None
    for split in ("test", "dev", "validation"):
        if split in scores and scores[split]:
            item = scores[split][0]
            break
    if not item:
        return False

    main_score = item.get("main_score")
    if main_score is None:
        return False

    task_key = _TASK_KEY_MAP.get(task_name, task_name.lower())
    eval_time = details.get("evaluation_time") or 0.0

    summary = {
        "Retrieval": {
            task_key: {
                "main_metric": "ndcg_at_10",
                "main_score": main_score * 100,
                "eval_time (s)": "%.2f" % eval_time,
            }
        }
    }
    write_json(summary_path, summary)
    print(f"[evaluator] restored empty summary from details: {task_name}")
    return True


def _restore_empty_summaries(output_dir: str, task_names: Iterable[str]) -> int:
    n = 0
    for t in task_names:
        sp = os.path.join(output_dir, f"{t}.json")
        dp = os.path.join(output_dir, "details", f"{t}.json")
        if os.path.exists(sp) and os.path.exists(dp):
            cur = read_json(sp, default=None)
            if not cur:
                if _restore_summary_from_details(sp, dp, t):
                    n += 1
    return n


def run_evaluation(
    *,
    model: Any,
    model_id: str,
    task_names: Iterable[str],
    output_root_dir: str,
    task_batch_sizes: dict[str, int],
    cache_path: str = "./cached_results",
    task_max_seq_lengths: Optional[dict[str, int]] = None,
) -> dict:
    """与えられたモデルで指定タスクを評価する.

    モデルロードは呼び出し側で行う (build_model を使う). タスクごとに try/except +
    finally で囲み, OOM 等を捕捉して次タスクへ進みつつ, finally で中間テンソルを解放
    する (タスク間の GPU メモリ累積を防ぐ). モデル本体の解放は所有者である呼び出し側の
    責務 (eval.py 参照).

    Returns:
      実行結果のサマリ (results / errors / skipped / output_dir / elapsed_sec).
    """
    from jmteb.v2 import JMTEBV2Evaluator

    import torch

    model_dir_name = _model_subdir(model_id)
    output_dir = os.path.join(output_root_dir, model_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    task_names = list(task_names)
    missing_bs = [t for t in task_names if t not in task_batch_sizes]
    assert not missing_bs, f"task_batch_sizes に値が無いタスク: {missing_bs}"

    print(f"[evaluator] model={model_id} output_dir={output_dir}")
    print(f"[evaluator] task_batch_sizes={task_batch_sizes}")

    # seq 上限を 1 回だけ確定する. ターゲットモデルの最大コンテキスト長 (architectural) が
    # 取れればそれを上限に, 取れなければ現在の max_seq_length を上限にフォールバックする.
    # config 由来なので per-task で上げ下げしても基準値が動かない (単調減少の心配なし).
    current_seq = get_max_seq_length(model)
    model_max_context = get_model_max_context(model)
    seq_cap = model_max_context if model_max_context is not None else current_seq
    print(f"[evaluator] model_max_context={model_max_context} (current max_seq_length={current_seq}) -> seq cap={seq_cap}")
    print(f"[evaluator] task_max_seq_lengths={task_max_seq_lengths}")

    results: dict[str, dict] = {}
    errors: list[dict] = []
    skipped: list[str] = []
    overall_start = time.time()

    for task_name in task_names:
        task_start = time.time()
        summary_path = os.path.join(output_dir, f"{task_name}.json")
        details_path = os.path.join(output_dir, "details", f"{task_name}.json")

        # スキップ判定 (summary + details の双方が必要)
        if _task_already_done(output_dir, task_name):
            print(f"[evaluator] skip (already done): {task_name}")
            results[task_name] = read_json(summary_path, default={})
            skipped.append(task_name)
            continue

        # 既に summary だけある (details が cache 揮発で失われた) ケースは
        # 上書きしないで details 補完のためにキャッシュへの再評価を促す.

        # per-task max_seq_length = min(指定値, モデル最大コンテキスト長). 未指定なら現状維持.
        req_seq = (task_max_seq_lengths or {}).get(task_name)
        if req_seq is not None:
            eff_seq = min(int(req_seq), seq_cap) if seq_cap is not None else int(req_seq)
            set_max_seq_length(model, eff_seq)

        task_bs = task_batch_sizes[task_name]
        print(f"[evaluator] === run task: {task_name} (max_seq_length={get_max_seq_length(model)}, batch_size={task_bs}) ===")

        try:
            tasks = get_target_tasks([task_name])
            with tempfile.TemporaryDirectory() as work_dir:
                JMTEBV2Evaluator(
                    model=model,
                    tasks=tasks,
                    save_path=work_dir,
                    batch_size=task_bs,
                    task_batch_sizes=task_batch_sizes,
                    cache_path=cache_path,
                ).run()

                src_summary = os.path.join(work_dir, "summary.json")
                if os.path.exists(src_summary):
                    # summary.json は results/<model>/<task>.json として保存
                    shutil.copy2(src_summary, summary_path)
                    results[task_name] = read_json(summary_path, default={})
                else:
                    print(f"[evaluator] WARN: summary.json not produced for {task_name}")
                    results[task_name] = {}

            # details を逐次コピー (cache_path 内に既に書かれているはず)
            _copy_details(cache_path, model_id, os.path.join(output_dir, "details"))
            # model_meta.json を model_meta/<TASK>.json として配置 (details と 1 対 1)
            _copy_model_meta(cache_path, model_id, output_dir, [task_name])

            elapsed = time.time() - task_start
            print(f"[evaluator] OK   {task_name}  ({elapsed:.1f}s)")
        except Exception as e:
            tb = traceback.format_exc()
            elapsed = time.time() - task_start
            msg = f"{type(e).__name__}: {e}"
            print(f"[evaluator] ERR  {task_name}  ({elapsed:.1f}s): {msg}")
            print(tb)
            errors.append(
                {
                    "task": task_name,
                    "error": msg,
                    "elapsed_sec": elapsed,
                    "traceback": tb,
                }
            )
        finally:
            # タスクごとに中間テンソル (埋め込み・類似度行列等) を解放する. タスク間で
            # GPU メモリが累積して OOM するのを防ぐためのもので, モデル本体の解放
            # (= 呼び出し側の責務) とは別の関心事. ここはモデル単位ではなくタスク単位
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 全タスク終了後にもう一度 details をフル同期
    n = _copy_details(cache_path, model_id, os.path.join(output_dir, "details"))
    print(f"[evaluator] details files copied: {n}")
    m = _copy_model_meta(cache_path, model_id, output_dir, task_names)
    print(f"[evaluator] model_meta files copied: {m}")

    # JMTEB の split 不一致で summary が {} のままになるケースを details から復元
    restored = _restore_empty_summaries(output_dir, task_names)
    if restored:
        print(f"[evaluator] restored {restored} empty summaries from details")
        # 復元結果を in-memory の results にも反映
        for t in task_names:
            sp = os.path.join(output_dir, f"{t}.json")
            if os.path.exists(sp):
                try:
                    results[t] = read_json(sp)
                except Exception:
                    pass

    overall_elapsed = time.time() - overall_start
    summary = {
        "model_id": model_id,
        "output_dir": output_dir,
        "results": results,
        "errors": errors,
        "skipped": skipped,
        "elapsed_sec": overall_elapsed,
        "task_batch_sizes": task_batch_sizes,
        "task_max_seq_lengths": task_max_seq_lengths,
    }
    return summary
