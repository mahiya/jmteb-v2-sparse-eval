"""モデルローダ. dense / sparse を一つの API (build_model) で扱う.

max_seq_length はここでは設定しない. モデルの素の上限を保ったままロードし,
タスクごとの切り詰めは run_evaluation 側で per-task に適用する (evaluator.py 参照).
"""

from __future__ import annotations

from typing import Any, Optional


def _seq_target(model_obj: Any) -> Any:
    """max_seq_length を読み書きする実体を返す.

    mteb / sentence_transformers のモデルは内側の `.model` が seq 長を持つことが多いので,
    あれば inner を優先し, 無ければ model_obj 自身を対象にする.
    """
    inner = getattr(model_obj, "model", None)
    return inner if inner is not None else model_obj


def get_max_seq_length(model_obj: Any) -> Optional[int]:
    """モデルの現在の max_seq_length を返す (持っていなければ None)."""
    val = getattr(_seq_target(model_obj), "max_seq_length", None)
    return int(val) if val is not None else None


def get_model_max_context(model_obj: Any) -> Optional[int]:
    """ターゲットモデルの最大コンテキスト長 (architectural) を推定して返す.

    HF transformers の config (max_position_embeddings 等) を複数経路で探す.
    sparse (SparseEncoder) / dense (mteb) でラッパ構造が違うため経路を順に試し,
    どうしても取れなければ None を返す (呼び出し側で現在の max_seq_length にフォールバック).
    """
    target = _seq_target(model_obj)
    config = None
    for accessor in (
        lambda t: t._first_module().auto_model.config,
        lambda t: t[0].auto_model.config,
        lambda t: t.auto_model.config,
        lambda t: t.model.config,
        lambda t: t.config,
    ):
        try:
            c = accessor(target)
        except Exception:
            continue
        if c is not None:
            config = c
            break
    if config is None:
        return None
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        val = getattr(config, key, None)
        if val:
            try:
                return int(val)
            except Exception:
                pass
    return None


def set_max_seq_length(model_obj: Any, max_seq_length: Optional[int]) -> None:
    """モデルの max_seq_length を設定する (None なら no-op).

    モデル最大コンテキスト長との min を取るキャップは呼び出し側 (run_evaluation) の責務.
    ここでは渡された値をそのまま設定し, 失敗時はエラーを print して握りつぶす.
    """
    if max_seq_length is None:
        return
    try:
        _seq_target(model_obj).max_seq_length = int(max_seq_length)
    except Exception as e:
        print(f"[model_loader] failed to set max_seq_length: {e}")


def build_model(model_id: str, is_sparse: bool) -> Any:
    """dense / sparse を切り替えてモデルをロードする.

    is_sparse=True なら SPLADE 等を SpladeMTEBWrapper でラップして返し, False なら
    mteb.get_model でロードする. 戻り値は JMTEBV2Evaluator(model=...) にそのまま渡せる.

    mteb / sparse_wrapper の import を関数内に置くのは遅延ロードのため. setup_environment()
    が pin 版 mteb を入れ JMTEB を sys.path に通した後でないと import できない.
    """
    if is_sparse:
        from .sparse_wrapper import SpladeMTEBWrapper

        return SpladeMTEBWrapper(model_id)
    import mteb

    return mteb.get_model(model_id)
