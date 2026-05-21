"""SparseEncoder を MTEB の EncoderProtocol に適合させる薄いラッパー."""

from __future__ import annotations

from typing import Any


class SpladeMTEBWrapper:
    """SPLADE / SparseEncoder の MTEB Retrieval 用ラッパー.

    @runtime_checkable な EncoderProtocol が isinstance でメソッド存在をチェック
    するため `similarity_pairwise` も実装が必要 (Retrieval では実際には呼ばれない).
    """

    def __init__(self, model_id: str, **load_kwargs: Any):
        from mteb.models.model_meta import ModelMeta
        from sentence_transformers import SparseEncoder

        self._model_id = model_id
        self.model = SparseEncoder(model_id, **load_kwargs)
        # README.md が無いモデルでも動くようフォールバック.
        try:
            self.mteb_model_meta = ModelMeta.from_sentence_transformer_model(self.model)
        except Exception as e:
            print(
                f"[sparse_wrapper] from_sentence_transformer_model failed: {e}. "
                "retrying with compute_metadata=False"
            )
            self.mteb_model_meta = ModelMeta.from_sentence_transformer_model(
                self.model, compute_metadata=False
            )
            # name が None だと結果保存時にエラーになるので念のため埋める.
            if not getattr(self.mteb_model_meta, "name", None):
                self.mteb_model_meta.name = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        prompt_type=None,
        batch_size: int = 32,
        **kwargs,
    ):
        # mteb v2.4 の inputs は List[Dict[str, List[str]]] (column-batch)
        try:
            texts = [t for batch in inputs for t in batch["text"]]
        except (TypeError, KeyError):
            texts = list(inputs)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_sparse_tensor=True,
        )

    def similarity(self, a, b):
        # 通常は SparseEncoder.similarity (= sparse-sparse torch.mm) に委譲.
        # ただし corpus が大きく sparse vector の NNZ も大きいケース (例: JaCWIRRetrieval) で
        # cuSPARSE の SpGEMM が `insufficient resources` で失敗するため,
        # その場合は b を dense に展開して sparse-dense matmul (= SpMM) でリトライする.
        import torch

        try:
            return self.model.similarity(a, b)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cusparse" not in msg and "spgemm" not in msg and "insufficient resources" not in msg:
                raise
            print(
                f"[sparse_wrapper] similarity: cuSPARSE SpGEMM failed ({type(e).__name__}). "
                "falling back to sparse×dense matmul."
            )
            torch.cuda.empty_cache()

            def _to_dense_if_sparse(t):
                return t.to_dense() if (hasattr(t, "is_sparse") and t.is_sparse) else t

            # b (corpus chunk) を dense に展開. shape (chunk, vocab). float32 で chunk=50k × vocab=30k ≒ 6GB
            b_dense = _to_dense_if_sparse(b)
            if hasattr(a, "is_sparse") and a.is_sparse:
                # torch.sparse.mm: sparse × dense → dense
                return torch.sparse.mm(a, b_dense.transpose(0, 1))
            return torch.matmul(a, b_dense.transpose(0, 1))

    def similarity_pairwise(self, a, b):
        # Retrieval では呼ばれないが EncoderProtocol の isinstance チェック用に必要.
        return self.model.similarity_pairwise(a, b)
