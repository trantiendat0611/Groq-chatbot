"""Tạo embedding cho văn bản bằng fastembed (ONNX, không cần PyTorch).

Model mặc định là bản đa ngôn ngữ nên hiểu tốt tiếng Việt.
Lần đầu sử dụng, fastembed sẽ tải model (~470MB) về máy và cache lại.
Nếu fastembed chưa cài hoặc tải model thất bại, hệ thống tự fallback
về tìm kiếm từ khóa — app vẫn chạy bình thường.
"""

from src.config import DEFAULT_EMBEDDING_MODEL


class EmbeddingUnavailableError(RuntimeError):
    """Không khởi tạo được model embedding (thiếu thư viện, lỗi tải model...)."""


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "Chưa cài fastembed. Chạy: pip install fastembed"
            ) from exc

        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"Không khởi tạo được model embedding '{model_name}': {exc}"
            ) from exc

        self.model_name = model_name
        # Model họ E5 yêu cầu prefix "query:"/"passage:" để đạt chất lượng tốt.
        self._is_e5 = "e5" in model_name.lower()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._is_e5:
            texts = [f"passage: {text}" for text in texts]
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, query: str) -> list[float]:
        if self._is_e5:
            query = f"query: {query}"
        return next(iter(self._model.embed([query]))).tolist()


_cached_embedder: FastEmbedEmbedder | None = None
_embedder_failed = False


def get_default_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> FastEmbedEmbedder | None:
    """Trả về embedder dùng chung, hoặc None nếu không khả dụng (đã thử và thất bại)."""
    global _cached_embedder, _embedder_failed

    if _cached_embedder is not None:
        return _cached_embedder
    if _embedder_failed:
        return None

    try:
        _cached_embedder = FastEmbedEmbedder(model_name)
        return _cached_embedder
    except EmbeddingUnavailableError:
        _embedder_failed = True
        return None


def get_ready_embedder() -> FastEmbedEmbedder | None:
    """Trả về embedder CHỈ KHI đã khởi tạo sẵn — không bao giờ kích hoạt tải model.

    Dùng cho các đường nóng (hồi tưởng trí nhớ mỗi lượt chat) để tránh
    bất ngờ tải model ~470MB ngay giữa câu trả lời.
    """
    return _cached_embedder
