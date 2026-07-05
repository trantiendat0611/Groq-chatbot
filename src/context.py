"""Quản lý cửa sổ ngữ cảnh: ước lượng token và cắt bớt lịch sử trước khi gọi API."""

from src.chat_types import ChatMessage


# Ước lượng thô: trung bình 1 token ~ 3 ký tự với văn bản Việt/Anh trộn lẫn.
# Cố tình ước lượng dư (chia 3 thay vì 4) để không bao giờ vượt ngưỡng thật.
CHARS_PER_TOKEN = 3

# Mỗi message có overhead định dạng (role, phân tách) phía API.
TOKENS_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: ChatMessage) -> int:
    return estimate_tokens(message.get("content", "")) + TOKENS_PER_MESSAGE_OVERHEAD


def trim_messages_to_budget(
    messages: list[ChatMessage],
    max_history_tokens: int,
) -> list[ChatMessage]:
    """Giữ lại các message MỚI NHẤT sao cho tổng token nằm trong ngân sách.

    - Luôn giữ message cuối cùng (câu hỏi hiện tại) kể cả khi nó một mình
      đã vượt ngân sách — không thể trả lời nếu thiếu nó.
    - Cắt từ message cũ nhất trở đi.
    - Không cắt lửng: message hoặc được giữ nguyên vẹn hoặc bị bỏ.
    """
    if not messages:
        return []

    kept_reversed: list[ChatMessage] = []
    remaining = max_history_tokens

    for message in reversed(messages):
        cost = estimate_message_tokens(message)
        if cost > remaining and kept_reversed:
            break
        kept_reversed.append(message)
        remaining -= cost

    kept = list(reversed(kept_reversed))

    # Tránh mở đầu bằng câu trả lời của assistant mà thiếu câu hỏi gốc:
    # nếu message đầu tiên còn lại là assistant, bỏ nó đi (trừ khi chỉ còn 1 message).
    while len(kept) > 1 and kept[0].get("role") == "assistant":
        kept = kept[1:]

    return kept
