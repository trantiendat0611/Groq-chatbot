"""Khung công cụ (tools) cho agent + các công cụ tích hợp sẵn.

Mỗi Tool gồm: tên, mô tả, JSON schema tham số (chuẩn OpenAI/Groq function
calling) và hàm Python thực thi. Kết quả tool luôn là chuỗi văn bản trả
về cho model đọc.
"""

import ast
import json
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime


MAX_TOOL_RESULT_CHARS = 4000


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


def tool_schemas(tools: list[Tool]) -> list[dict]:
    """Chuyển danh sách Tool sang định dạng `tools` của Groq API."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def execute_tool(tools: list[Tool], name: str, arguments_json: str) -> str:
    """Thực thi tool theo tên với tham số JSON; mọi lỗi trả về chuỗi mô tả lỗi.

    Không raise: lỗi tool phải quay về model để nó tự xử lý/diễn giải,
    không được làm sập cả lượt trả lời.
    """
    tool = next((item for item in tools if item.name == name), None)
    if tool is None:
        return f"Lỗi: không có công cụ tên '{name}'."

    try:
        arguments = json.loads(arguments_json) if arguments_json.strip() else {}
        if not isinstance(arguments, dict):
            return "Lỗi: tham số công cụ phải là một JSON object."
    except json.JSONDecodeError as exc:
        return f"Lỗi: tham số không phải JSON hợp lệ ({exc})."

    try:
        result = tool.func(**arguments)
    except TypeError as exc:
        return f"Lỗi: tham số không khớp với công cụ ({exc})."
    except Exception as exc:
        return f"Lỗi khi chạy công cụ {name}: {exc}"

    result = str(result)
    if len(result) > MAX_TOOL_RESULT_CHARS:
        result = result[:MAX_TOOL_RESULT_CHARS] + "\n... (kết quả đã được cắt bớt)"
    return result


# ---------------------------------------------------------------------------
# Máy tính an toàn: duyệt cây cú pháp AST, chỉ cho phép phép toán số học.
# Tuyệt đối không dùng eval() — eval cho phép chạy code tùy ý.
# ---------------------------------------------------------------------------

# Chặn DoS: `9**9**9` sẽ treo CPU vô hạn nếu tính thẳng.
# Giới hạn để mọi biểu thức hợp lệ vẫn chạy, còn biểu thức "bom" bị từ chối ngay.
MAX_POW_EXPONENT = 1000
MAX_RESULT_DIGITS = 5000
MAX_AST_NODES = 200


def _guarded_pow(base: float, exponent: float) -> float:
    if abs(exponent) > MAX_POW_EXPONENT:
        raise ValueError(
            f"Số mũ quá lớn (tối đa {MAX_POW_EXPONENT}). Hãy dùng biểu thức nhỏ hơn."
        )

    # Ước lượng số chữ số của kết quả TRƯỚC khi tính, tránh treo CPU.
    if base != 0:
        estimated_digits = abs(exponent) * math.log10(abs(base))
        if estimated_digits > MAX_RESULT_DIGITS:
            raise ValueError(
                f"Kết quả quá lớn (khoảng {int(estimated_digits)} chữ số, "
                f"tối đa {MAX_RESULT_DIGITS})."
            )

    return operator.pow(base, exponent)


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _guarded_pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Hằng số không được phép: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Tên không được phép: {node.id}")

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](
            _evaluate_node(node.left), _evaluate_node(node.right)
        )

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise ValueError("Chỉ được gọi các hàm toán học cho phép.")
        if node.keywords:
            raise ValueError("Không hỗ trợ tham số có tên.")
        args = [_evaluate_node(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[node.func.id](*args)

    raise ValueError(f"Biểu thức chứa thành phần không được phép: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Biểu thức không hợp lệ: {exc}") from exc

    # Biểu thức lồng quá sâu cũng là một dạng tấn công làm cạn tài nguyên.
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_AST_NODES:
        raise ValueError("Biểu thức quá phức tạp.")

    result = _evaluate_node(tree)

    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def make_calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description=(
            "Tính toán biểu thức số học chính xác. Hỗ trợ + - * / // % **, "
            "các hàm sqrt, sin, cos, tan, log, exp, abs, round, min, max "
            "và hằng số pi, e. Dùng khi cần tính toán thay vì tự nhẩm."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Biểu thức cần tính, ví dụ: (125 * 37) / sqrt(16)",
                }
            },
            "required": ["expression"],
        },
        func=lambda expression: safe_calculate(expression),
    )


def make_time_tool() -> Tool:
    def get_time() -> str:
        now = datetime.now()
        weekdays = [
            "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm",
            "Thứ Sáu", "Thứ Bảy", "Chủ Nhật",
        ]
        return (
            f"Bây giờ là {now.strftime('%H:%M:%S')}, "
            f"{weekdays[now.weekday()]}, ngày {now.strftime('%d/%m/%Y')} (giờ máy local)."
        )

    return Tool(
        name="get_current_time",
        description="Lấy ngày giờ hiện tại theo đồng hồ máy. Dùng khi câu hỏi liên quan tới thời gian thực.",
        parameters={"type": "object", "properties": {}},
        func=get_time,
    )


def _default_web_search(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as client:
        return list(client.text(query, max_results=max_results))


def format_search_results(results: list[dict]) -> str:
    if not results:
        return "Không tìm thấy kết quả nào."

    sections = []
    for index, item in enumerate(results, start=1):
        title = item.get("title") or "(không có tiêu đề)"
        url = item.get("href") or item.get("url") or ""
        snippet = item.get("body") or item.get("snippet") or ""
        sections.append(f"[{index}] {title}\nURL: {url}\n{snippet}")
    return "\n\n".join(sections)


def make_web_search_tool(
    search_fn: Callable[[str, int], list[dict]] | None = None,
    max_results: int = 5,
) -> Tool:
    search_fn = search_fn or _default_web_search

    def run_search(query: str) -> str:
        try:
            results = search_fn(query, max_results)
        except Exception as exc:
            return f"Tìm kiếm web thất bại ({exc}). Hãy trả lời bằng kiến thức sẵn có và nói rõ chưa tra cứu được web."
        return format_search_results(results)

    return Tool(
        name="web_search",
        description=(
            "Tìm kiếm thông tin trên Internet (DuckDuckGo). Dùng khi cần thông tin "
            "mới, thời sự, giá cả, sự kiện sau thời điểm huấn luyện, hoặc khi không chắc chắn."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Từ khóa tìm kiếm, ngắn gọn và cụ thể.",
                }
            },
            "required": ["query"],
        },
        func=run_search,
    )


def make_document_search_tool(search_fn: Callable[[str, int], str], top_k: int = 4) -> Tool:
    """Tool tra cứu tài liệu đã nạp (RAG-as-a-tool): agent tự quyết khi nào cần tra."""
    return Tool(
        name="search_documents",
        description=(
            "Tìm kiếm trong các tài liệu người dùng đã upload (knowledge base). "
            "Dùng khi câu hỏi có thể liên quan tới nội dung tài liệu riêng của người dùng."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Nội dung cần tìm trong tài liệu.",
                }
            },
            "required": ["query"],
        },
        func=lambda query: search_fn(query, top_k),
    )


def make_remember_tool(save_fn: Callable[[str], bool]) -> Tool:
    def remember(fact: str) -> str:
        saved = save_fn(fact)
        if saved:
            return f"Đã ghi nhớ: {fact}"
        return "Điều này đã có trong trí nhớ từ trước."

    return Tool(
        name="remember",
        description=(
            "Lưu một sự thật quan trọng, ổn định về người dùng vào trí nhớ dài hạn "
            "(tên, nghề nghiệp, sở thích, dự án đang làm, ràng buộc...). "
            "Chỉ dùng khi người dùng chia sẻ thông tin đáng nhớ lâu dài hoặc yêu cầu ghi nhớ. "
            "Không lưu thông tin nhạy cảm (mật khẩu, số thẻ...)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "Sự thật cần nhớ, viết ngắn gọn một câu.",
                }
            },
            "required": ["fact"],
        },
        func=remember,
    )
