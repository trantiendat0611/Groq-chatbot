import pytest

from src.tools import (
    execute_tool,
    format_search_results,
    make_calculator_tool,
    make_document_search_tool,
    make_remember_tool,
    make_time_tool,
    make_web_search_tool,
    safe_calculate,
    tool_schemas,
)


# --- Máy tính an toàn ---


def test_calculator_basic_arithmetic():
    assert safe_calculate("2 + 3 * 4") == "14"
    assert safe_calculate("(10 - 4) / 3") == "2"
    assert safe_calculate("2 ** 10") == "1024"


def test_calculator_math_functions_and_constants():
    assert safe_calculate("sqrt(16)") == "4"
    assert float(safe_calculate("round(pi, 2)")) == 3.14
    assert safe_calculate("max(3, 7, 5)") == "7"


def test_calculator_rejects_code_execution():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('dir')")
    with pytest.raises(ValueError):
        safe_calculate("open('file.txt')")
    with pytest.raises(ValueError):
        safe_calculate("[1,2,3]")


def test_calculator_rejects_unknown_names():
    with pytest.raises(ValueError):
        safe_calculate("x + 1")


def test_calculator_invalid_syntax():
    with pytest.raises(ValueError):
        safe_calculate("2 +")


# --- Khung tool ---


def test_tool_schemas_format():
    schemas = tool_schemas([make_calculator_tool()])
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "calculator"
    assert "parameters" in schemas[0]["function"]


def test_execute_tool_happy_path():
    tools = [make_calculator_tool()]
    assert execute_tool(tools, "calculator", '{"expression": "6 * 7"}') == "42"


def test_execute_tool_unknown_tool():
    result = execute_tool([make_calculator_tool()], "khong_ton_tai", "{}")
    assert "không có công cụ" in result


def test_execute_tool_invalid_json():
    result = execute_tool([make_calculator_tool()], "calculator", "{hỏng json")
    assert result.startswith("Lỗi")


def test_execute_tool_wrong_arguments():
    result = execute_tool([make_calculator_tool()], "calculator", '{"sai_ten": "1+1"}')
    assert result.startswith("Lỗi")


def test_execute_tool_internal_error_returned_as_string():
    result = execute_tool([make_calculator_tool()], "calculator", '{"expression": "1/0"}')
    assert "Lỗi" in result


def test_execute_tool_truncates_long_results():
    def huge(**kwargs):
        return "x" * 10_000

    from src.tools import Tool

    tool = Tool(name="huge", description="", parameters={"type": "object", "properties": {}}, func=huge)
    result = execute_tool([tool], "huge", "{}")
    assert len(result) < 10_000
    assert "cắt bớt" in result


# --- Tool thời gian ---


def test_time_tool_returns_datetime_text():
    tool = make_time_tool()
    result = tool.func()
    assert "ngày" in result


# --- Tool tìm kiếm web (inject hàm giả, không gọi mạng) ---


def test_web_search_tool_formats_results():
    fake_results = [
        {"title": "Kết quả A", "href": "https://a.vn", "body": "Tóm tắt A"},
        {"title": "Kết quả B", "href": "https://b.vn", "body": "Tóm tắt B"},
    ]
    tool = make_web_search_tool(search_fn=lambda query, n: fake_results)
    output = tool.func(query="tin tức AI")
    assert "[1] Kết quả A" in output
    assert "https://b.vn" in output


def test_web_search_tool_handles_failure_gracefully():
    def broken(query, n):
        raise ConnectionError("mất mạng")

    tool = make_web_search_tool(search_fn=broken)
    output = tool.func(query="bất kỳ")
    assert "thất bại" in output


def test_format_search_results_empty():
    assert "Không tìm thấy" in format_search_results([])


# --- Tool tra cứu tài liệu ---


def test_document_search_tool_delegates_to_search_fn():
    calls = []

    def fake_search(query, top_k):
        calls.append((query, top_k))
        return "[1] Source: a.txt\nnội dung liên quan"

    tool = make_document_search_tool(fake_search, top_k=3)
    output = tool.func(query="tìm gì đó")

    assert calls == [("tìm gì đó", 3)]
    assert "nội dung liên quan" in output


# --- Tool ghi nhớ ---


def test_remember_tool_saves_and_reports():
    saved = []
    tool = make_remember_tool(lambda fact: (saved.append(fact), True)[1])
    output = tool.func(fact="Người dùng tên Đạt")
    assert saved == ["Người dùng tên Đạt"]
    assert "Đã ghi nhớ" in output


def test_remember_tool_duplicate():
    tool = make_remember_tool(lambda fact: False)
    output = tool.func(fact="trùng lặp")
    assert "đã có" in output.lower()
