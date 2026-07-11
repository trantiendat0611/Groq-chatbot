import streamlit as st

from src import user_memory, vector_store
from src.agent import AgentService
from src.chat_service import ChatService, ChatServiceError, GenerationSettings
from src.embeddings import get_default_embedder, get_ready_embedder
from src.memory import (
    add_chat_message,
    chat_exists,
    create_chat,
    delete_chat,
    ensure_chat_exists,
    get_chat_messages,
    get_chat_title,
    init_database,
    list_chats,
    rename_chat,
)
from src.prompts import AGENT_PROMPT_SUFFIX, SYSTEM_PROMPT
from src.rag import (
    build_chunks,
    build_rag_system_prompt,
    format_rag_context,
    format_source_label,
)
from src.tools import (
    Tool,
    make_calculator_tool,
    make_document_search_tool,
    make_remember_tool,
    make_time_tool,
    make_web_search_tool,
)


st.set_page_config(
    page_title="Groq Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background: #050505;
        color: #f5f5f5;
    }

    [data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid #262626;
    }

    [data-testid="stSidebar"] * {
        color: #f5f5f5;
    }

    [data-testid="stHeader"] {
        background: #050505;
    }

    [data-testid="stToolbar"] {
        opacity: 0.9;
    }

    [data-testid="collapsedControl"] {
        color: #f5f5f5;
    }

    footer {
        display: none;
    }

    .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    h1 {
        font-size: 1.8rem;
        font-weight: 650;
        letter-spacing: 0;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        color: #a3a3a3;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    [data-testid="stChatMessage"] {
        background: #111111;
        border: 1px solid #262626;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.85rem;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #f5f5f5;
        color: #050505;
        border-color: #f5f5f5;
    }

    [data-testid="stChatInput"] {
        background: #050505;
    }

    [data-testid="stChatInput"] textarea {
        background: #111111;
        color: #f5f5f5;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
    }

    .stButton > button {
        width: 100%;
        background: #111111;
        color: #f5f5f5;
        border: 1px solid #303030;
        border-radius: 8px;
        min-height: 2.4rem;
    }

    .stButton > button:hover {
        background: #f5f5f5;
        color: #050505;
        border-color: #f5f5f5;
    }

    .conversation-title {
        color: #f5f5f5;
        font-size: 0.92rem;
        font-weight: 600;
        margin: 1rem 0 0.4rem;
    }

    .empty-state {
        border: 1px solid #262626;
        border-radius: 8px;
        padding: 1.25rem;
        color: #c7c7c7;
        background: #0d0d0d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


TOOL_LABELS = {
    "calculator": "Máy tính",
    "get_current_time": "Xem giờ hiện tại",
    "web_search": "Tìm kiếm web",
    "search_documents": "Tra cứu tài liệu",
    "remember": "Ghi nhớ",
}


@st.cache_resource
def get_chat_service() -> ChatService:
    return ChatService()


def make_title(message: str) -> str:
    title = " ".join(message.strip().split())
    if len(title) > 36:
        return f"{title[:33]}..."
    return title or "Chat mới"


def set_active_chat(conversation_id: int) -> None:
    st.session_state.active_conversation_id = conversation_id


def reset_system_prompt() -> None:
    st.session_state.system_prompt = SYSTEM_PROMPT.strip()


def build_settings(
    model: str,
    temperature: float,
    max_tokens: int,
    fallback_model: str,
) -> GenerationSettings:
    return GenerationSettings(
        model=model.strip() or fallback_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def make_document_searcher(conversation_id: int):
    """Tạo hàm tra cứu tài liệu giới hạn trong MỘT đoạn chat."""

    def run_document_search(query: str, top_k: int) -> str:
        results, _method = vector_store.search(
            query,
            top_k=top_k,
            embedder=get_default_embedder(),
            conversation_id=conversation_id,
        )
        if not results:
            return "Không tìm thấy đoạn nào liên quan trong tài liệu đã nạp."
        return format_rag_context(results)

    return run_document_search


def save_user_fact(fact: str) -> bool:
    # Không tải model giữa câu chat; memory chưa có embedding vẫn được nhớ lại.
    embedder = get_ready_embedder()
    embedding = embedder.embed_query(fact) if embedder else None
    return user_memory.add_memory(fact, embedding=embedding) is not None


def build_agent_tools(
    conversation_id: int,
    include_documents: bool,
    include_memory: bool,
    top_k: int,
) -> list[Tool]:
    tools = [
        make_calculator_tool(),
        make_time_tool(),
        make_web_search_tool(),
    ]
    # Tài liệu gắn theo đoạn chat: agent chỉ thấy file nạp trong chat này.
    if include_documents and vector_store.count_chunks(conversation_id=conversation_id) > 0:
        tools.append(
            make_document_search_tool(
                make_document_searcher(conversation_id), top_k=top_k
            )
        )
    if include_memory:
        tools.append(make_remember_tool(save_user_fact))
    return tools


def stream_answer(
    messages_for_llm: list[dict[str, str]],
    system_prompt: str,
    settings: GenerationSettings,
) -> str:
    answer_parts = []
    placeholder = st.empty()

    for chunk in get_chat_service().stream_reply(
        messages_for_llm,
        system_prompt=system_prompt,
        settings=settings,
    ):
        answer_parts.append(chunk)
        placeholder.markdown("".join(answer_parts))

    return "".join(answer_parts)


def shorten(text: str, limit: int = 700) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (rút gọn)"


def render_agent_answer(
    agent_service: AgentService,
    messages_for_llm: list[dict[str, str]],
    system_prompt: str,
    settings: GenerationSettings,
) -> str:
    """Hiển thị tiến trình agent: các bước dùng công cụ + câu trả lời streaming."""
    answer_parts: list[str] = []
    tool_area = st.container()
    placeholder = st.empty()
    current_status = None

    for event in agent_service.run_stream(
        messages_for_llm,
        system_prompt=system_prompt,
        settings=settings,
    ):
        if event.type == "text":
            answer_parts.append(event.content)
            placeholder.markdown("".join(answer_parts))
        elif event.type == "tool_call":
            label = TOOL_LABELS.get(event.tool_name, event.tool_name)
            with tool_area:
                current_status = st.status(f"Đang dùng công cụ: {label}...", expanded=False)
                with current_status:
                    st.code(event.arguments, language="json")
        elif event.type == "tool_result" and current_status is not None:
            label = TOOL_LABELS.get(event.tool_name, event.tool_name)
            with current_status:
                st.text(shorten(event.content))
            current_status.update(label=f"Đã dùng: {label}", state="complete")

    return "".join(answer_parts)


init_database()
chat_service = get_chat_service()
default_settings = chat_service.default_settings()

if "active_conversation_id" not in st.session_state:
    set_active_chat(ensure_chat_exists())

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPT.strip()

if not chat_exists(st.session_state.active_conversation_id):
    set_active_chat(ensure_chat_exists())

active_conversation_id = st.session_state.active_conversation_id
conversations = list_chats()


with st.sidebar:
    st.title("Groq Assistant")

    with st.expander("Cài đặt", expanded=False):
        model = st.text_input(
            "Model",
            value=default_settings.model,
            help="Tên model Groq. Nếu model không tồn tại trong tài khoản của bạn, Groq sẽ báo lỗi.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(default_settings.temperature),
            step=0.1,
            help="Thấp hơn: ổn định hơn. Cao hơn: sáng tạo hơn.",
        )
        max_tokens = st.number_input(
            "Max tokens",
            min_value=64,
            max_value=4096,
            value=int(default_settings.max_tokens),
            step=64,
            help="Giới hạn độ dài câu trả lời.",
        )
        agent_enabled = st.checkbox(
            "Chế độ Agent (dùng công cụ)",
            value=True,
            help=(
                "Cho phép AI tự quyết định dùng máy tính, tìm kiếm web, "
                "tra cứu tài liệu đã nạp và ghi nhớ thông tin về bạn."
            ),
        )
        system_prompt = st.text_area(
            "System prompt",
            key="system_prompt",
            height=150,
            help="Chỉ dẫn nền cho trợ lý AI.",
        ).strip()

        st.button(
            "Khôi phục prompt mặc định",
            key="reset_system_prompt",
            on_click=reset_system_prompt,
        )

    active_settings = build_settings(
        model,
        temperature,
        int(max_tokens),
        fallback_model=default_settings.model,
    )

    with st.expander("Tài liệu của đoạn chat", expanded=False):
        st.caption(
            "Tài liệu gắn riêng với từng đoạn chat — AI chỉ đọc được file "
            "đã nạp trong đoạn chat đang mở."
        )
        rag_enabled = st.checkbox(
            "Dùng tài liệu khi trả lời",
            value=True,
            help=(
                "Chế độ Agent: AI tự tra cứu tài liệu qua công cụ search_documents. "
                "Chế độ thường: các đoạn liên quan được chèn thẳng vào prompt."
            ),
        )
        rag_top_k = st.slider(
            "Số đoạn lấy vào prompt",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
        )
        uploaded_files = st.file_uploader(
            "Upload TXT, Markdown hoặc PDF",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )

        if st.button("Nạp tài liệu", key="index_documents"):
            if not uploaded_files:
                st.warning("Hãy chọn ít nhất một tài liệu trước.")
            else:
                try:
                    files = [(file.name, file.getvalue()) for file in uploaded_files]
                    chunks = build_chunks(files)

                    if not chunks:
                        st.warning("Không trích xuất được nội dung từ tài liệu đã chọn.")
                    else:
                        with st.spinner(
                            "Đang tạo embedding... (lần đầu sẽ tải model ~470MB, hãy kiên nhẫn)"
                        ):
                            embedder = get_default_embedder()
                            embeddings_list = (
                                embedder.embed_texts([chunk.text for chunk in chunks])
                                if embedder
                                else None
                            )

                        # Nạp lại cùng file trong cùng đoạn chat thì thay thế bản cũ.
                        for file_name, _ in files:
                            vector_store.delete_source(
                                file_name, conversation_id=active_conversation_id
                            )
                        vector_store.add_chunks(
                            chunks,
                            embeddings_list,
                            conversation_id=active_conversation_id,
                        )

                        if embedder is None:
                            st.info(
                                "Không dùng được model embedding — tài liệu sẽ được "
                                "tìm kiếm bằng từ khóa."
                            )
                        st.success(
                            f"Đã nạp {len(chunks)} đoạn từ {len(files)} file "
                            "vào đoạn chat này."
                        )
                except Exception as exc:
                    st.error(f"Không nạp được tài liệu: {exc}")

        indexed_sources = vector_store.list_sources(
            conversation_id=active_conversation_id
        )
        if indexed_sources:
            total_chunks = vector_store.count_chunks(
                conversation_id=active_conversation_id
            )
            st.caption(
                f"Đoạn chat này có {total_chunks} đoạn từ {len(indexed_sources)} file."
            )
            with st.popover("File đã nạp"):
                for source_name, chunk_count in indexed_sources:
                    st.write(f"{source_name} ({chunk_count} đoạn)")

            if st.button("Xóa tài liệu của đoạn chat", key="clear_rag_documents"):
                vector_store.clear_store(conversation_id=active_conversation_id)
                st.rerun()

    with st.expander("Trí nhớ dài hạn", expanded=False):
        memory_enabled = st.checkbox(
            "Dùng trí nhớ dài hạn",
            value=True,
            help=(
                "AI tự ghi nhớ những điều quan trọng bạn chia sẻ (tên, sở thích, "
                "dự án...) và nhớ lại trong các cuộc chat sau."
            ),
        )
        stored_memories = user_memory.list_memories()
        if stored_memories:
            st.caption(f"Đang nhớ {len(stored_memories)} điều về bạn:")
            for memory in stored_memories[:20]:
                memory_column, delete_column = st.columns([4, 1])
                memory_column.caption(f"• {memory.content}")
                if delete_column.button("X", key=f"delete_memory_{memory.id}"):
                    user_memory.delete_memory(memory.id)
                    st.rerun()

            if st.button("Xóa toàn bộ trí nhớ", key="clear_memories"):
                user_memory.clear_memories()
                st.rerun()
        else:
            st.caption(
                "Chưa có gì trong trí nhớ. Hãy chia sẻ thông tin về bạn hoặc nói "
                "\"hãy nhớ rằng...\" khi chat."
            )

    if st.button("+ Chat mới", key="new_chat"):
        set_active_chat(create_chat())
        st.rerun()

    st.markdown('<div class="conversation-title">Lịch sử chat</div>', unsafe_allow_html=True)

    for conversation in conversations:
        label = conversation.title
        if conversation.id == active_conversation_id:
            label = f"> {label}"

        if st.button(label, key=f"conversation_{conversation.id}"):
            set_active_chat(conversation.id)
            st.rerun()

    if conversations:
        st.divider()
        if st.button("Xóa chat hiện tại", key="delete_current_chat"):
            delete_chat(active_conversation_id)
            # Tài liệu gắn với đoạn chat này cũng đi theo.
            vector_store.clear_store(conversation_id=active_conversation_id)
            set_active_chat(ensure_chat_exists())
            st.rerun()


chat_title = get_chat_title(active_conversation_id)
messages = get_chat_messages(active_conversation_id)

st.title("Trợ lý AI")
mode_label = "Agent (có công cụ)" if agent_enabled else "Chat thường"
st.markdown(
    f'<div class="subtitle">Model: {active_settings.model} · Chế độ: {mode_label} · '
    "lịch sử chat được lưu bằng SQLite.</div>",
    unsafe_allow_html=True,
)

if not messages:
    st.markdown(
        '<div class="empty-state">Bắt đầu bằng một câu hỏi ở ô chat bên dưới. '
        "Ở chế độ Agent, AI có thể tự tính toán, tìm kiếm web, tra cứu tài liệu "
        "bạn đã nạp và ghi nhớ thông tin quan trọng.</div>",
        unsafe_allow_html=True,
    )

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Nhập câu hỏi của bạn...")

if prompt:
    prompt = prompt.strip()

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_for_llm = [
        *messages,
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        system_prompt_for_request = system_prompt

        # Trí nhớ dài hạn: nhớ lại những điều liên quan tới câu hỏi.
        # Chỉ dùng embedding nếu model đã nạp sẵn (không tải model giữa câu chat).
        if memory_enabled:
            ready_embedder = get_ready_embedder()
            recall_embedding = (
                ready_embedder.embed_query(prompt) if ready_embedder else None
            )
            recalled = user_memory.search_memories(
                query_embedding=recall_embedding, top_k=5
            )
            memories_block = user_memory.format_memories_block(recalled)
            if memories_block:
                system_prompt_for_request = (
                    f"{system_prompt_for_request}\n\n{memories_block}"
                )

        if agent_enabled:
            system_prompt_for_request = (
                f"{system_prompt_for_request}\n\n{AGENT_PROMPT_SUFFIX.strip()}"
            )
            agent_tools = build_agent_tools(
                active_conversation_id,
                include_documents=rag_enabled,
                include_memory=memory_enabled,
                top_k=int(rag_top_k),
            )
            agent_service = AgentService(get_chat_service(), tools=agent_tools)

            with st.chat_message("assistant"):
                answer = render_agent_answer(
                    agent_service,
                    messages_for_llm,
                    system_prompt_for_request,
                    active_settings,
                )
        else:
            rag_sources = []
            search_method = None

            if rag_enabled and vector_store.count_chunks(
                conversation_id=active_conversation_id
            ) > 0:
                retrieved, search_method = vector_store.search(
                    prompt,
                    top_k=int(rag_top_k),
                    embedder=get_default_embedder(),
                    conversation_id=active_conversation_id,
                )
                if retrieved:
                    rag_context = format_rag_context(retrieved)
                    system_prompt_for_request = build_rag_system_prompt(
                        system_prompt_for_request, rag_context
                    )
                    rag_sources = [
                        format_source_label(result.chunk) for result in retrieved
                    ]

            with st.chat_message("assistant"):
                answer = stream_answer(
                    messages_for_llm, system_prompt_for_request, active_settings
                )
                if rag_sources:
                    method_label = (
                        "ngữ nghĩa" if search_method == "semantic" else "từ khóa"
                    )
                    st.caption(
                        f"Nguồn đã dùng (tìm kiếm {method_label}): "
                        + "; ".join(rag_sources)
                    )

        if not answer.strip():
            answer = "(Không nhận được nội dung trả lời. Hãy thử lại.)"

        if chat_title == "Chat mới":
            rename_chat(active_conversation_id, make_title(prompt))

        add_chat_message(active_conversation_id, "user", prompt)
        add_chat_message(active_conversation_id, "assistant", answer)
        st.rerun()
    except ChatServiceError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Lỗi không mong đợi: {exc}")
