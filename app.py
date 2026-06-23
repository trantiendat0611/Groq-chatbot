import streamlit as st

from src.chat_service import ChatService, GenerationSettings
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
from src.prompts import SYSTEM_PROMPT
from src.rag import (
    build_chunks,
    build_rag_system_prompt,
    format_rag_context,
    format_source_label,
    retrieve_chunks_with_fallback,
)


st.set_page_config(
    page_title="Groq Assistant",
    page_icon=None,
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


def reset_rag_index() -> None:
    st.session_state.rag_chunks = []
    st.session_state.rag_file_names = []


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


init_database()
chat_service = get_chat_service()
default_settings = chat_service.default_settings()

if "active_conversation_id" not in st.session_state:
    set_active_chat(ensure_chat_exists())

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPT.strip()

if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []

if "rag_file_names" not in st.session_state:
    st.session_state.rag_file_names = []

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

    with st.expander("Tài liệu RAG", expanded=False):
        rag_enabled = st.checkbox(
            "Dùng tài liệu khi trả lời",
            value=True,
            help="Khi bật, chatbot sẽ tìm đoạn liên quan trong tài liệu đã nạp và đưa vào prompt.",
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
                        st.session_state.rag_chunks = chunks
                        st.session_state.rag_file_names = [name for name, _ in files]
                        st.success(f"Đã nạp {len(chunks)} đoạn từ {len(files)} file.")
                except Exception as exc:
                    st.error(f"Không nạp được tài liệu: {exc}")

        if st.session_state.rag_chunks:
            st.caption(
                f"Đang có {len(st.session_state.rag_chunks)} đoạn từ "
                f"{len(st.session_state.rag_file_names)} file."
            )
            with st.popover("File đã nạp"):
                for file_name in st.session_state.rag_file_names:
                    st.write(file_name)

        st.button(
            "Xóa tài liệu đã nạp",
            key="clear_rag_documents",
            on_click=reset_rag_index,
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
            set_active_chat(ensure_chat_exists())
            st.rerun()


chat_title = get_chat_title(active_conversation_id)
messages = get_chat_messages(active_conversation_id)

st.title("Trợ lý AI")
st.markdown(
    f'<div class="subtitle">Model: {active_settings.model} · '
    "lịch sử chat được lưu bằng SQLite.</div>",
    unsafe_allow_html=True,
)

if not messages:
    st.markdown(
        '<div class="empty-state">Bắt đầu bằng một câu hỏi ở ô chat bên dưới.</div>',
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
        rag_sources = []
        used_rag_fallback = False
        system_prompt_for_request = system_prompt

        if rag_enabled and st.session_state.rag_chunks:
            retrieved_chunks, used_rag_fallback = retrieve_chunks_with_fallback(
                prompt,
                st.session_state.rag_chunks,
                top_k=int(rag_top_k),
            )
            rag_context = format_rag_context(retrieved_chunks)
            system_prompt_for_request = build_rag_system_prompt(system_prompt, rag_context)
            rag_sources = [format_source_label(result.chunk) for result in retrieved_chunks]

        with st.chat_message("assistant"):
            answer = stream_answer(messages_for_llm, system_prompt_for_request, active_settings)
            if rag_sources:
                if used_rag_fallback:
                    st.caption("Không thấy khớp từ khóa trực tiếp; đã dùng các đoạn đầu tài liệu làm ngữ cảnh.")
                st.caption("Nguồn đã dùng: " + "; ".join(rag_sources))

        if chat_title == "Chat mới":
            rename_chat(active_conversation_id, make_title(prompt))

        add_chat_message(active_conversation_id, "user", prompt)
        add_chat_message(active_conversation_id, "assistant", answer)
        st.rerun()
    except Exception as exc:
        st.error(f"Không gọi được Groq API: {exc}")
