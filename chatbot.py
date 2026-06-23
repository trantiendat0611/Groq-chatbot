from src.chat_service import ChatService


def main() -> None:
    chat_service = ChatService()
    messages = []

    print("Groq Chatbot đã sẵn sàng. Gõ 'thoát', 'exit' hoặc 'quit' để dừng.\n")

    while True:
        user_input = input("Bạn: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"thoát", "exit", "quit"}:
            print("Bot: Hẹn gặp lại bạn!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            answer = chat_service.generate_reply(messages)
        except Exception as exc:
            print(f"Bot: Có lỗi khi gọi Groq API: {exc}")
            messages.pop()
            continue

        print(f"Bot: {answer}\n")

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
