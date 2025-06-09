import asyncio
from chatbot_modules.chatbot import OWASPChatbot # Adjusted import statement

async def run_interactive_chatbot():
    """Run an interactive terminal chat session with the OWASP Chatbot."""
    chatbot = OWASPChatbot()
    await chatbot._async_init_components() # Initialize components
    await chatbot.run_chat_session()

if __name__ == "__main__":
    asyncio.run(run_interactive_chatbot())