import asyncio
from typing import Dict, Any, List, Tuple

from chatbot_modules.config import CONFIG
from chatbot_modules.components import ComponentManager
from chatbot_modules.classification import Classifier
from chatbot_modules.retrieval import RetrievalManager
from chatbot_modules.llm_service import generate_llm_response
from chatbot_modules.prompts import TOPIC_PROMPTS
from chatbot_modules.utils import sanitize_input, format_chat_history_for_llm, format_response
from chatbot_modules.constants import QuestionCategory, OWASPTopic # Explicitly import

class OWASPChatbot:
    def __init__(self):
        self.component_manager = ComponentManager()
        self.classifier = Classifier()
        self.retrieval_manager = RetrievalManager(self.component_manager) # Pass component manager
        self.chat_history: List[Tuple[str, str]] = []

    async def _async_init_components(self):
        """Initialize all components asynchronously."""
        await self.component_manager.init_all()

    async def process_question(self, question: str) -> Dict[str, Any]:
        """Process a question and generate a response with metadata."""
        sanitized_question = sanitize_input(question)
        if not sanitized_question:
            return {"error": "Empty or invalid question after sanitization."}
        question = sanitized_question # Use sanitized question for further processing

        print("\n🔍 Analyzing your question...")

        general_category = await self.classifier.classify_question_category(question)
        print(f"General Category: {general_category.value}")

        specific_owasp_topic = await self.classifier.classify_owasp_topic(question)

        context = await self.retrieval_manager.retrieve_relevant_context(question, general_category, specific_owasp_topic)

        history_for_llm = format_chat_history_for_llm(self.chat_history)

        prompt_template = TOPIC_PROMPTS.get(general_category.name.lower(), TOPIC_PROMPTS["basic_understanding"])

        dynamic_guidance = ""
        if specific_owasp_topic:
            if specific_owasp_topic == OWASPTopic.INJECTION:
                dynamic_guidance = "Focus on common injection types like SQLi, XSS, and command injection, and their code-level implications."
            elif specific_owasp_topic == OWASPTopic.BROKEN_ACCESS_CONTROL:
                dynamic_guidance = "Emphasize authorization bypasses, insecure direct object references (IDOR), and privilege escalation."
            if dynamic_guidance:
                print(f"Applying dynamic prompt guidance for {specific_owasp_topic.name}: {dynamic_guidance}")
                dynamic_guidance = f"Guidance: {dynamic_guidance}\n\n"

        context_disclaimer = ""
        context_found = True
        if "No relevant context found" in context or "Failed to retrieve context" in context or not context.strip():
            context_disclaimer = (
                "Note: I could not find highly specific external information for this query in my knowledge base. "
                "I will attempt to answer based on my general cybersecurity knowledge, but the information may be less detailed or specific. "
            )
            context = "" # Ensure context is empty string for prompt format if no context found
            context_found = False


        formatted_prompt = (
            f"[INST] {prompt_template['prompt']}\n\n"
            f"{dynamic_guidance}"
            f"{history_for_llm}"
            f"{context_disclaimer}"
            f"Context:\n{context}\n\n"
            f"Question: {question} [/INST]"
        )

        print("Generating response...")
        response = await generate_llm_response(
            self.component_manager.llm, # Pass the actual LLM instance
            formatted_prompt,
            temperature=0.7 if general_category in [QuestionCategory.EXAMPLE_SCENARIOS, QuestionCategory.PROACTIVE_SUGGESTIONS] else 0.5
        )

        formatted_response = format_response(response, general_category)

        self.chat_history.append((question, formatted_response))
        if len(self.chat_history) > CONFIG["MAX_CHAT_HISTORY_TURNS"]:
            self.chat_history = self.chat_history[-CONFIG["MAX_CHAT_HISTORY_TURNS"]:]

        return {
            "question": question,
            "category": general_category.value,
            "response": formatted_response,
            "context_used": context_found
        }

    def _print_header(self, text: str, width: int = 60, char: str = '=') -> None:
        """Print a formatted header."""
        print(f"\n{char * width}")
        print(f"{text.upper():^{width}}")
        print(f"{char * width}\n")

    def _print_section(self, title: str, content: str, indent: int = 4) -> None:
        """Print a section with title and indented content."""
        indent_str = ' ' * indent
        print(f"\n\033[1;34m{title}:\033[0m")
        # Split content into paragraphs and print with indentation
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            print(f"{indent_str}{para.strip()}")
        print()

    async def _format_response(self, response: Dict[str, Any]) -> str:
        """Format the chatbot response with proper formatting and handle markdown."""
        formatted = []
        
        # Add response header
        formatted.append(f"\n\033[1;32m{'OWASP SECURITY ASSISTANT':^60}\033[0m")
        formatted.append(f"\033[90m{'='*60}\033[0m")
        
        # Get and clean the response text
        response_text = response.get('response', '').strip()
        if not response_text:
            return "\nNo response generated.\n"
            
        # First, handle code blocks with language specification
        import re
        
        # Process code blocks with language spec
        response_text = re.sub(
            r'```(\w*)\n(.*?)```', 
            lambda m: f'\n\033[1;36mCode ({m.group(1) or "text"}):\033[0m\n{m.group(2).strip()}\n', 
            response_text, 
            flags=re.DOTALL
        )
        
        # Process inline code
        response_text = re.sub(
            r'`([^`]+)`',
            lambda m: f'\033[36m{m.group(1)}\033[0m',
            response_text
        )
        
        # Process lists (both numbered and bulleted)
        response_text = re.sub(
            r'^\s*(\d+\.|[-*+])\s*(.*?)(?=\n\s*\n|$)', 
            lambda m: f"\n  • {m.group(2).strip()}", 
            response_text,
            flags=re.MULTILINE
        )
        
        # Process URLs
        response_text = re.sub(
            r'(https?://[^\s<>"]+)', 
            '\033[4;34m\1\033[0m', 
            response_text
        )
        
        # Process bold and italic text
        response_text = re.sub(r'\*\*(.*?)\*\*', '\033[1m\1\033[0m', response_text)
        response_text = re.sub(r'\*(.*?)\*', '\033[3m\1\033[0m', response_text)
        
        # Clean up any remaining markdown artifacts
        response_text = re.sub(r'^#+\s*', '', response_text, flags=re.MULTILINE)
        
        formatted.append(f"\n{response_text}\n")
        
        # Add context information if available
        context_used = response.get('context_used', '')
        if context_used and isinstance(context_used, str) and len(context_used) > 10:
            display_text = f"Context used: {context_used[:100]}..." if len(context_used) > 100 else str(context_used)
            formatted.append(f"\n\033[90m{display_text}\033[0m")
        
        formatted.append(f"\033[90m{'='*60}\033[0m")
        return '\n'.join(formatted)

    async def run_chat_session(self):
        """Run an interactive chat session with the OWASP chatbot."""
        self._print_header("OWASP Security Chatbot")
        print("Welcome! I'm your OWASP security assistant. Ask me anything about web application security.")
        print("Type 'exit' or 'quit' to end the session.\n")
        print("\033[3mTip: Try asking about OWASP Top 10 vulnerabilities, prevention methods, or request examples.\033[0m\n")

        try:
            while True:
                try:
                    user_input = input("\n\033[1mYou:\033[0m ").strip()

                    if user_input.lower() in ['exit', 'quit']:
                        print("\n\033[1;32mThank you for using OWASP Security Assistant. Stay secure! 👋\033[0m")
                        break

                    if not user_input:
                        print("\033[91mPlease enter a valid question.\033[0m")
                        continue

                    print("\n\033[3mAnalyzing your question...\033[0m")
                    response = await self.process_question(user_input)
                    
                    # Print the formatted response
                    formatted_response = await self._format_response(response)
                    print(formatted_response)

                except KeyboardInterrupt:
                    print("\n\n\033[1;33mSession interrupted. Type 'exit' or press Ctrl+C again to quit.\033[0m")
                    continue
                except Exception as e:
                    print(f"\n\033[91mAn error occurred: {str(e)}\033[0m")
                    continue

        except KeyboardInterrupt:
            print("\n\n\033[1;33mSession ended by user. Goodbye! 👋\033[0m")
        except Exception as e:
            print(f"\n\033[91mAn unexpected error occurred: {str(e)}\033[0m")