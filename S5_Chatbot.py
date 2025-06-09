import os
import re
import json
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import torch
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import networkx as nx
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT"),
    "PINECONE_INDEX_NAME": "owasp-qa",
    "VECTOR_DIMENSION": 768,
    "EMBEDDING_MODEL_PATH": './fine_tuned_owasp_model_advanced',
    "LOCAL_MODEL_DIR": "./pretrained_language_model",
    "LLM_FILENAME": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "KNOWLEDGE_GRAPH_PATH": "./security_knowledge_graph.gml",
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
    "EMBEDDING_DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
    "LLM_DEVICE": 'cuda' if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9 else 'cpu',
    "LLM_GPU_LAYERS": -1 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9 else 0,
    "LLM_CTX_SIZE": 4096,
    "BATCH_SIZE": 8,
    "MAX_RETRIES": 3,
    "TEMPERATURE": 0.7,
    "TOP_P": 0.9,
    "TOP_K": 40,
    "MAX_CHAT_HISTORY_TURNS": 3
}

# Create directories if they don't exist
os.makedirs(CONFIG["LOCAL_MODEL_DIR"], exist_ok=True)

# Question categories
class QuestionCategory(Enum):
    BASIC_UNDERSTANDING = "📘 Basic understanding questions"
    TECHNICAL_EXPLANATION = "🔍 Technical explanation"
    VULNERABILITY_ID = "⚠ Vulnerability identification"
    PREVENTION_METHODS = "🛡 Prevention methods"
    EXAMPLE_SCENARIOS = "💥 Example scenarios"
    REFERENCES = "🔗 References"
    STATISTICS = "📊 Statistics"
    PROACTIVE_SUGGESTIONS = "🧠 Proactive suggestions"

# OWASP Top 10 topics for direct mapping to namespaces
class OWASPTopic(Enum):
    BROKEN_ACCESS_CONTROL = "A01_2021_Broken_Access_Control"
    CRYPTOGRAPHIC_FAILURES = "A02_2021_Cryptographic_Failures"
    INJECTION = "A03_2021_Injection"
    INSECURE_DESIGN = "A04_2021_Insecure_Design"
    SECURITY_MISCONFIGURATION = "A05_2021_Security_Misconfiguration"
    VULNERABLE_COMPONENTS = "A06_2021_Vulnerable_Components"
    IDENTIFICATION_AUTHENTICATION_FAILURES = "A07_2021_Identification_Authentication_Failures"
    SOFTWARE_INTEGRITY_FAILURES = "A08_2021_Software_Integrity_Failures"
    SECURITY_LOGGING_MONITORING_FAILURES = "A09_2021_Security_Logging_Monitoring_Failures"
    SERVER_SIDE_REQUEST_FORGERY = "A10_2021_Server_Side_Request_ForT_Forgery"

@dataclass
class QueryContext:
    category: QuestionCategory
    requires_technical_depth: bool
    needs_examples: bool
    requires_references: bool

# Define default/fallback prompts first
PROMPT_TEMPLATES_FILE_PATH = os.path.join(os.getcwd(), 'Prompt_templates', 'prompt_templates.json')

DEFAULT_TOPIC_PROMPTS = {
    "category_classifier": {
        "prompt": """Analyze the following question and determine its category. 
        Categories:
        1. 📘 Basic understanding - General questions about concepts
        2. 🔍 Technical explanation - In-depth technical details
        3. ⚠ Vulnerability identification - Identifying specific vulnerabilities
        4. 🛡 Prevention methods - How to prevent security issues
        5. 💥 Example scenarios - Real-world examples or case studies
        6. 🔗 References - Requests for sources or citations
        7. 📊 Statistics - Requests for data or metrics
        8. 🧠 Proactive suggestions - Recommendations not explicitly asked for
        
        Return only the category number (1-8) and nothing else.
        
        Question: {question}"""
    },
    "owasp_topic_classifier": {
        "prompt": """Analyze the following question and determine if it is directly related to one of the OWASP Top 10 2021 categories.
        Categories (return only the specific OWASP ID, e.g., A01, A02, A03, etc.):
        A01: Broken Access Control
        A02: Cryptographic Failures
        A03: Injection
        A04: Insecure Design
        A05: Security Misconfiguration
        A06: Vulnerable and Outdated Components
        A07: Identification and Authentication Failures
        A08: Software and Data Integrity Failures
        A09: Security Logging and Monitoring Failures
        A10: Server-Side Request Forgery (SSRF)

        If the question is clearly about one of these, return ONLY its ID (e.g., A03).
        If it's a general security question not directly tied to a single OWASP Top 10 category, return 'GENERAL'.
        
        Question: {question}"""
    },
    **{category.name.lower(): {
        "prompt": f"You are a cybersecurity expert. {category.value}. {prompt}"
    } for category, prompt in [
        (QuestionCategory.BASIC_UNDERSTANDING, 
         "Provide a clear, concise explanation suitable for beginners. Include key points and simple analogies."),
        (QuestionCategory.TECHNICAL_EXPLANATION,
         "Provide a detailed technical explanation. Include relevant protocols, standards, and technical specifications."),
        (QuestionCategory.VULNERABILITY_ID,
         "Identify and explain the vulnerability. Include CVE references if available."),
        (QuestionCategory.PREVENTION_METHODS,
         "List and explain prevention methods. Include implementation details and best practices."),
        (QuestionCategory.EXAMPLE_SCENARIOS,
         "Provide real-world examples or case studies. Include what happened and lessons learned."),
        (QuestionCategory.REFERENCES,
         "Provide authoritative references, standards, and sources. Include links if available."),
        (QuestionCategory.STATISTICS,
         "Provide relevant statistics and data. Include sources and timeframes."),
        (QuestionCategory.PROACTIVE_SUGGESTIONS,
         "Provide proactive security recommendations. Include risk assessment and implementation priority.")
    ]}
}

TOPIC_PROMPTS = DEFAULT_TOPIC_PROMPTS # Start with fallback

try:
    with open(PROMPT_TEMPLATES_FILE_PATH) as f:
        loaded_prompts = json.load(f)
        TOPIC_PROMPTS.update(loaded_prompts) 
    print("✅ prompt_templates.json loaded successfully.")
except FileNotFoundError:
    print(f"Warning: prompt_templates.json not found at {PROMPT_TEMPLATES_FILE_PATH}. Using fallback prompts.")
except json.JSONDecodeError as e:
    print(f"Error decoding prompt_templates.json at {PROMPT_TEMPLATES_FILE_PATH}: {e}. Using fallback prompts.")


class OWASPChatbot:
    def __init__(self):
        self.components = {
            'pc': None,
            'index': None,
            'embedding_model': None,
            'llm': None,
            'knowledge_graph': None,
            'category_cache': {}
        }
        self.chat_history: List[Tuple[str, str]] = []

    async def _async_init_components(self):
        """Asynchronously initialize components for better performance."""
        print("Initializing components...")
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
        
        tasks = [
            self.initialize_pinecone(),
            self.initialize_embedding_model(),
            self.initialize_llm(),
            self.initialize_knowledge_graph()
        ]
        await asyncio.gather(*tasks)
        print("\n✅ All components initialized successfully!\n")

    async def initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            self.components['pc'] = Pinecone(api_key=CONFIG["PINECONE_API_KEY"])
            
            if CONFIG["PINECONE_INDEX_NAME"] not in [index.name for index in self.components['pc'].list_indexes()]:
                print(f"Creating new Pinecone index: {CONFIG['PINPINE_INDEX_NAME']}")
                self.components['pc'].create_index(
                    name=CONFIG["PINECONE_INDEX_NAME"],
                    dimension=CONFIG["VECTOR_DIMENSION"],
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            
            self.components['index'] = self.components['pc'].Index(CONFIG["PINECONE_INDEX_NAME"])
            print("✅ Pinecone initialized")
        except Exception as e:
            print(f"❌ Error initializing Pinecone: {e}")

    async def initialize_embedding_model(self):
        """Initialize the embedding model with GPU optimization."""
        try:
            if os.path.exists(CONFIG["EMBEDDING_MODEL_PATH"]):
                self.components['embedding_model'] = SentenceTransformer(
                    CONFIG["EMBEDDING_MODEL_PATH"], 
                    device=CONFIG["EMBEDDING_DEVICE"]
                )
                self.components['embedding_model'].max_seq_length = 512
                self.components['embedding_model'].eval()
                print("✅ Embedding model loaded")
            else:
                print(f"❌ Embedding model not found at: {CONFIG['EMBEDDING_MODEL_PATH']}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")

    async def initialize_llm(self):
        """Initialize the LLM with GPU optimization if available."""
        try:
            llm_path = os.path.join(CONFIG["LOCAL_MODEL_DIR"], CONFIG["LLM_FILENAME"])
            
            if not os.path.exists(llm_path):
                print(f"Downloading {CONFIG['LLM_FILENAME']}...")
                hf_hub_download(
                    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                    filename=CONFIG["LLM_FILENAME"],
                    local_dir=CONFIG["LOCAL_MODEL_DIR"],
                    local_dir_use_symlinks=False
                )
            
            self.components['llm'] = Llama(
                model_path=llm_path,
                n_gpu_layers=CONFIG["LLM_GPU_LAYERS"],
                n_ctx=CONFIG["LLM_CTX_SIZE"],
                n_batch=CONFIG["BATCH_SIZE"],
                n_threads=os.cpu_count() // 2 if CONFIG["LLM_DEVICE"] == 'cpu' else 0,
                verbose=False
            )
            print("✅ LLM loaded")
        except Exception as e:
            print(f"❌ Error loading LLM: {e}")

    async def initialize_knowledge_graph(self):
        """Initialize the knowledge graph if available."""
        try:
            if os.path.exists(CONFIG["KNOWLEDGE_GRAPH_PATH"]):
                self.components['knowledge_graph'] = nx.read_gml(CONFIG["KNOWLEDGE_GRAPH_PATH"])
                print(f"✅ Knowledge Graph loaded with {self.components['knowledge_graph'].number_of_nodes()} nodes")
            else:
                print(f"⚠ Knowledge Graph file not found at: {CONFIG['KNOWLEDGE_GRAPH_PATH']}")
        except Exception as e:
            print(f"❌ Error loading knowledge graph: {e}")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embeddings with batching and error handling."""
        if not self.components['embedding_model']:
            return None
            
        try:
            with torch.no_grad():
                return self.components['embedding_model'].encode(
                    text,
                    batch_size=CONFIG["BATCH_SIZE"],
                    show_progress_bar=False,
                    convert_to_numpy=True
                ).tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    async def classify_question_category(self, question: str) -> QuestionCategory:
        """Classify the question into one of the predefined general categories."""
        if question in self.components['category_cache']:
            return self.components['category_cache'][question]
            
        try:
            prompt = TOPIC_PROMPTS["category_classifier"]["prompt"].format(question=question)
            response = await self.generate_llm_response(prompt, max_tokens=10, temperature=0.3)
            
            match = re.search(r'\d+', response)
            if match:
                category_num = int(match.group()) - 1
                if 0 <= category_num < len(QuestionCategory):
                    category = list(QuestionCategory)[category_num]
                    self.components['category_cache'][question] = category
                    return category
            return await self.fallback_classification(question)
        except Exception as e:
            print(f"Error in general category classification: {e}")
            return await self.fallback_classification(question)

    async def _classify_owasp_topic(self, question: str) -> Optional[OWASPTopic]:
        """
        Classify the question to a specific OWASP Top 10 topic (A01, A02, etc.).
        Returns the OWASPTopic Enum member or None if not directly classifiable.
        """
        try:
            prompt = TOPIC_PROMPTS["owasp_topic_classifier"]["prompt"].format(question=question)
            response = await self.generate_llm_response(prompt, max_tokens=10, temperature=0.3)
            
            response_clean = response.strip().upper()

            for topic in OWASPTopic:
                if topic.value.startswith(response_clean) or topic.value.split('_')[0] == response_clean:
                    return topic
            
            return None
        except Exception as e:
            print(f"Error in OWASP topic classification: {e}")
            return None

    async def fallback_classification(self, question: str) -> QuestionCategory:
        """Fallback classification using keyword matching for general categories."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["what is", "what are", "define", "explain", "meaning of"]):
            return QuestionCategory.BASIC_UNDERSTANDING
        elif any(word in question_lower for word in ["how to prevent", "how to secure", "prevent", "mitigate"]):
            return QuestionCategory.PREVENTION_METHODS
        elif any(word in question_lower for word in ["example", "case study", "real world"]):
            return QuestionCategory.EXAMPLE_SCENARIOS
        elif any(word in question_lower for word in ["reference", "source", "citation"]):
            return QuestionCategory.REFERENCES
        elif any(word in question_lower for word in ["statistic", "data", "how many", "how much"]):
            return QuestionCategory.STATISTICS
        elif any(word in question_lower for word in ["vulnerability", "exploit", "attack", "compromise", "weakness"]):
            return QuestionCategory.VULNERABILITY_ID
        elif any(word in question_lower for word in ["proactive", "suggest", "recommend", "best practice"]):
            return QuestionCategory.PROACTIVE_SUGGESTIONS
        else:
            return QuestionCategory.TECHNICAL_EXPLANATION

    async def retrieve_relevant_context(self, question: str, general_category: QuestionCategory) -> str:
        """Retrieve context from Pinecone with enhanced category filtering."""
        if not self.components['index']:
            return ""

        try:
            embedding = self.get_embedding(question)
            if not embedding:
                return ""

            specific_owasp_topic = await self._classify_owasp_topic(question)
            
            namespaces_to_search: List[str] = []

            if specific_owasp_topic:
                print(f"Detected specific OWASP topic: {specific_owasp_topic.value}. Searching only this namespace.")
                namespaces_to_search = [specific_owasp_topic.value]
            else:
                print(f"No specific OWASP topic detected. Using general category namespaces for: {general_category.value}")
                namespaces_to_search = self.get_relevant_namespaces(general_category)

            kg_info = ""
            if self.components['knowledge_graph']:
                question_words = set(q.lower() for q in re.findall(r'\b\w+\b', question))
                found_nodes = [
                    node for node in self.components['knowledge_graph'].nodes()
                    if node.lower() in question_words
                ]
                if found_nodes:
                    kg_info = f"Knowledge Graph related terms: {', '.join(found_nodes)}. "

            all_matches = []
            for namespace in namespaces_to_search:
                try:
                    pinecone_filter = None
                    if not specific_owasp_topic and namespace != "general":
                         pinecone_filter = {"category": {"$eq": general_category.name}}

                    results = self.components['index'].query(
                        vector=embedding,
                        top_k=5,
                        include_metadata=True,
                        namespace=namespace,
                        filter=pinecone_filter
                    )
                    all_matches.extend(results.matches)
                except Exception as e:
                    print(f"Error querying namespace {namespace}: {e}")

            all_matches.sort(key=lambda x: x.score, reverse=True)
            
            context_parts = []
            if kg_info:
                context_parts.append(f"Knowledge Graph Context: {kg_info}")
                context_parts.append("---")

            for match in all_matches[:5]:
                metadata = match.metadata or {}
                context_parts.append(f"Source: {metadata.get('source', 'Unknown')}")
                context_parts.append(f"Content: {metadata.get('text', '')}")
                context_parts.append(f"Relevance: {match.score:.2f}")
                context_parts.append("---")

            return "\n".join(context_parts) if context_parts else "No relevant context found."

        except Exception as e:
            print(f"Error in context retrieval: {e}")
            return ""

    def get_relevant_namespaces(self, general_category: QuestionCategory) -> List[str]:
        """Get relevant Pinecone namespaces based on general question category."""
        category_to_namespaces = {
            QuestionCategory.BASIC_UNDERSTANDING: [
                "A01_2021_Broken_Access_Control",
                "A02_2021_Cryptographic_Failures",
                "A03_2021_Injection",
                "A04_2021_Insecure_Design",
                "A05_2021_Security_Misconfiguration",
                "A06_2021_Vulnerable_Components",
                "A07_2021_Identification_Authentication_Failures",
                "A08_2021_Software_Integrity_Failures",
                "A09_2021_Security_Logging_Monitoring_Failures",
                "A10_2021_Server_Side_Request_ForT_Forgery"
            ],
            
            QuestionCategory.TECHNICAL_EXPLANATION: [
                "A01_2021_Broken_Access_Control",
                "A02_2021_Cryptographic_Failures",
                "A03_2021_Injection",
                "A04_2021_Insecure_Design",
                "A05_2021_Security_Misconfiguration",
                "A06_2021_Vulnerable_Components",
                "A07_2021_Identification_Authentication_Failures",
                "A08_2021_Software_Integrity_Failures",
                "A09_2021_Security_Logging_Monitoring_Failures",
                "A10_2021_Server_Side_Request_ForT_Forgery"
            ],
            
            QuestionCategory.VULNERABILITY_ID: [
                "A01_2021_Broken_Access_Control",
                "A02_2021_Cryptographic_Failures",
                "A03_2021_Injection",
                "A04_2021_Insecure_Design",
                "A05_2021_Security_Misconfiguration",
                "A06_2021_Vulnerable_Components",
                "A07_2021_Identification_Authentication_Failures",
                "A08_2021_Software_Integrity_Failures",
                "A09_2021_Security_Logging_Monitoring_Failures",
                "A10_2021_Server_Side_Request_ForT_Forgery"
            ],
            
            QuestionCategory.PREVENTION_METHODS: [
                "A01_2021_Broken_Access_Control",
                "A02_2021_Cryptographic_Failures",
                "A03_2021_Injection",
                "A04_2021_Insecure_Design",
                "A05_2021_Security_Misconfiguration",
                "A06_2021_Vulnerable_Components",
                "A07_2021_Identification_Authentication_Failures",
                "A08_2021_Software_Integrity_Failures",
                "A09_2021_Security_Logging_Monitoring_Failures",
                "A10_2021_Server_Side_Request_ForT_Forgery"
            ],
            
            QuestionCategory.EXAMPLE_SCENARIOS: [
                "A01_2021_Broken_Access_Control",
                "A02_2021_Cryptographic_Failures",
                "A03_2021_Injection",
                "A04_2021_Insecure_Design",
                "A05_2021_Security_Misconfiguration",
                "A06_2021_Vulnerable_Components",
                "A07_2021_Identification_Authentication_Failures",
                "A08_2021_Software_Integrity_Failures",
                "A09_2021_Security_Logging_Monitoring_Failures",
                "A10_2021_Server_Side_Request_ForT_Forgery"
            ],
            
            QuestionCategory.REFERENCES: [
                "A01_2021_Broken_Access_Control",
                "A02_2021_Cryptographic_Failures",
                "A03_2021_Injection",
                "A04_2021_Insecure_Design",
                "A05_2021_Security_Misconfiguration",
                "A06_2021_Vulnerable_Components",
                "A07_2021_Identification_Authentication_Failures",
                "A08_2021_Software_Integrity_Failures",
                "A09_2021_Security_Logging_Monitoring_Failures",
                "A10_2021_Server_Side_Request_ForT_Forgery"
            ],
            
            QuestionCategory.STATISTICS: [
                "A01_2021_Broken_Access_Control",
                "A02_2021_Cryptographic_Failures",
                "A03_2021_Injection",
                "A04_2021_Insecure_Design",
                "A05_2021_Security_Misconfiguration",
                "A06_2021_Vulnerable_Components",
                "A07_2021_Identification_Authentication_Failures",
                "A08_2021_Software_Integrity_Failures",
                "A09_2021_Security_Logging_Monitoring_Failures",
                "A10_2021_Server_Side_Request_ForT_Forgery"
            ],
            
            QuestionCategory.PROACTIVE_SUGGESTIONS: [
                "A01_2021_Broken_Access_Control",
                "A02_2021_Cryptographic_Failures",
                "A03_2021_Injection",
                "A04_2021_Insecure_Design",
                "A05_2021_Security_Misconfiguration",
                "A06_2021_Vulnerable_Components",
                "A07_2021_Identification_Authentication_Failures",
                "A08_2021_Software_Integrity_Failures",
                "A09_2021_Security_Logging_Monitoring_Failures",
                "A10_2021_Server_Side_Request_ForT_Forgery"
            ]
        }
        
        all_owasp_namespaces = [topic.value for topic in OWASPTopic]
        
        return category_to_namespaces.get(general_category, all_owasp_namespaces)

    def _format_chat_history_for_llm(self) -> str:
        """Formats the chat history for inclusion in the LLM prompt."""
        if not self.chat_history:
            return ""
        
        formatted_history = []
        for i, (user_q, bot_r) in enumerate(self.chat_history):
            formatted_history.append(f"Previous Turn {i+1}:")
            formatted_history.append(f"User: {user_q}")
            formatted_history.append(f"Assistant: {bot_r}")
        return "\n".join(formatted_history) + "\n\n"

    async def generate_llm_response(self, prompt: str, **generation_params) -> str:
        """Generate response using the local LLM with error handling and retries."""
        if not self.components['llm']:
            return "LLM is not available."

        params = {
            "max_tokens": 500,
            "temperature": CONFIG["TEMPERATURE"],
            "top_p": CONFIG["TOP_P"],
            "top_k": CONFIG["TOP_K"],
            "stop": ["</s>", "[/INST]", "\nUser:"],
            "echo": False
        }
        params.update(generation_params)

        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                output = self.components['llm'](prompt, **params)
                return output["choices"][0]["text"].strip()
            except Exception as e:
                if attempt == CONFIG["MAX_RETRIES"] - 1:
                    print(f"Failed to generate response after {CONFIG['MAX_RETRIES']} attempts: {e}")
                    return "I'm having trouble generating a response. Please try again later."
                await asyncio.sleep(1)

        return "Failed to generate response."

    async def process_question(self, question: str) -> Dict[str, Any]:
        """Process a question and generate a response with metadata."""
        if not question.strip():
            return {"error": "Empty question"}

        print("\n🔍 Analyzing your question...")
        
        general_category = await self.classify_question_category(question)
        print(f"General Category: {general_category.value}")
        
        context = await self.retrieve_relevant_context(question, general_category)
        
        history_for_llm = self._format_chat_history_for_llm()

        prompt_template = TOPIC_PROMPTS.get(general_category.name.lower(), TOPIC_PROMPTS["basic_understanding"])
        
        formatted_prompt = (
            f"<s>[INST] {prompt_template['prompt']}\n\n"
            f"{history_for_llm}"
            f"Context:\n{context}\n\n"
            f"Question: {question} [/INST]"
        )
        
        print("Generating response...")
        response = await self.generate_llm_response(
            formatted_prompt,
            temperature=0.7 if general_category in [QuestionCategory.EXAMPLE_SCENARIOS, QuestionCategory.PROACTIVE_SUGGESTIONS] else 0.5
        )
        
        formatted_response = self.format_response(response, general_category)

        self.chat_history.append((question, formatted_response))
        if len(self.chat_history) > CONFIG["MAX_CHAT_HISTORY_TURNS"]:
            self.chat_history = self.chat_history[-CONFIG["MAX_CHAT_HISTORY_TURNS"]:]
        
        return {
            "question": question,
            "category": general_category.value,
            "response": formatted_response,
            "context_used": bool(context)
        }

    def format_response(self, response: str, category: QuestionCategory) -> str:
        """Format the response based on the question category."""
        response = re.sub(r'###\s*', '', response)
        response = re.sub(r'####\s*', '', response)
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        if category == QuestionCategory.PREVENTION_METHODS:
            response = re.sub(r'^\s*(\d+\.|-|\*)\s*', r'• ', response, flags=re.MULTILINE)
        elif category == QuestionCategory.TECHNICAL_EXPLANATION:
            response = re.sub(r'```(\w*)\n(.*?)```', r'```\1\n\2\n```', response, flags=re.DOTALL)
        
        return response.strip()

async def run_interactive_chatbot():
    """
    Runs an interactive terminal chat session with the OWASP Chatbot.
    """
    print("Initializing OWASP Chatbot. This may take a few moments...")
    # Create instance of chatbot
    chatbot = OWASPChatbot() 
    # Await async initialization of components
    await chatbot._async_init_components() 
    
    print("Chatbot ready! Type 'exit' or 'quit' to end the session.")

    while True:
        user_question = input("\n\u001b[34mYou:\u001b[0m ").strip() # Blue for user input
        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_question:
            print("\u001b[33mPlease enter a question.\u001b[0m") # Yellow for warnings
            continue

        start_time = time.time()
        try:
            response_data = await chatbot.process_question(user_question)
            end_time = time.time()
            total_time = end_time - start_time

            if "error" in response_data:
                print(f"\n\u001b[31m❌ Error:\u001b[0m {response_data['error']}") # Red for errors
            else:
                print("\n" + "\u001b[36m=" * 60 + "\u001b[0m") # Cyan for separators
                print(f"\u001b[36m📋 Category:\u001b[0m {response_data['category']}")
                print("\u001b[36m=" * 60 + "\u001b[0m")
                print(f"\u001b[32mAssistant:\u001b[0m {response_data['response']}") # Green for bot response
                print("\u001b[36m=" * 60 + "\u001b[0m")
                
                if response_data.get('context_used'):
                    print("\u001b[90mℹ️  Response generated using available security knowledge base.\u001b[0m") # Grey for info
                
                print(f"\n\u001b[90mTotal processing time: {total_time:.2f}s\u001b[0m") # Grey for performance

        except Exception as e:
            print(f"\n\u001b[31mAn unexpected error occurred:\u001b[0m {e}")
            print("\u001b[31mPlease try again or check the logs.\u001b[0m")

# Main execution block
if __name__ == "__main__":
    asyncio.run(run_interactive_chatbot())