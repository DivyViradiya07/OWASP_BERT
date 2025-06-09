<div align="center">
  <h1 align="center">🛡️ OWASP Top 10 Security Chatbot</h1>
  <h3 align="center">BERT-Powered Security Assistant for OWASP Top 10 Vulnerabilities</h3>
  
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#notebooks">Notebooks</a> •
    <a href="#dataset">Dataset</a>
  </p>
  
  <p align="center">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.9+-blue.svg">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
    <img alt="Open Source" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
  </p>
</div>

A sophisticated, context-aware chatbot that provides comprehensive knowledge about OWASP Top 10 security vulnerabilities. Built with a fine-tuned BERT model and modern NLP techniques, this tool helps developers and security professionals understand and mitigate common web application security risks.

<div align="center">
  <img src="https://img.shields.io/badge/BERT-FF6F00?style=flat&logo=huggingface&logoColor=white" alt="BERT">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter">
</div>

---

<a id="features"></a>
## 🚀 Key Features

<table>
  <tr>
    <td width="50%">
      <h3>🎯 Specialized Knowledge</h3>
      <ul>
        <li>Fine-tuned BERT model for OWASP security topics</li>
        <li>Comprehensive coverage of OWASP Top 10 2021</li>
        <li>Regularly updated vulnerability database</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🧠 Smart Interaction</h3>
      <ul>
        <li>Context-aware conversations</li>
        <li>Advanced question classification</li>
        <li>Related topic suggestions</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <h3>⚙️ Technical Excellence</h3>
      <ul>
        <li>Modular, maintainable architecture</li>
        <li>Asynchronous processing</li>
        <li>Input sanitization & security</li>
      </ul>
    </td>
    <td>
      <h3>📊 Data & Analysis</h3>
      <ul>
        <li>Structured QA dataset</li>
        <li>Jupyter notebooks for analysis</li>
        <li>Knowledge graph integration</li>
      </ul>
    </td>
  </tr>
</table>

---

## 📁 Project Structure

```
OWASP_BERT/
├── chatbot_modules/            # Core chatbot implementation
│   ├── __init__.py
│   ├── chatbot.py              # Main chatbot class
│   ├── classification.py       # Question classification logic
│   ├── components.py           # Component management
│   ├── config.py               # Configuration settings
│   ├── constants.py            # Constants and enums
│   ├── llm_service.py          # LLM integration
│   ├── prompts.py              # System and user prompts
│   ├── retrieval.py            # Context retrieval logic
│   └── utils.py                # Utility functions
├── QA_Pairs/                   # Training data
│   └── Owasp_Top10/
│       ├── A01_2021.json        # Broken Access Control
│       ├── A02_2021.json        # Cryptographic Failures
│       └── ...                  # Other OWASP Top 10 JSONs
├── requirements.txt            # Python dependencies
└── .gitignore                 # Git ignore file
```

---

<a id="installation"></a>
## 📁 Project Structure

```
OWASP_BERT/
├── chatbot_modules/    # Core chatbot implementation
│   ├── __init__.py     # Package initialization
│   ├── chatbot.py      # Main chatbot class and conversation handling
│   ├── classification.py  # Question classification  
│   ├── components.py  # Component management and initialization
│   ├── config.py      # Configuration settings and constants
│   ├── constants.py   # Enums and fixed values
│   ├── llm_service.py # Integration with language models
│   ├── prompts.py     # System and user prompt templates
│   ├── retrieval.py   # Context retrieval and semantic search
│   └── utils.py       # Helper functions and utilities
├── QA_Pairs/          # Training and knowledge data
│   ├── Enhanced_QA/   # Enhanced question-answer pairs
│   ├── Owasp_Top10/    # OWASP Top 10 categories and questions
│   ├── Populated/      # Processed and populated QA data
│   └── Processed_CVEs/ # Processed CVE information
├── Prompt_Templates/   # Templates for different types of prompts
├── main Executions     # Main execution files
│   ├── CVEs_Data_Processing.ipynb
│   ├── Data_Preprocessing.ipynb
│   ├── S1_Semantic_Search.ipynb
│   ├── S2_Knowledge_Graph.ipynb
│   ├── S3_Embedding_Generation.ipynb
│   ├── S4_Model_Download.ipynb
│   └── S5_Chatbot.py
├── README.md                 # Project documentation
├── requirements.txt           # Python dependencies
└── .env.example              # Example environment variables
```

---

## 🧰 Getting Started

### ✅ Prerequisites

- 🐍 **Python 3.9+**
- 📦 **Package Manager**: `pip` or `poetry`
- 🖥️ **Hardware**: GPU with CUDA support (recommended)
- 🔑 **API Keys**: OpenAI API key for LLM integration

> 💡 **Tip**: For optimal performance, we recommend using a machine with at least 16GB RAM and an NVIDIA GPU with CUDA support.

---

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/DivyViradiya07/OWASP_BERT.git
   cd OWASP_BERT
   ```

2. **Create and activate a virtual environment (recommended):**

   **Linux/Mac:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   **Windows:**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies using Poetry:**

   ```bash
   # Install Poetry if you don't have it
   pip install poetry
   
   # Install project dependencies
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

<a id="usage"></a>
## 🎯 Using the Chatbot

### Basic Usage

1. **Starting the Chatbot**
   ```bash
   # Activate virtual environment (if using)
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate  # Windows

   # Install dependencies
   pip install -r requirements.txt

   # Start the chatbot
   python -m chatbot_modules.chatbot
   ```

2. **Interacting with the Chatbot**
   - Ask security-related questions in natural language
   - The chatbot will provide detailed, context-aware responses
   - Use commands like 'help' to see available options
   - Type 'exit' or 'quit' to end the session

### Advanced Features

#### Context-Aware Conversations
```
User: What is SQL injection?
Chatbot: [Explains SQL injection]
User: How can I prevent it in Python?
Chatbot: [Provides Python-specific prevention techniques]
```

#### Vulnerability Analysis
```
User: Analyze this SQL query for injection risks: SELECT * FROM users WHERE id = '123'
Chatbot: [Identifies potential risks and suggests parameterized queries]
```

#### Security Recommendations
```
User: How can I secure my API endpoints?
Chatbot: [Lists security best practices for API endpoints]
```

### Environment Variables
Create a `.env` file with the following variables:
```
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
DEBUG=false
LOG_LEVEL=INFO
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.7
```

### Command Line Arguments
```bash
# Run with debug mode
python -m chatbot_modules.chatbot --debug

# Specify a different model
python -m chatbot_modules.chatbot --model gpt-4

# Load custom configuration
python -m chatbot_modules.chatbot --config custom_config.yaml
```

---

## 📚 Understanding the Notebooks

### 1. Data Preprocessing (`Data_Preprocessing.ipynb`)
- **Purpose**: Prepare and clean the raw OWASP dataset
- **Key Operations**:
  - Load and parse JSON data
  - Clean and normalize text
  - Split data into training/validation sets
  - Generate embeddings for text

### 2. CVE Data Processing (`CVEs_Data_Processing.ipynb`)
- **Purpose**: Process and integrate CVE data
- **Key Operations**:
  - Fetch CVE data from NVD database
  - Map CVEs to OWASP categories
  - Enrich vulnerability information
  - Generate structured QA pairs

### 3. Semantic Search (`S1_Semantic_Search.ipynb`)
- **Purpose**: Implement and test semantic search
- **Key Operations**:
  - Generate document embeddings
  - Implement cosine similarity search
  - Test search accuracy
  - Optimize search parameters

### 4. Knowledge Graph (`S2_Knowledge_Graph.ipynb`)
- **Purpose**: Create and visualize security concepts
- **Key Operations**:
  - Extract entities and relationships
  - Build knowledge graph
  - Visualize connections
  - Query the graph

### 5. Embedding Generation (`S3_Embedding_Generation.ipynb`)
- **Purpose**: Generate vector embeddings
- **Key Operations**:
  - Load pre-trained models
  - Generate document embeddings
  - Store embeddings efficiently
  - Test embedding quality

### 6. Model Download (`S4_Model_Download.ipynb`)
- **Purpose**: Manage ML models
- **Key Operations**:
  - Download pre-trained models
  - Verify model integrity
  - Configure model parameters
  - Test model performance

## 🛠️ Development Guide

### Setting Up Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/DivyViradiya07/OWASP_BERT.git
   cd OWASP_BERT
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_retrieval.py

# Run with coverage
pytest --cov=chatbot_modules
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for better code clarity
- Document all public functions with docstrings
- Keep functions small and focused

### Adding New Features
1. Create a new branch
2. Write tests for your feature
3. Implement the feature
4. Update documentation
5. Submit a pull request

## 📈 Performance Tuning

### Improving Response Time
- Enable response caching
- Optimize embedding generation
- Use batch processing for multiple queries
- Implement request rate limiting

### Enhancing Accuracy
- Fine-tune the model on domain-specific data
- Expand the training dataset
- Implement feedback loop for continuous improvement
- Use ensemble methods for critical classifications

## 🌐 API Integration

The chatbot can be integrated into web applications using the provided API:

```python
from chatbot_modules.chatbot import OWASPChatbot

# Initialize the chatbot
chatbot = OWASPChatbot()

# Get response to a query
response = chatbot.get_response("How to prevent XSS attacks?")
print(response['answer'])
```

### API Endpoints (if implemented)
- `POST /api/chat` - Send a message to the chatbot
- `GET /api/topics` - List available security topics
- `GET /api/vulnerabilities` - Search for vulnerabilities

## 🔒 Security Considerations

- Never expose API keys in client-side code
- Implement proper input validation
- Use HTTPS for all API communications
- Regularly update dependencies
- Monitor for unusual activity

## 📊 Monitoring and Logging

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
```

### Monitoring
- Track response times
- Monitor error rates
- Log user interactions (anonymized)
- Set up alerts for critical issues

---

## 💡 Additional Resources

### 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/DivyViradiya07/OWASP_BERT.git
cd OWASP_BERT

# Install dependencies
pip install -r requirements.txt

# Start the chatbot
python -m chatbot_modules.chatbot
```

### 🔍 Example Queries

Try asking the chatbot about:

```
• "Explain SQL injection with examples"
• "How to prevent XSS attacks in React?"
• "What are common misconfigurations in AWS S3?"
• "Show me the OWASP Top 10 for 2021"
• "How does broken access control work?"
```

### 🎯 Advanced Usage

For development and customization:

```bash
# Run with debug mode
export DEBUG=true
python -m chatbot_modules.chatbot

# Process custom dataset
python -m scripts.process_data --input data/custom_qa.json
```

## 🛠️ Development

### Adding New Features

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the existing code style

3. Run tests (if available):
   ```bash
   poetry run pytest
   ```

4. Submit a pull request

---

<a id="notebooks"></a>
## 📚 Jupyter Notebooks

Explore and extend the project's capabilities with our comprehensive notebooks:

### Data Processing
| Notebook | Description | Key Features |
|----------|-------------|--------------|
| `Data_Preprocessing.ipynb` | Processes raw OWASP dataset | Data cleaning, transformation, preparation |
| `CVEs_Data_Processing.ipynb` | Integrates CVE data | CVE parsing, vulnerability mapping |

### Model and Embeddings
| Notebook | Description | Key Features |
|----------|-------------|--------------|
| `S3_Embedding_Generation.ipynb` | Generates BERT embeddings | Vectorization, similarity scoring |
| `S4_Model_Download.ipynb` | Manages BERT model | Model downloading, setup |

### Search and Knowledge Graph
| Notebook | Description | Key Features |
|----------|-------------|--------------|
| `S1_Semantic_Search.ipynb` | Implements search | Contextual search, ranking |
| `S2_Knowledge_Graph.ipynb` | Visualizes OWASP concepts | Graph construction, visualization |

These notebooks provide a step-by-step guide to understanding and working with the data and models used in the chatbot. They're particularly useful for:
- Modifying the data processing pipeline
- Updating the knowledge base
- Retraining or fine-tuning the models
- Understanding the underlying data structures

<a id="dataset"></a>
## 📂 Dataset Structure

Our comprehensive dataset is organized by OWASP Top 10 2021 categories in the `QA_Pairs/Owasp_Top10/` directory. Each category contains meticulously curated Q&A pairs to train and evaluate the chatbot's understanding of web application security.

### Dataset Structure

Each JSON file contains structured Q&A pairs with the following format:

```json
{
  "topic": "A01:2021 - Broken Access Control",
  "description": "...",
  "questions": [
    {
      "question": "What is broken access control?",
      "answer": "...",
      "category": "basic_understanding",
      "owasp_topics": ["broken_access_control"]
    },
    ...
  ]
}
```

### Available Categories

- `A01_2021.json`: Broken Access Control
- `A02_2021.json`: Cryptographic Failures
- `A03_2021.json`: Injection
- `A04_2021.json`: Insecure Design
- `A05_2021.json`: Security Misconfiguration
- `A06_2021.json`: Vulnerable and Outdated Components
- `A07_2021.json`: Identification and Authentication Failures
- `A08_2021.json`: Software and Data Integrity Failures
- `A09_2021.json`: Security Logging and Monitoring Failures
- `A10_2021.json`: Server-Side Request Forgery

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add some amazing feature'`)
4. 🔀 Push to the branch (`git push origin feature/amazing-feature`)
5. 🎉 Open a Pull Request

### 🐛 Reporting Issues
Found a bug or have a feature request? Please [open an issue](https://github.com/DivyViradiya07/OWASP_BERT/issues).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OWASP Foundation](https://owasp.org) for the Top 10 security risks documentation
- [Hugging Face](https://huggingface.co/transformers/) for the Transformers library
- [OpenAI](https://openai.com) for their powerful language models

## 📚 Resources

- [OWASP Top 10 2021](https://owasp.org/Top10/)
- [Hugging Face Documentation](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

<div align="center">
  Made with ❤️ by the OWASP BERT Team
</div>

