# 🛡️ OWASP Top 10 Security Chatbot with BERT

A context-aware chatbot that provides in-depth knowledge about OWASP Top 10 security vulnerabilities, powered by a fine-tuned BERT model.

---

## 🚀 Features

* ✅ Fine-tuned BERT model specialized for OWASP security topics
* 🧠 Context-aware responses with real-time interaction
* 🔄 Related topic suggestions for deeper security understanding
* 📒 Modular and easy-to-follow Jupyter Notebooks for training and inference

---

## 📁 Project Structure

```
OWASP_BERT/
├── BERT_Finetuning.ipynb        # Notebook for fine-tuning the BERT model
├── Context_Chatbot.ipynb        # Context-aware chatbot implementation
├── QA_Pairs/
│   └── Owasp_Top10/
│       ├── A01_2021.json        # Broken Access Control
│       ├── A02_2021.json        # Cryptographic Failures
│       └── ...                  # Other OWASP Top 10 JSONs
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git ignore file
```

---

## 🧰 Getting Started

### ✅ Prerequisites

* Python 3.8+
* `pip` (Python package manager)
* Jupyter Notebook
* GPU with NVIDIA Cuda support for Fine Tuning the BERT

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

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 💡 Usage

### 1. Fine-tuning the Model

* Open `BERT_Finetuning.ipynb` in Jupyter Notebook
* Follow the steps to fine-tune the BERT model on the OWASP QA dataset
* The fine-tuned model will be saved locally (not included in the repo due to size)

### 2. Using the Chatbot

* Open either `Context_Chatbot.ipynb`
* Run the notebook cells
* Start chatting about OWASP Top 10 vulnerabilities

---

## 📂 Dataset

The QA pairs are structured by OWASP Top 10 categories under:

```
QA_Pairs/Owasp_Top10/
```

Each JSON file contains structured Q\&A pairs, e.g.,:

* `A01_2021.json` → Broken Access Control
* `A02_2021.json` → Cryptographic Failures
* … and more.

---

## ⚠️ Notes

* The fine-tuned model (\~4GB) is **not included** in the repository.
* You can enhance accuracy and response quality by adding more QA pairs to the dataset.
* The chatbot suggests **related security topics** based on the current conversation.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to your fork
5. Submit a Pull Request

---
## 🙏 Acknowledgments

* [OWASP Foundation](https://owasp.org) for the Top 10 security risks documentation
* [Hugging Face](https://huggingface.co/transformers/) for the Transformers library
---

