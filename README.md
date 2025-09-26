# RAG-QA-BOT

A Retrieval-Augmented Generation (RAG) question-answering bot built in Python.

---

## 📖 Overview

This project uses a RAG architecture to answer user queries by:
1. Retrieving relevant context from a knowledge base or document store.
2. Generating responses using the **Gemini API**.

---

## ⚙️ Prerequisites

- Python 3.12
- A **Gemini API key** (stored in a `.env` file)

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/codeREXus/RAG-QA-BOT.git
cd RAG-QA-BOT
```

### 2. Create a `.env` File
In the root directory, create a `.env` file and add your Gemini API key:
```text
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Install Dependencies
```bash
pip install -r req.txt
```

### 4. Run the Bot
```bash
python bot.py
```



