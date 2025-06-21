# 📄 SmartDoc QA

An AI-powered document question-answering system that uses **Retrieval-Augmented Generation (RAG)**, up to **2 LLMs**, **tool calling**, and supports **monitoring**, **evaluation**, and **deployment**.

---

## 🚀 Features

- ✅ Upload documents (PDF, Word)  
- 🔍 Retrieve relevant context using vector search  
- 🧠 Generate accurate answers with LLMs  
- 🛠️ Tool calling (e.g., calculator, summarizer, web search)  
- 🧪 Offline & online evaluation support  
- 📊 Real-time monitoring of performance  
- 🌐 Deployable via Streamlit, Docker, or HuggingFace Spaces  

---

## 🧠 Architecture Overview

1. **Document Ingestion**  
   - Convert uploaded files to text  
   - Chunk and embed using transformer models  

2. **Retrieval**  
   - FAISS or Chroma vector DB for similarity search  

3. **Answer Generation**  
   - Primary LLM + optional secondary LLM  
   - Injects retrieved chunks into prompt context  

4. **Tool Calling**  
   - Invokes custom tools based on query need  

5. **Monitoring**  
   - Logs latency, token usage, and tool calls  

6. **Evaluation**  
   - Test sets, user feedback, and automatic metrics  

---

## 📂 Project Structure

```
smartdoc-qa/
├── backend/
│   ├── main.py              # Entry API or orchestration script
│   ├── rag_pipeline.py      # RAG flow logic
│   ├── tool_router.py       # Tool invocation handler
│   ├── embeddings.py        # Text splitting & embedding
│   ├── llm_wrapper.py       # LLM abstraction layer
│   ├── evaluation.py        # Offline evaluation tools
│   └── monitor.py           # Logging, tracing
│
├── docs/                    # Sample documents
├── logs/                    # Monitoring logs
├── tests/                   # Evaluation test queries
├── ui/
│   └── streamlit_app.py     # Web UI
├── .env                     # API keys and secrets
├── Dockerfile               # For containerized deployment
└── README.md
```

---

## 🧰 Tech Stack

| Category         | Tool/Tech                         |
|------------------|------------------------------------|
| LLM              | OpenAI GPT-4 / Claude / Mistral    |
| Embedding Model  | OpenAI / Sentence Transformers     |
| Vector DB        | FAISS or Chroma                    |
| Framework        | LangChain or LlamaIndex            |
| Frontend         | Streamlit                          |
| Backend          | FastAPI (optional)                 |
| Monitoring       | Python logging / Prometheus (opt)  |
| Evaluation       | Trulens, LangChain Eval, Custom    |
| Deployment       | Docker, HuggingFace Spaces         |

---

## ⚙️ Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/yourname/smartdoc-qa.git
cd smartdoc-qa
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_claude_key
```

### 3. Run App (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

---

## 🧪 Evaluation

- Run offline eval:

```bash
python backend/evaluation.py
```

- Online feedback:
  - Users can rate answers in the UI  
  - Stored in `logs/feedback.json`  

---

## 📈 Monitoring

- Logs saved to `logs/monitor.log`  
- Tracks:  
  - Query time  
  - Token usage  
  - Tool invocation  
  - Retrieval quality  

---

## 🐳 Deployment

### Option 1: Docker

```bash
docker build -t smartdoc-qa .
docker run -p 8501:8501 smartdoc-qa
```

### Option 2: HuggingFace Spaces

- Add `README.md`, `app.py`, `requirements.txt`  
- Push to a public repo connected to HF Spaces  

---

## 🧠 Future Enhancements

- Add user auth for persistent histories  
- Plug in Langfuse or Trulens for deeper eval  
- Use OpenLLM or Ollama for local inference  

---

## 📜 License

MIT License.

---

## 👨‍💻 Author

Built by [Your Name] – AI & LLM Applications Engineer.
