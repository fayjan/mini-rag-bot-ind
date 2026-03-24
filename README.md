# 🏗️ Construction Marketplace AI Assistant (Mini RAG)

A lightweight Retrieval-Augmented Generation (RAG) pipeline designed to answer construction-related queries using internal policy and specification documents.

## 🚀 Features
- **Semantic Search:** Uses `sentence-transformers` for high-dimensional text embeddings.
- **Vector Storage:** Local `FAISS` (Facebook AI Similarity Search) index for lightning-fast retrieval.
- **Grounded Generation:** LLM is strictly constrained via system prompting to prevent hallucinations.
- **Transparent UI:** Built with Streamlit to display retrieved chunks alongside the final answer.

## 🛠️ Tech Stack
| Component | Choice | Reason |
| :--- | :--- | :--- |
| **Embedding Model** | `all-MiniLM-L6-v2` | Excellent balance between latency and accuracy for short document chunks. |
| **Vector Store** | `FAISS` | Standard for local semantic search; handles L2 distance calculations efficiently. |
| **LLM** |Llama 3.2 3B (Local via Ollama)|For privacy and zero latency.
| **Frontend** | `Streamlit` | Rapid deployment of a clean, functional chatbot interface. |

## 🧩 Implementation Details

### 1. Document Processing & Chunking
Documents are loaded from `data/docx.md`. To maintain context while keeping embeddings precise, I implemented **Recursive Character Chunking**. Each chunk is approximately 500 characters with a small overlap to ensure no technical specification is "cut in half" at the boundary.

### 2. Retrieval Logic
The user query is embedded using the same `SentenceTransformer`. We then perform a **k-nearest neighbor (k-NN) search** in the FAISS index to find the top 3 most relevant segments.

### 3. Grounding & Anti-Hallucination
The LLM is provided with a strict System Prompt:
> "You are a construction assistant. Use ONLY the provided context to answer. If the answer is not in the context, state that you do not have that information. Do not use outside 
knowledge."


## 📺 Project Demonstration

<p align="center">
  <video src="https://github.com/fayjan/mini-rag-bot-ind/releases/download/v1.0-demo/ai-chat-bot.mp4" width="100%" controls>
    Your browser does not support the video tag.
  </video>
</p>

[🔗 Click here to watch the video if it doesn't load](https://github.com/fayjan/mini-rag-bot-ind/releases/download/v1.0-demo/ai-chat-bot.mp4)

---


## ⚙️ How to Run

### Prerequisites
1. **Install Ollama:** Download from [ollama.com](https://ollama.com).
2. **Download Model:** Open terminal/command prompt and run:
   ```bash
   ollama run llama3.2


### Setup & Installation

1. **Clone the repo:**
   ```bash
    git clone https://github.com/MD-Fayjan/mini-rag-bot-ind.git
    cd mini-rag-bot-ind
   ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the App:**
    ```bash
    streamlit run app.py
    ```