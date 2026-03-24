import streamlit as st
import requests
import os
from rag_engine import ConstructionRAG

# Mac Fix for FAISS library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

st.set_page_config(page_title="Construction AI Assistant", layout="wide")
st.title("🏗️ Construction AI Assistant")
st.caption("Running locally on device via Ollama (Llama 3.2 3B)")

# 1. Initialize RAG Engine
if 'rag' not in st.session_state:
    with st.spinner("Indexing doc1, doc2, and doc3..."):
        st.session_state.rag = ConstructionRAG()
        st.session_state.rag.ingest_readme_files("./data")
    st.success("Documents Ready!")

# 2. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display Chat History (One after another)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "retrieved_chunks" in message:
            with st.expander("📄 View Retrieved Context (Transparency)"):
                for idx, chunk in enumerate(message["retrieved_chunks"]):
                    st.info(f"**Chunk {idx+1} (Source: {chunk['source']})**\n\n{chunk['text']}")

# 4. Chat Input
if prompt := st.chat_input("Ask a question about construction policies..."):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- RETRIEVAL ---
    results = st.session_state.rag.retrieve(prompt)
    
    if not results:
        response_text = "Information not found in internal documents."
        retrieved_data = []
    else:
        context_text = "\n\n".join([f"Source {r['source']}: {r['text']}" for r in results])
        retrieved_data = results 

        # --- GENERATION (Optimized for 3B/2B) ---
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2", # Shifted to 3B (standard llama3.2 tag)
                        "prompt": f"System: Answer strictly using this context:\n{context_text}\n\nUser: {prompt}",
                        "stream": False,
                        "options": {
                            "temperature": 0 # Forces factual accuracy for construction specs
                        }
                    }
                )
                response_text = response.json().get('response', 'Error: No response.')
        except Exception:
            response_text = "Error: Local LLM is not responding. Is Ollama running?"

    with st.chat_message("assistant"):
        st.markdown(response_text)
        with st.expander("📄 View Retrieved Context (Transparency)"):
            for idx, r in enumerate(retrieved_data):
                st.info(f"**Chunk {idx+1} (Source: {r['source']})**\n\n{r['text']}")
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text, 
        "retrieved_chunks": retrieved_data 
    })