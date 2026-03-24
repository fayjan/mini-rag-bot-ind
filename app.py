import streamlit as st
import requests
from rag_engine import MiniRAG

st.set_page_config(page_title="Construction Marketplace Assistant")
st.title("🏗️ Construction AI Assistant")

# Initialize RAG
if 'rag' not in st.session_state:
    st.session_state.rag = MiniRAG()
    # Load your documents
    with open("data/documents.txt", "r") as f:
        content = f.read()
    st.session_state.rag.process_documents(content)

query = st.text_input("Ask about construction policies or specs:")

if query:
    # 1. Retrieve
    context_chunks = st.session_state.rag.retrieve(query)
    context_text = "\n".join(context_chunks)
    
    # 2. Display Context (Transparency Requirement)
    with st.expander("See Retrieved Context"):
        st.write(context_chunks)

    # 3. Generate (Using OpenRouter)
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}"},
        json={
            "model": "google/gemini-2.0-flash-lite-preview-02-05:free", 
            "messages": [
                {"role": "system", "content": f"Answer ONLY using this context: {context_text}. If not in context, say 'I don't know'."},
                {"role": "user", "content": query}
            ]
        }
    )
    
    answer = response.json()['choices'][0]['message']['content']
    st.subheader("Answer:")
    st.write(answer)