import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class ConstructionRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = [] 

    def ingest_readme_files(self, directory_path="./data"):
        all_text_chunks = []
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        for filename in os.listdir(directory_path):
            if filename.endswith(".md") or filename.endswith(".txt"):
                with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                    # Chunking by double newlines to keep paragraphs intact
                    chunks = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 10]
                    for chunk in chunks:
                        all_text_chunks.append(chunk)
                        self.metadata.append({"text": chunk, "source": filename})

        if all_text_chunks:
            embeddings = self.encoder.encode(all_text_chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query, k=3):
        if self.index is None: return []
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        return [self.metadata[i] for i in indices[0] if i != -1]