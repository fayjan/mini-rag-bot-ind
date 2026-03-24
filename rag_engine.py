import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MiniRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def process_documents(self, text):
        # Simple chunking by sentences or fixed length
        self.chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        embeddings = self.encoder.encode(self.chunks)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query, k=3):
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        return [self.chunks[i] for i in indices[0]]