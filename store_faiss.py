import faiss
import json
import numpy as np
import os
import pickle

INPUT_FILE = "data/embedded_chunks.jsonl"
INDEX_DIR = "faiss_index"
DIM = 1536  # embedding dimension for OpenAI "text-embedding-3-small"

def store_faiss():
    os.makedirs(INDEX_DIR, exist_ok=True)
    index = faiss.IndexFlatIP(DIM)
    metadata = []

    with open(INPUT_FILE) as infile:
        for line in infile:
            data = json.loads(line)
            vector = np.array(data["embedding"], dtype="float32")

            # Normalize the vector for cosine similarity
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue  # skip empty vectors
            vector /= norm

            index.add(np.array([vector]))  # now using cosine similarity
            metadata.append({
                "file": data["file"],
                "chunk_id": data["chunk_id"],
                "text": data["text"]
            })

    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    store_faiss()
