import json
import os
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "data/extracted_chunks.jsonl"
OUTPUT_FILE = "data/embedded_chunks.jsonl"
CHECKPOINT_FILE = "data/embedded_chunk_ids.txt"

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

def load_already_processed_ids():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return set(line.strip() for line in f)

def save_processed_ids(chunk_ids):
    with open(CHECKPOINT_FILE, "a") as f:
        for cid in chunk_ids:
            f.write(cid + "\n")

def get_embeddings_batch(texts):
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [e.embedding for e in response.data]

def embed_chunks():
    already_done = load_already_processed_ids()
    buffer = []

    with open(INPUT_FILE) as infile, open(OUTPUT_FILE, "a") as outfile:
        batch = []
        batch_meta = []

        for line in tqdm(infile, desc="Processing chunks"):
            data = json.loads(line)
            chunk_id = data["chunk_id"]

            if chunk_id in already_done:
                continue

            batch.append(data["text"])
            batch_meta.append(data)

            if len(batch) == BATCH_SIZE:
                try:
                    embeddings = get_embeddings_batch(batch)
                    for meta, emb in zip(batch_meta, embeddings):
                        meta["embedding"] = emb
                        json.dump(meta, outfile)
                        outfile.write("\n")
                    save_processed_ids([m["chunk_id"] for m in batch_meta])
                except Exception as e:
                    print(f"Failed batch: {e}")
                batch, batch_meta = [], []

        # process remaining
        if batch:
            try:
                embeddings = get_embeddings_batch(batch)
                for meta, emb in zip(batch_meta, embeddings):
                    meta["embedding"] = emb
                    json.dump(meta, outfile)
                    outfile.write("\n")
                save_processed_ids([m["chunk_id"] for m in batch_meta])
            except Exception as e:
                print(f"Failed final batch: {e}")

if __name__ == "__main__":
    embed_chunks()
