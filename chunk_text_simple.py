import json
from pathlib import Path

INPUT_FILE = "data/raw_text.jsonl"
OUTPUT_FILE = "data/extracted_chunks.jsonl"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_chunks():
    with open(INPUT_FILE) as infile, open(OUTPUT_FILE, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            chunks = chunk_text(data["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                json.dump({
                    "file": data["file"],
                    "chunk_id": f"{data['file']}_chunk_{i}",
                    "text": chunk
                }, outfile)
                outfile.write("\n")

if __name__ == "__main__":
    process_chunks()
