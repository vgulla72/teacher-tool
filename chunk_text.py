from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

INPUT_FILE = "data/raw_text.jsonl"
OUTPUT_FILE = "data/extracted_chunks.jsonl"

# Customize these
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# Initialize the LangChain splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

def process_chunks():
    with open(INPUT_FILE) as infile, open(OUTPUT_FILE, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            chunks = splitter.split_text(data["text"])
            for i, chunk in enumerate(chunks):
                json.dump({
                    "file": data["file"],
                    "chunk_id": f"{data['file']}_chunk_{i}",
                    "text": chunk
                }, outfile)
                outfile.write("\n")

if __name__ == "__main__":
    process_chunks()
