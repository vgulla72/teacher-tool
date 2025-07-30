import os
from pathlib import Path
import docx
import pptx
import fitz  # PyMuPDF
import json

SOURCE_DIR = "/Users/vasanthagullapalli/Documents/Newsletter"
OUTPUT_FILE = "data/raw_text.jsonl"

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_pptx(path):
    prs = pptx.Presentation(path)
    text_runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_runs.append(shape.text)
    return "\n".join(text_runs)

def extract_all_text():
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as outfile:
        for root, _, files in os.walk(SOURCE_DIR):
            for file in files:
                ext = file.lower().split('.')[-1]
                path = os.path.join(root, file)
                try:
                    if ext == "pdf":
                        content = extract_text_from_pdf(path)
                    elif ext == "docx":
                        content = extract_text_from_docx(path)
                    elif ext == "pptx":
                        content = extract_text_from_pptx(path)
                    elif ext == "txt":
                        content = Path(path).read_text()
                    else:
                        continue  # skip unsupported formats
                    json.dump({"file": path, "text": content}, outfile)
                    outfile.write("\n")
                except Exception as e:
                    print(f"Failed to extract {path}: {e}")

if __name__ == "__main__":
    extract_all_text()
