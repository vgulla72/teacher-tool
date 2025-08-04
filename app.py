import streamlit as st
import faiss
import numpy as np
import pickle
import os
from langchain_community.chat_models import ChatOllama
from langchain.schema.messages import HumanMessage
from openai import OpenAI

st.set_page_config(page_title="Personal Query Tool", layout="wide")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
TOP_K = 10

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def embed_query(query: str):
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    embedding = np.array(response.data[0].embedding, dtype="float32")
    return embedding / np.linalg.norm(embedding)  # normalize for cosine similarity

def search_index(query_embedding, index, metadata, k=TOP_K):
    vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(vector, k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

def generate_answer_with_context(question, retrieved_chunks):
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = f"""
You are a highly knowledgeable and detail-oriented teaching assistant helping a new  professor create accurate, clear, and engaging lesson plans.

Below is a selection of curated teaching materials. Use this content to:
- Answer the professor's question with contextual clarity.
- Understand key concepts from the examples provided. Generate the concepts by understanding the context of the content and not use examples verbatim.
- Suggest a structured lesson plan if relevant.

Always ensure your response:
- Is factually correct.
- Uses clear and concise language.
- Reflects the tone and depth suitable for undergraduate or graduate-level  courses.

------------------------
Teaching Materials:
{context}
------------------------

Question: {question}

Answer:
"""

    ollama_model = "mistral"  # or "llama3", etc.
    llm = ChatOllama(
        model=ollama_model,
        base_url="http://localhost:11434",
        temperature=0.4,
        request_timeout=120
    )

    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()



def main():
    st.title("📘 Search Teaching Material")

    user_query = st.text_input("Enter a topic or question", "")

    if st.button("Search") and user_query.strip():
        with st.spinner("Searching..."):
            try:
                index, metadata = load_faiss_index()
                query_embedding = embed_query(user_query)
                results = search_index(query_embedding, index, metadata)

                st.subheader("📚 Retrieved Chunks")
                for i, result in enumerate(results):
                    st.markdown(f"**{i+1}. File:** `{os.path.basename(result['file'])}`")
                    st.code(result["text"].strip(), language="markdown")
                    st.markdown("---")

                st.subheader("🧠 Suggested Lesson Plan / Answer")
                answer = generate_answer_with_context(user_query, results)
                st.markdown(answer)

            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
