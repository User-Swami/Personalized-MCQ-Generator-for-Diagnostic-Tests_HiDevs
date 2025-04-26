__import__('pysqlite3')
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
st.set_page_config(page_title="🎯 Personalized MCQ Generator", layout="centered")

import chromadb
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import random

# Custom CSS
st.markdown("""
    <style>
        body {
            background-image: url('https://www.transparenttextures.com/patterns/graphy.png');
            background-color: #fdf6e3;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-container {
            background-color: #ffffffcc;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            border: 5px solid transparent;
            border-image: linear-gradient(to right, #f06, #4a90e2) 1;
        }
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #1e3a8a;
            margin-bottom: 1rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #374151;
            text-align: center;
            margin-bottom: 2rem;
        }
        .mcq-box {
            background-color: #e0f2fe;
            padding: 1.5rem;
            border: 2px dashed #60a5fa;
            border-radius: 12px;
        }
        .alphabet-border {
            border: 6px solid #e0e7ff;
            padding: 20px;
            border-radius: 25px;
            background-image: linear-gradient(45deg, #dbeafe 25%, transparent 25%),
                              linear-gradient(-45deg, #dbeafe 25%, transparent 25%),
                              linear-gradient(45deg, transparent 75%, #bfdbfe 75%),
                              linear-gradient(-45deg, transparent 75%, #bfdbfe 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        }
    </style>
""", unsafe_allow_html=True)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Utility Functions
def load_pdf(file):
    reader = PdfReader(file)
    return "".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def initialize_chromadb():
    client = chromadb.PersistentClient(path="./mcq_chroma_db")
    return client.get_or_create_collection(name="mcq_knowledge_base")

def store_embeddings(chunks, collection):
    existing_docs = set(collection.get().get("documents", []))
    new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]
    if new_chunks:
        embeddings = [embedding_model.embed_query(chunk) for chunk in new_chunks]
        collection.add(
            ids=[str(i) for i in range(len(existing_docs), len(existing_docs) + len(new_chunks))],
            documents=new_chunks,
            embeddings=embeddings
        )
        return len(new_chunks)
    return 0

def generate_mcq(text):
    words = text.split()
    if len(words) < 5:
        return None
    question = f"What is the main idea of: '{' '.join(words[:10])}...'?"
    correct = "Correct answer based on content"
    options = [correct, "Random guess A", "Random guess B", "Random guess C"]
    random.shuffle(options)
    return question, correct, options

# Main UI
with st.container():
    st.markdown("<div class='main-container alphabet-border'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>🎯 Personalized MCQ Generator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload your notes or textbook PDF and test yourself with AI-generated questions!</div>", unsafe_allow_html=True)

    pdf_file = st.file_uploader("📥 Upload Study Material (PDF only):", type=["pdf"])
    if pdf_file:
        with st.spinner("🧠 Reading and understanding your content..."):
            text = load_pdf(pdf_file)
            chunks = chunk_text(text)
            collection = initialize_chromadb()
            added = store_embeddings(chunks, collection)
            st.success(f"✅ Embedded {added} chunks successfully!")

            if st.button("🧪 Generate MCQs"):
                st.markdown("<div class='mcq-box'>", unsafe_allow_html=True)
                for i in range(min(5, len(chunks))):
                    mcq = generate_mcq(chunks[i])
                    if mcq:
                        question, correct, options = mcq
                        st.write(f"**Q{i+1}: {question}**")
                        st.radio("Choose your answer:", options, key=f"q{i}")
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("📘 Please upload a PDF to begin generating MCQs.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Made with 🤖 HuggingFace, LangChain, and Streamlit")
