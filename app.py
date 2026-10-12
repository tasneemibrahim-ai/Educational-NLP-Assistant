import streamlit as st
from transformers import pipeline
import time
import fitz  # PyMuPDF

# Set Streamlit page config
st.set_page_config(page_title="Educational NLP Assistant", layout="centered")

# Load QA and summarization pipelines
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

qa_pipeline = load_qa_pipeline()
summarizer = load_summarizer()

# Title
st.title("Educational NLP Assistant using Small Language Models")

# Task selection
mode = st.radio("Choose task:", ["Question Answering", "Summarization"])

# File upload
uploaded_file = st.file_uploader("Upload a .txt or .pdf file as context:", type=["txt", "pdf"])
uploaded_context = ""

if uploaded_file is not None:
    file_type = uploaded_file.type
    try:
        if file_type == "text/plain":
            uploaded_context = uploaded_file.read().decode("utf-8")
        elif file_type == "application/pdf":
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                uploaded_context = "\n".join(page.get_text() for page in doc)
        st.success("File uploaded and processed.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Use uploaded content or manual entry
if uploaded_context:
    context = uploaded_context
else:
    context = st.text_area("Enter the context manually:", height=200)

# === QUESTION ANSWERING ===
if mode == "Question Answering":
    st.subheader("Question Answering")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer") and context and question:
        with st.spinner("Processing..."):
            start_time = time.time()
            result = qa_pipeline(question=question, context=context)
            elapsed = time.time() - start_time
            st.success(f"Answer: {result['answer']}")
            st.caption(f"Confidence: {result['score']:.2f} | Latency: {elapsed:.2f} seconds")

# === SUMMARIZATION ===
elif mode == "Summarization":
    def chunk_text(text, max_chunk_length=1000):
        sentences = text.split('. ')
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_length:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    st.subheader("Summarization")
    if st.button("Summarize Context") and context:
        with st.spinner("Summarizing..."):
            try:
                if len(context.strip()) < 50:
                    st.warning("Context is too short to summarize. Please provide more content.")
                else:
                    chunks = chunk_text(context)
                    summaries = []
                    for i, chunk in enumerate(chunks):
                        result = summarizer(chunk, max_length=60, min_length=20, do_sample=False)
                        if result and 'summary_text' in result[0]:
                            summaries.append(result[0]['summary_text'])
                    full_summary = " ".join(summaries)
                    st.info(f"Summary:\n\n{full_summary}")
            except Exception as e:
                st.error(f"Summarization error: {e}")
