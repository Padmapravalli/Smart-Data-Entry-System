# SMART DOCUMENT QA SYSTEM WITH OLLAMA, HIGHLIGHTING, AND CHAT BUBBLES

import streamlit as st
import pandas as pd
import fitz
import docx
import pytesseract
import chardet
import cv2
import numpy as np
from PIL import Image
import socket
import os
import ollama
import re

# Setup
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
DATA_FILE = "smart_doc_entries.csv"

# Session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Internet check
def is_connected():
    try:
        socket.create_connection(("www.google.com", 80), timeout=5)
        return True
    except OSError:
        return False

# Image preprocessing
def preprocess_image(pil_img):
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

# File readers
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        page_text = page.get_text().strip()
        if not page_text:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(preprocess_image(img), lang='eng')
        text += page_text + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt(file):
    raw = file.read()
    encoding = chardet.detect(raw)['encoding']
    return raw.decode(encoding)

def read_excel(file):
    df = pd.read_excel(file)
    return df.to_string(index=False)

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)

def read_image(file):
    img = Image.open(file)
    processed_img = preprocess_image(img)
    return pytesseract.image_to_string(processed_img, lang='eng')

def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == "pdf":
        return read_pdf(file)
    elif ext == "docx":
        return read_docx(file)
    elif ext == "txt":
        return read_txt(file)
    elif ext in ["xls", "xlsx"]:
        return read_excel(file)
    elif ext == "csv":
        return read_csv(file)
    elif ext in ["jpg", "jpeg", "png"]:
        return read_image(file)
    else:
        return "Unsupported file format."

# LLM-based QA using OLLAMA
def ask_llm(question, context):
    try:
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "Answer questions based on the given context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"[LLM Error: {e}]"

# Highlight query matches
def highlight_text(text, query):
    if not query:
        return text
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

# Save conversation
def save_entry(text, chat_history):
    full = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    df = pd.DataFrame([{"Extracted_Content": text, "Chat_History": full}])
    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, index=False)
    else:
        df.to_csv(DATA_FILE, mode='a', header=False, index=False)

# UI
st.set_page_config(page_title="📄 Smart Document Assistant", layout="wide")
st.title("📑 Smart Document Extractor & Chat Assistant")

uploaded_file = st.file_uploader("📤 Upload your document", type=["pdf", "docx", "txt", "xls", "xlsx", "csv", "png", "jpg", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with st.spinner("🔍 Extracting text..."):
        content = extract_text(uploaded_file)

    with col1:
        st.subheader("📜 Extracted Content (Highlight Active)")
        query_highlight = st.text_input("🔎 Enter keyword to highlight in the content")
        highlighted = highlight_text(content, query_highlight)
        st.markdown(highlighted, unsafe_allow_html=True)

    with col2:
        st.subheader("💬 Ask a Question")
        question = st.text_input("Type your question about the document")

        if question:
            response = ask_llm(question, content)
            st.session_state.chat_history.append((question, response))

        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(f"**You:** {q}")
            with st.chat_message("assistant"):
                st.markdown(f"**AI:** {a}")

        if st.button("✅ Save Extracted Data & Conversation"):
            save_entry(content, st.session_state.chat_history)
            st.success("Saved successfully!")

# Show previous entries
if os.path.exists(DATA_FILE):
    st.sidebar.subheader("📊 All Saved Entries")
    df = pd.read_csv(DATA_FILE)
    st.sidebar.dataframe(df, use_container_width=True)
