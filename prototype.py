import streamlit as st
import tempfile
import os
import pickle
from PyPDF2 import PdfReader
from PIL import Image
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama
import time
import fitz

# Streamlit page configuration
st.set_page_config(page_title="PDF & Image Chatbot", layout="wide")
st.title("PDF & Image Chatbot")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'images' not in st.session_state:
    st.session_state.images = []


# Function to process PDF
def process_pdf(file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore


def extract_images_from_pdf(pdf_path):
    images = []
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        page = doc[page_number]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append((page_number + 1, img_index + 1, image))
    return images


def query_ollama_with_image(image, query):
    try:
        # Save the image to a temporary file
        temp_image_path = "/tmp/temp_image.png"
        image.save(temp_image_path)

        # Pass the image path and query to the Ollama chat API
        response = ollama.chat(
            model='llava',  # Always using llava model
            messages=[
                {
                    'role': 'user',
                    'content': query,
                    'images': [temp_image_path]
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"An error occurred while querying LLaVA: {e}")
        return "Error occurred."


# Handle PDF upload or selection
st.sidebar.subheader("PDF Options")
option = st.sidebar.radio(
    "Would you like to upload your own PDF or use a default document?",
    ('Upload my own PDF', 'Use a default document')
)

default_pdfs = {
    'Data Science': '/path/to/data_science.pdf',
    'Tech Research': '/path/to/llms.pdf',
    'Humanities': '/path/to/humanities.pdf',
    'Pets': '/path/to/pets.pdf',
    'Business': '/path/to/business.pdf',
    'Short Story': '/path/to/short_story.pdf'
}

if option == 'Use a default document':
    domain = st.sidebar.selectbox("Choose a domain:", list(default_pdfs.keys()))
    selected_pdf_path = default_pdfs[domain]
    st.session_state.pdf_path = selected_pdf_path
    st.write(f"Using default document from the {domain} domain")

elif option == 'Upload my own PDF':
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.pdf_path = tmp_file.name

# User input for chunk size and overlap
st.sidebar.subheader("Text Chunking Settings")
chunk_size = st.sidebar.number_input("Enter chunk size (characters per chunk):", min_value=500, max_value=2000,
                                     value=1000)
chunk_overlap = st.sidebar.number_input("Enter chunk overlap (characters overlap between chunks):", min_value=0,
                                        max_value=500, value=200)

# Process PDF Button
if st.sidebar.button('Process PDF'):
    if not st.session_state.is_processing and st.session_state.pdf_path:
        st.session_state.is_processing = True
        try:
            st.session_state.vectorstore = process_pdf(st.session_state.pdf_path, chunk_size, chunk_overlap)
            st.write("PDF processing complete.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        st.session_state.is_processing = False

# Extract images from PDF
if st.sidebar.button("Extract Images"):
    if st.session_state.pdf_path:
        with st.spinner("Extracting images from PDF..."):
            st.session_state.images = extract_images_from_pdf(st.session_state.pdf_path)
            if st.session_state.images:
                st.success(f"Extracted {len(st.session_state.images)} images from the PDF.")
            else:
                st.warning("No images found in the PDF.")

# Chat interface for text-based interaction
if st.session_state.vectorstore:
    st.subheader("Chat with your PDF")

    # Display images and custom query
    if st.session_state.images:
        image_choices = [f"Image {img_index} from Page {page_number}" for page_number, img_index, _ in st.session_state.images]
        selected_image_choice = st.selectbox("Select an image to query", image_choices)

        # Find the selected image
        selected_image = None
        for page_number, img_index, image in st.session_state.images:
            if f"Image {img_index} from Page {page_number}" == selected_image_choice:
                selected_image = image
                break

        custom_query = st.text_area("Enter your custom query for the selected image:")

        if custom_query:
            with st.spinner("Querying LLaVA..."):
                response = query_ollama_with_image(selected_image, custom_query)
                st.write("LLaVA's response:", response)

    else:
        st.warning("No images extracted from the PDF yet.")

else:
    st.info("Please upload or select a PDF file to start chatting.")
