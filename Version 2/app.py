# app.py

import streamlit as st
import tempfile
import os

# Import modularized functions
import config
import pdf_processor
import vector_store
import llm_handler

# --- Page and Session State Setup ---
st.set_page_config(page_title="AI PDF & Image Chatbot", layout="wide")
st.title("📄🖼️ AI Chatbot for PDFs & Images")

def initialize_session_state():
    """Initializes session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'images' not in st.session_state:
        st.session_state.images = []

# --- Sidebar for PDF Upload and Processing ---
def setup_sidebar():
    """Configures the sidebar for file upload and processing controls."""
    st.sidebar.header("⚙️ Setup")

    option = st.sidebar.radio(
        "Choose PDF Source:",
        ('Upload your own', 'Use a default document')
    )

    if option == 'Use a default document':
        domain = st.sidebar.selectbox("Choose a domain:", list(config.DEFAULT_PDFS.keys()))
        st.session_state.pdf_path = config.DEFAULT_PDFS.get(domain)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                st.session_state.pdf_path = tmp.name

    st.sidebar.subheader("Chunking Settings")
    chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, config.DEFAULT_CHUNK_SIZE, 100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, config.DEFAULT_CHUNK_OVERLAP, 50)

    if st.sidebar.button("Process PDF", use_container_width=True, type="primary"):
        if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
            with st.spinner("Processing PDF... This may take a moment."):
                text_chunks = pdf_processor.extract_text_and_split(st.session_state.pdf_path, chunk_size, chunk_overlap)
                st.session_state.vectorstore = vector_store.create_vector_store(text_chunks)
                st.session_state.images = pdf_processor.extract_images_from_pdf(st.session_state.pdf_path)
            st.success(f"PDF processed! Found {len(st.session_state.images)} images.")
            st.session_state.chat_history = [] # Reset chat
        else:
            st.warning("Please select a valid PDF file first.")

# --- Main Chat Interface ---
def main_interface():
    """Renders the main chat interface using tabs."""
    if not st.session_state.vectorstore:
        st.info("👋 Welcome! Please upload or select a PDF and click 'Process PDF' to begin.")
        return

    tab1, tab2 = st.tabs(["💬 Chat with Text", "🖼️ Chat with Images"])

    with tab1:
        st.subheader("Query the PDF's Text Content")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about the PDF..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = llm_handler.get_text_chat_response(
                        st.session_state.vectorstore, prompt, st.session_state.chat_history
                    )
                    st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    with tab2:
        st.subheader("Query Images in the PDF")
        if not st.session_state.images:
            st.warning("No images were found in this PDF.")
        else:
            img_choices = {f"Page {p_num}, Image {idx}": img for p_num, idx, img in st.session_state.images}
            selected_key = st.selectbox("Select an image:", img_choices.keys())
            if selected_key:
                selected_img = img_choices[selected_key]
                st.image(selected_img, caption=f"Selected: {selected_key}", use_column_width=True)
                if img_prompt := st.text_input("Ask a question about this image:", key=selected_key):
                    with st.spinner("Analyzing image..."):
                        response = llm_handler.query_ollama_with_image(selected_img, img_prompt)
                        st.info(response)

# --- App Execution ---
if __name__ == "__main__":
    initialize_session_state()
    setup_sidebar()
    main_interface()