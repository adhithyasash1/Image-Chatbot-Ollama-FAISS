# app.py

import streamlit as st
import tempfile
import os
import io

# Import modularized functions
import config
import pdf_processor
import vector_store
import llm_handler

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Chatbot | PDF & Image", layout="wide")
st.title("📄🖼️ AI Chatbot for PDFs & Images")


# --- Session State Initialization ---
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'images' not in st.session_state:
        st.session_state.images = []


# --- UI Sidebar ---
def setup_sidebar():
    st.sidebar.header("⚙️ Setup")

    # PDF selection
    option = st.sidebar.radio(
        "Choose PDF Source:",
        ('Upload your own PDF', 'Use a default document'),
        key='pdf_option'
    )

    if option == 'Use a default document':
        domain = st.sidebar.selectbox("Choose a domain:", list(config.DEFAULT_PDFS.keys()))
        st.session_state.pdf_path = config.DEFAULT_PDFS[domain]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file:
            # Use a temporary file to store the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.pdf_path = tmp_file.name

    # Chunking settings
    st.sidebar.subheader("Chunking Settings")
    chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, config.DEFAULT_CHUNK_SIZE)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, config.DEFAULT_CHUNK_OVERLAP)

    # Process button
    if st.sidebar.button("Process PDF", use_container_width=True):
        if st.session_state.pdf_path:
            if os.path.exists(st.session_state.pdf_path):
                with st.spinner("Processing PDF... This may take a moment."):
                    # 1. Extract text and split into chunks
                    text_chunks = pdf_processor.extract_text_and_split(
                        st.session_state.pdf_path, chunk_size, chunk_overlap
                    )
                    # 2. Create vector store
                    st.session_state.vectorstore = vector_store.create_vector_store(text_chunks)
                    # 3. Extract images
                    st.session_state.images = pdf_processor.extract_images_from_pdf(st.session_state.pdf_path)

                st.success(f"PDF processed successfully! Found {len(st.session_state.images)} images.")
            else:
                st.error("Error: The selected PDF file does not exist.")
        else:
            st.warning("Please select or upload a PDF first.")


# --- Main Chat Interface ---
def main_interface():
    if not st.session_state.vectorstore:
        st.info("👋 Welcome! Please upload or select a PDF and click 'Process PDF' to begin.")
        return

    tab1, tab2 = st.tabs(["💬 Chat with Text", "🖼️ Chat with Images"])

    # --- Text Chat Tab ---
    with tab1:
        st.subheader("Chat with your PDF's Content")
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF content..."):
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

    # --- Image Chat Tab ---
    with tab2:
        st.subheader("Chat with Images in your PDF")
        if not st.session_state.images:
            st.warning("No images were found in the processed PDF.")
        else:
            image_choices = {f"Page {p_num}, Image {img_idx}": img for p_num, img_idx, img in st.session_state.images}
            selected_key = st.selectbox("Select an image to analyze:", list(image_choices.keys()))

            if selected_key:
                selected_image = image_choices[selected_key]
                st.image(selected_image, caption=f"Selected: {selected_key}", use_column_width=True)

                img_prompt = st.text_input("Ask a question about this image:", key=f"img_prompt_{selected_key}")
                if img_prompt:
                    with st.spinner("Analyzing image with LLaVA..."):
                        response = llm_handler.query_ollama_with_image(selected_image, img_prompt)
                        st.info(f"LLaVA's Response: {response}")


# --- Main Application Execution ---
if __name__ == "__main__":
    initialize_session_state()
    setup_sidebar()
    main_interface()