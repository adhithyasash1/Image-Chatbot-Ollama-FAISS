# config.py

# 🧠 Model Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llava"  # Ensure you run `ollama pull llava`

# 📄 Text Chunking Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# 📁 Default PDF Documents
# Create a 'default_pdfs' folder and place your documents inside.
DEFAULT_PDFS = {
    'Data Science': './default_pdfs/data_science.pdf',
    'Tech Research': './default_pdfs/llms.pdf',
    # Add other default PDFs here
}