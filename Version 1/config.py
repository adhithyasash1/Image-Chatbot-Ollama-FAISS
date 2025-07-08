# config.py

# Model Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llava" # Ensure you have pulled this model with `ollama pull llava`

# Text Chunking Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Default PDF Documents
# Create a folder named 'default_pdfs' and place your documents inside it.
DEFAULT_PDFS = {
    'Data Science': './default_pdfs/data_science.pdf',
    'Business': './default_pdfs/business.pdf',
    # Add more default PDFs here
}