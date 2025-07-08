# Streamlit PDF & Image Chatbot

AI-powered chatbot built with Streamlit that allows you to have interactive conversations with your PDF documents. You can ask questions about the text content and also query specific images found within the PDF.

## Project Structure

```
/Image Chatbot with FAISS & LLaVA
├── default_pdfs/         # Folder for default PDF documents
├── app.py                # Main Streamlit application (UI layer)
├── pdf_processor.py      # Functions for text/image extraction
├── vector_store.py       # Functions for embedding and FAISS
├── llm_handler.py        # Functions for interacting with Ollama
├── config.py             # All configurations and settings
└── requirements.txt      # Project dependencies
```

## Setup and Installation

### 1. Prerequisites

You must have **Ollama** installed and running on your local machine.
-   [Download Ollama here](https://ollama.com/)
-   Once installed, pull the required `llava` model by running:
    ```bash
    ollama pull llava
    ```

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 3. Set up a Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the necessary Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Add Default PDFs

Create a folder named `default_pdfs` in the root of the project directory. Place any PDF files you want to use as default options inside this folder. The `config.py` file should be updated to point to these files.

## How to Run

With your virtual environment activated and Ollama running in the background, start the Streamlit application:

```bash
streamlit run app.py
```