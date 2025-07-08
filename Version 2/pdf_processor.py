# pdf_processor.py

import fitz  # PyMuPDF
from PIL import Image
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def extract_text_and_split(file_path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Loads text from a PDF and splits it into chunks."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    return texts

def extract_images_from_pdf(file_path: str) -> List[tuple]:
    """Extracts all images from a PDF file."""
    images = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_num)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append((page_num + 1, img_index + 1, image))
        doc.close()
    except Exception as e:
        print(f"Error extracting images: {e}")
    return images