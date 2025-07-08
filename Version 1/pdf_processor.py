# pdf_processor.py

import fitz  # PyMuPDF
from PIL import Image
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def extract_text_and_split(file_path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Loads text from a PDF, splits it into documents, and then into chunks.

    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts

def extract_images_from_pdf(file_path: str) -> List[tuple]:
    """
    Extracts all images from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        List[tuple]: A list of tuples, where each tuple contains
                     (page_number, image_index, PIL.Image object).
    """
    images = []
    doc = fitz.open(file_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append((page_number + 1, img_index + 1, image))
    doc.close()
    return images