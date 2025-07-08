# llm_handler.py

import ollama
from PIL import Image
import io
import base64
from . import config

def query_ollama_with_image(image: Image.Image, query: str) -> str:
    """
    Queries the Ollama LLaVA model with an image and a text prompt.
    This version encodes the image to base64 to avoid saving to disk.

    Args:
        image (Image.Image): The image to be analyzed.
        query (str): The user's question about the image.

    Returns:
        str: The content of the model's response.
    """
    try:
        # Convert PIL image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': query,
                    'images': [img_base64]
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred while querying LLaVA: {e}"


def get_text_chat_response(vectorstore, query: str, chat_history: list) -> str:
    """
    Queries the Ollama model with context from the vector store for a text-based chat.

    Args:
        vectorstore: The FAISS vector store containing the document context.
        query (str): The user's current question.
        chat_history (list): The history of the conversation.

    Returns:
        str: The model's response.
    """
    try:
        # Retrieve relevant context
        context_docs = vectorstore.similarity_search(query, k=4)
        context = "\n".join([doc.page_content for doc in context_docs])

        # Prepare chat history for the prompt
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        # Create a detailed prompt
        prompt = f"""
        You are a helpful AI assistant. Use the following context from a PDF document and the conversation history
        to answer the user's question. If you don't know the answer, just say that you don't know.

        Context:
        {context}

        Conversation History:
        {formatted_history}

        User Question:
        {query}

        Answer:
        """

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        return response['message']['content']

    except Exception as e:
        return f"An error occurred during chat: {e}"