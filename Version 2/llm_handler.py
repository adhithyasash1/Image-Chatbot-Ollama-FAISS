# llm_handler.py

import ollama
from PIL import Image
import io
import base64
import config

def query_ollama_with_image(image: Image.Image, query: str) -> str:
    """Queries Ollama with an image and text, using base64 encoding."""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': query,
                'images': [img_base64]
            }]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred while querying LLaVA: {e}"

def get_text_chat_response(vectorstore, query: str, chat_history: list) -> str:
    """Queries Ollama with context from the vector store for text-based chat."""
    try:
        context_docs = vectorstore.similarity_search(query, k=4)
        context = "\n".join([doc.page_content for doc in context_docs])

        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        prompt = f"""
        Use the following context from a PDF document and the conversation history to answer the question.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        History:
        {formatted_history}

        Question: {query}
        """

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred during chat: {e}"