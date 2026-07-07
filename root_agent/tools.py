"""
create a translator tool that translates from english to telugu ,hindi ,etc for the agent upon the request from the user
"""

import os
from dotenv import load_dotenv
from google.genai import Client
from google.adk.tools import FunctionTool

load_dotenv()


def translate_text(text: str, target_language: str) -> dict:
    """
    Translate English text into the specified target language (e.g., Telugu, Hindi, Spanish, French, etc.).

    Args:
        text: The English text to be translated.
        target_language: The language to translate the text into (e.g., Telugu, Hindi, Spanish).

    Returns:
        A dictionary containing the translation under the key "translated_text".
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = Client(api_key=api_key)

    prompt = (
        f"Translate the following English text to {target_language}. "
        "Return only the translated text, preserving formatting and markdown if any.\n\n"
        f"Text:\n{text}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"translated_text": response.text.strip()}


translator_tool = FunctionTool(translate_text)
