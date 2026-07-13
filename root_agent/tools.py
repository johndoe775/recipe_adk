"""
create a translator tool that translates from english to telugu ,hindi ,etc for the agent upon the request from the user
"""

from google.adk.tools import FunctionTool
from root_agent.llm import client


def translate_text(text: str, target_language: str) -> dict:
    """
    Translate English text into the specified target language (e.g., Telugu, Hindi, Spanish, French, etc.).

    Args:
        text: The English text to be translated.
        target_language: The language to translate the text into (e.g., Telugu, Hindi, Spanish).

    Returns:
        A dictionary containing the translation under the key "translated_text".
    """
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


def modify_recipe(recipe_text: str, modifications: str) -> dict:
    """
    Modify/rewrite a recipe according to user requests (e.g., spicy, vegan, low-oil, quick version).

    Args:
        recipe_text: The full text or details of the original recipe.
        modifications: The specific changes or dietary preferences requested by the user.

    Returns:
        A dictionary containing the modified recipe under the key "modified_recipe".
    """
    prompt = (
        f"You are a professional chef. Modify the following recipe to accommodate these requests: {modifications}.\n\n"
        "Adjust ingredients, instructions, cooking time, and tips appropriately.\n"
        "Return the full modified recipe with these sections:\n"
        "- Recipe Name\n"
        "- Ingredients\n"
        "- Instructions\n"
        "- Cooking Time\n"
        "- Tips\n\n"
        f"Original Recipe:\n{recipe_text}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"modified_recipe": response.text.strip()}


translator_tool = FunctionTool(translate_text)
modify_recipe_tool = FunctionTool(modify_recipe)


