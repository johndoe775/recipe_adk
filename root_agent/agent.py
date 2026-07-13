from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from root_agent.tools import translator_tool, modify_recipe_tool

from pinecone import Pinecone
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

import os

load_dotenv()

# -----------------------------------
# Pinecone Search Tool
# -----------------------------------

def search_recipes(query: str) -> dict:
    """
    Search recipes from Pinecone and return top matches.
    """

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("recipes")

    hf_client = InferenceClient(
        provider="hf-inference",
        api_key=os.environ["HF_TOKEN"]
    )

    query_vector = hf_client.feature_extraction(
        query,
        model="sentence-transformers/all-MiniLM-L6-v2"
    ).tolist()

    results = index.query(
        vector=query_vector,
        top_k=10,
        include_metadata=True
    )

    recipes = []

    for i, match in enumerate(results["matches"], start=1):
        recipes.append({
            "id": i,
            "title": match["metadata"].get("title", f"Recipe {i}"),
            "content": match["metadata"]["content"]
        })

    return {"recipes": recipes}


search_tool = FunctionTool(search_recipes)

# -----------------------------------
# Agent
# -----------------------------------

root_agent = Agent(
    name="recipe_agent",
    model="gemini-2.5-flash",
    tools=[search_tool, translator_tool, modify_recipe_tool],
    instruction="""
You are a recipe assistant.

Workflow:

1. Ask the user what recipe they want.
2. Call search_recipes.
3. Display the recipe choices as a numbered list.
4. Ask the user to select one recipe.
5. Ask if they want modifications such as:
   - spicy
   - low oil
   - vegan
   - high protein
   - quick version

6. Generate the final recipe using the selected recipe.
7. Return:
   - Recipe Name
   - Ingredients
   - Instructions
   - Cooking Time
   - Tips

8. If the user requests the recipe or response to be translated into another language (e.g. Telugu, Hindi, Spanish, etc.), call the translate_text tool to perform the translation.
"""
)