
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from root_agent.tools import translator_tool, modify_recipe_tool

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from dotenv import load_dotenv

import os
import torch

load_dotenv()

# -----------------------------------
# Initialize model and Pinecone ONCE
# -----------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(
    MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("recipes")


# -----------------------------------
# Cached Query Embeddings
# -----------------------------------

@lru_cache(maxsize=1000)
def embed_query(query: str) -> tuple:
    """
    Cache embeddings for repeated queries.
    Tuple makes the cached result immutable.
    """
    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return tuple(vector.tolist())


# -----------------------------------
# Pinecone Search Tool
# -----------------------------------

def search_recipes(query: str) -> dict:
    """
    Search recipes from Pinecone and return top matches.
    Uses local cached embeddings and the default namespace.
    """

    query_vector = list(embed_query(query.strip().lower()))

    results = index.query(
        vector=query_vector,
        top_k=10,
        include_metadata=True
    )

    recipes = []

    for i, match in enumerate(results["matches"], start=1):
        metadata = match.get("metadata") or {}

        recipes.append({
            "id": match["id"],
            "rank": i,
            "score": match["score"],
            "title": metadata.get("title", f"Recipe {i}"),
            "content": metadata.get("content", "")
        })

    return {"recipes": recipes}


search_tool = FunctionTool(search_recipes)


# -----------------------------------
# Agent
# -----------------------------------

root_agent = Agent(
    name="recipe_agent",
    model="gemini-2.5-flash",
    tools=[
        search_tool,
        translator_tool,
        modify_recipe_tool
    ],
    instruction="""
You are a recipe assistant.

Workflow:

1. Ask the user what recipe they want.
2. Call search_recipes to retrieve relevant recipes.
3. Display the recipe choices as a numbered list.
4. Ask the user to select one recipe.
5. Ask whether they want modifications such as:
   - spicy
   - low oil
   - vegan
   - high protein
   - quick version

6. Generate the final recipe based on the selected recipe.
7. Return:
   - Recipe Name
   - Ingredients
   - Instructions
   - Cooking Time
   - Tips

8. If the user requests translation into another language,
   call translator_tool to perform the translation.

Do not invent ingredients or cooking instructions that
contradict the selected recipe.
"""
)
