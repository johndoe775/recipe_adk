from root_agent.schema import Recipe
from google.adk.models.lite_llm import LiteLlm
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
# Lazy, cached initialization.
# This keeps the same caching idea, but avoids heavy startup work
# during app import on Render.
# -----------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_groq_model():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from the environment")

    return LiteLlm(
        model="groq/openai/gpt-oss-20b",
        api_key=api_key,
    )


@lru_cache(maxsize=1)
def get_sentence_model():
    return SentenceTransformer(
        MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@lru_cache(maxsize=1)
def get_pinecone_index():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is missing from the environment")

    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index("recipes")


# -----------------------------------
# Cached Query Embeddings
# -----------------------------------

@lru_cache(maxsize=1000)
def embed_query(query: str) -> tuple:
    """
    Cache embeddings for repeated queries.
    Tuple makes the cached result immutable.
    """
    sentence_model = get_sentence_model()
    vector = sentence_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
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

    pinecone_index = get_pinecone_index()
    query_vector = list(embed_query(query.strip().lower()))

    results = pinecone_index.query(
        vector=query_vector,
        top_k=10,
        include_metadata=True,
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
    model="gemini-3.7-flash",
    #output_schema=Recipe,
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
