from root_agent.schema import Recipe
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from root_agent.tools import translator_tool, modify_recipe_tool

from pinecone import Pinecone
from functools import lru_cache
from dotenv import load_dotenv

import json
import os
import urllib.request

load_dotenv()

# -----------------------------------
# Lazy, cached initialization.
# Uses Hugging Face Inference API for all-MiniLM-L6-v2 embeddings
# to match existing Pinecone vectors without loading local model.
# -----------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_SERVICE_URL = "http://127.0.0.1:8001"


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
def get_embedding_client():
    """Return the configured external embedding service base URL."""
    service_url = os.getenv("EMBEDDING_SERVICE_URL", DEFAULT_EMBEDDING_SERVICE_URL).rstrip("/")
    if not service_url:
        raise ValueError("EMBEDDING_SERVICE_URL is missing from the environment")
    return {"base_url": service_url}


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
    Cache embeddings for repeated queries using the dedicated embedding service.
    Uses all-MiniLM-L6-v2 to match the existing Pinecone vectors.
    Tuple makes the cached result immutable.
    """
    client = get_embedding_client()
    service_url = client["base_url"]
    payload = json.dumps({"text": query.strip(), "normalize": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{service_url}/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    vector = data.get("embedding")
    if not vector:
        raise ValueError("Embedding service returned no embedding data")

    return tuple(vector)


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
    model="gemini-3.5-flash-lite",
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
