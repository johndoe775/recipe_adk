from pydantic import BaseModel, Field


class Recipe(BaseModel):
    recipe_name: str = Field(description="Name of the recipe")
    ingredients: list[str] = Field(
        description="Ingredients with quantities"
    )
    prep_time: str = Field(description="Preparation time")
    cook_time: str = Field(description="Cooking time")
    total_time: str = Field(description="Total time required")
    instructions: list[str] = Field(
        description="Step-by-step cooking instructions"
    )