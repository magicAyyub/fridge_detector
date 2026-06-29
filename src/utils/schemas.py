from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Literal, Optional


# ── Profil utilisateur (depuis l'onboarding) ───────────────────────────────

MealType = Literal["breakfast", "lunch", "dinner", "snack"]
SportsObjective = Literal["weight_loss", "muscle_gain", "maintenance", "endurance"]


class UserProfileInput(BaseModel):
    calorie_target: int = Field(
        description="Objectif calorique journalier (kcal) calculé à l'onboarding"
    )
    sports_objective: SportsObjective = Field(
        default="maintenance",
        description="Objectif sportif de l'utilisateur"
    )
    dietary_restrictions: List[str] = Field(
        default=[],
        description="Allergies/intolérances ex: ['gluten', 'lactose', 'nuts']"
    )


# Fraction de l'objectif journalier allouée à chaque repas
MEAL_CALORIE_RATIO: Dict[str, float] = {
    "breakfast": 0.25,
    "lunch":     0.35,
    "dinner":    0.35,
    "snack":     0.10,
}

# Cibles macro par objectif sportif
OBJECTIVE_TARGETS: Dict[str, Dict] = {
    "weight_loss": {
        "meal_ratio":    0.30,
        "min_protein_g": 20,
        "max_carbs_g":   40,
        "max_fat_g":     15,
    },
    "muscle_gain": {
        "meal_ratio":    0.35,
        "min_protein_g": 30,
        "max_carbs_g":   80,
        "max_fat_g":     25,
    },
    "maintenance": {
        "meal_ratio":    0.33,
        "min_protein_g": 15,
        "max_carbs_g":   70,
        "max_fat_g":     25,
    },
    "endurance": {
        "meal_ratio":    0.35,
        "min_protein_g": 15,
        "max_carbs_g":   100,
        "max_fat_g":     20,
    },
}


# ── Input recommandation ───────────────────────────────────────────────────

class IngredientsInput(BaseModel):
    fridge_dict: Optional[Dict[str, int]] = Field(
        default=None,
        description="Format fridge_detector : {label_EN: quantité}",
        examples=[{"tomato": 3, "egg": 4, "garlic": 2}],
    )
    ingredients: Optional[List[str]] = Field(
        default=None,
        description="Liste d'ingrédients EN (alternative à fridge_dict).",
    )
    user_profile: Optional[UserProfileInput] = Field(
        default=None,
        description="Profil utilisateur depuis l'onboarding pour filtrage calorique et objectif",
    )
    top_n: int = Field(default=15, ge=1, le=20)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def resolve_ingredients(self) -> "IngredientsInput":
        if self.fridge_dict:
            self.fridge_dict = {k.lower(): v for k, v in self.fridge_dict.items()}
            if not self.ingredients:
                self.ingredients = list(self.fridge_dict.keys())
        if not self.ingredients:
            raise ValueError("Fournir 'fridge_dict' ou 'ingredients'.")
        self.ingredients = [i.lower() for i in self.ingredients]
        return self


# ── Feedback ───────────────────────────────────────────────────────────────

class FeedbackInput(BaseModel):
    session_id: str = Field(description="Identifiant de session utilisateur")
    recipe_title: str = Field(description="Titre de la recette concernée")
    liked: Optional[bool] = Field(
        default=None,
        description="True = like, False = dislike, None = pas de note"
    )
    missing_ingredients: List[str] = Field(
        default=[],
        description="Ingrédients de la recette que l'utilisateur n'a pas"
    )
    cooked: bool = Field(
        default=False,
        description="L'utilisateur a cuisiné cette recette"
    )


class SubstituteRequest(BaseModel):
    ingredient: str = Field(description="Ingrédient à substituer")
    context_recipe: Optional[str] = Field(
        default=None,
        description="Titre de la recette pour contextualiser le substitut"
    )
    dietary_restrictions: List[str] = Field(
        default=[],
        description="Restrictions alimentaires à respecter pour le substitut"
    )


# ── Output recommandation ──────────────────────────────────────────────────

class RecipeStep(BaseModel):
    step: int
    instruction: str


class NutritionInfo(BaseModel):
    calories:        Optional[float] = Field(default=None, description="kcal")
    protein_g:       Optional[float] = None
    carbs_g:         Optional[float] = None
    total_fat_g:     Optional[float] = None
    sugar_g:         Optional[float] = None
    sodium_mg:       Optional[float] = None
    saturated_fat_g: Optional[float] = None


class RecipeResult(BaseModel):
    title: str
    score: float = Field(description="Score de correspondance [0, 1]")
    matched_ingredients: List[str]
    missing_ingredients: List[str]
    all_ingredients: List[str]
    steps: List[RecipeStep]
    nutrition: NutritionInfo
    meal_type: Optional[str] = None
    minutes: Optional[int] = None
    calorie_fit: Optional[str] = Field(
        default=None,
        description="'perfect' | 'low' | 'high' — adéquation avec l'objectif calorique"
    )


class RecipeRecommendationResponse(BaseModel):
    query_ingredients: List[str]
    fridge: Optional[Dict[str, int]] = None
    meal_type: Optional[str] = None
    recipes: List[RecipeResult]
    total_found: int = 0

    def model_post_init(self, __context):
        object.__setattr__(self, "total_found", len(self.recipes))


# ── Output feedback ────────────────────────────────────────────────────────

class SubstituteResult(BaseModel):
    original: str
    substitutes: List[str]
    notes: str


class FeedbackResponse(BaseModel):
    message: str
    substitutes: List[SubstituteResult] = []
