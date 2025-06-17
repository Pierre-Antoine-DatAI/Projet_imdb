# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Ajouter le dossier parent pour importer ML.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.ML_2 import get_recommendations

# Créer l'app FastAPI
app = FastAPI(title="API Recommandations Films", version="1.0.0")

# Modèle pour la requête
class RecommendationRequest(BaseModel):
    title: str
    top_n: int = 10

# Endpoint principal
@app.post("/recommendations")
async def get_film_recommendations(request: RecommendationRequest):
    try:
        # Appeler ta fonction ML
        results = get_recommendations(request.title, request.top_n)
        
        # Si c'est une string d'erreur
        if isinstance(results, str):
            raise HTTPException(status_code=404, detail=results)
        
        # Convertir le DataFrame en dictionnaire pour l'API
        recommendations = results.to_dict('records')
        
        return {
            "film_recherche": request.title,
            "nombre_resultats": len(recommendations),
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

# Endpoint de santé
@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "API fonctionnelle"}

# Pour lancer : uvicorn main:app --reload --port 8000
# bash cd Projet_imdb
# uvicorn api.main:app --reload --port 8000