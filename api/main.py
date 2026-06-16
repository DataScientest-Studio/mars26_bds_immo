"""
API FastAPI — Compagnon Immobilier
Prédit le prix au m² et le prix total d'un bien immobilier
"""
import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ── Chargement des modèles au démarrage ───────────────────────────────────────
def load_models():
    meta = pickle.load(open(os.path.join(MODEL_DIR, "meta_6modeles.pkl"), "rb"))
    models = {}
    for tranche in ("bas", "milieu", "haut"):
        for t in ("maison", "appart"):
            path = os.path.join(MODEL_DIR, f"model_xgb_{tranche}_{t}.pkl")
            models[f"{tranche}_{t}"] = pickle.load(open(path, "rb"))
    return meta, models

meta, models = load_models()
FEATURES = meta["features"]
Q33, Q66  = float(meta["q33"]), float(meta["q66"])

# ── Utilitaires ───────────────────────────────────────────────────────────────
DPE_NUM = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"Non renseigne":4}

def dept_to_int(code: str) -> float:
    code = str(code).zfill(2).upper()
    if code == "2A": return 201.0
    if code == "2B": return 202.0
    try: return float(int(code))
    except: return 0.0

def get_tranche(pm2: float) -> str:
    if pm2 < Q33: return "bas"
    elif pm2 <= Q66: return "milieu"
    return "haut"

# ── Schéma de la requête ──────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    code_departement: str = Field(..., example="75", description="Code département (ex: 75, 13, 2A)")
    code_commune: str     = Field(..., example="75056", description="Code INSEE commune (5 chiffres)")
    type_bien: str        = Field(..., example="Appartement", description="'Appartement' ou 'Maison'")
    surface: float        = Field(..., gt=0, example=70.0, description="Surface en m²")
    nb_pieces: Optional[float] = Field(3.0, example=3.0, description="Nombre de pièces")
    dpe_classe: Optional[str]  = Field("Non renseigne", example="C", description="Classe DPE (A à G)")
    latitude: Optional[float]  = Field(None, example=48.8566)
    longitude: Optional[float] = Field(None, example=2.3522)
    dist_gare_km: Optional[float] = Field(None, example=0.5, description="Distance à la gare en km")

# ── Schéma de la réponse ──────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prix_m2_estime: float
    prix_total_estime: float
    tranche_marche: str
    modele_utilise: str

# ── App FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Compagnon Immobilier API",
    description="API de prédiction de prix immobilier — Projet MLOps Liora",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"message": "Compagnon Immobilier API", "status": "ok", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "modeles_charges": len(models)}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # Validation type de bien
    if req.type_bien not in ("Appartement", "Maison"):
        raise HTTPException(status_code=422, detail="type_bien doit être 'Appartement' ou 'Maison'")

    is_maison = 1 if req.type_bien == "Maison" else 0
    type_key  = "maison" if is_maison else "appart"
    dpe_val   = DPE_NUM.get(req.dpe_classe, 4)
    dept_val  = dept_to_int(req.code_departement)

    # Construction du vecteur de features
    row = {f: 0.0 for f in FEATURES}
    row["surface_reelle_bati"] = req.surface
    row["nombre_pieces_principales"] = req.nb_pieces or 3.0
    row["is_maison"]           = float(is_maison)
    row["code_departement"]    = dept_val
    row["dpe_classe_num"]      = float(dpe_val)

    if req.latitude  is not None: row["latitude"]  = req.latitude
    if req.longitude is not None: row["longitude"] = req.longitude
    if req.dist_gare_km is not None: row["dist_gare_km"] = req.dist_gare_km

    X = np.array([[row[f] for f in FEATURES]])

    # Première passe pour déterminer la tranche
    model_init = models.get(f"milieu_{type_key}")
    pm2_init   = float(model_init.predict(X)[0])
    tranche    = get_tranche(pm2_init)

    # Prédiction finale avec le bon modèle
    model_key  = f"{tranche}_{type_key}"
    model      = models.get(model_key)
    pm2        = float(model.predict(X)[0])
    prix_total = pm2 * req.surface

    return PredictionResponse(
        prix_m2_estime=round(pm2, 2),
        prix_total_estime=round(prix_total, 2),
        tranche_marche=tranche,
        modele_utilise=model_key,
    )
