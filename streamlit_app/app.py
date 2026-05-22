import streamlit as st

# 1. Configuration de la page (Doit toujours être en premier)
st.set_page_config(
    page_title="Compagnon Immobilier",
    layout="wide",
    page_icon="🏡"
)

# 2. En-tête principal avec un peu de style
st.title("🏡 Compagnon Immobilier")
st.subheader("Prédiction des prix immobiliers en France")
st.caption("Projet de Machine Learning de Master — Basé sur les données de Demandes de Valeurs Foncières (DVF)")

# Une petite ligne de séparation verte (couleur primaire)
st.divider()

# 3. Structuration en 2 colonnes pour équilibrer l'écran large
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### 🎯 Objectifs du Projet")
    
    # Utilisation de st.info pour mettre en valeur les objectifs avec une touche de couleur
    st.info("**Explorer le marché** : Analyser les tendances, les prix au m² et l'impact de la géographie sur le marché immobilier français.")
    st.info("**Construire des modèles prédictifs** : Entraîner des algorithmes pour estimer la valeur d'un bien à partir de ses caractéristiques.")
    st.info("**Application métier interactive** : Mettre à disposition des utilisateurs un outil d'aide à la décision visuel et simple.")

with col2:
    st.markdown("### 🛠️ Stack Technique")
    
    # Présentation propre des technos sous forme de tags / liste stylisée
    st.markdown("""
    - **Language principale :** `Python`
    - **Data Wrangling :** `Pandas` & `NumPy`
    - **Modélisation ML :** `XGBoost` *(et Scikit-Learn)*
    - **Déploiement / UI :** `Streamlit`
    """)
    
    # Un petit clin d'œil académique pour le jury
    st.success("💡 **Note pour le jury** : Les données DVF ")