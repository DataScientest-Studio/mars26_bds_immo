import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Optimisation & Enrichissement",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Phase IV : Optimisation Avancée & Expertise Augmentée")
st.markdown("""
Cette page retrace l'ingénierie de pointe appliquée à notre modèle champion (**XGBoost**). 
Découvrez comment nous avons configuré nos hyperparamètres pour briser le *plafond de verre* statistique, 
ainsi que l'impact de l'injection de **données socio-économiques exogènes** (INSEE, Filosofi, Criminalité, Transports).
""")

# ==========================================
# SECTION 1 : STRATÉGIE D'OPTIMISATION
# ==========================================
st.header("1. Stratégie d'Optimisation : GridSearchCV & Early Stopping")

col_strat1, col_strat2 = st.columns([3, 2])

with col_strat1:
    st.markdown("""
    L'application d'un `GridSearchCV` classique sur un dataset de **3,7 millions de lignes** présente un coût computationnel exponentiel. Pour contourner ce point de blocage, nous avons déployé une **approche hybride** :
    
    *   **GridSearchCV Ciblé** : Pour fixer l'architecture globale optimale des arbres (profondeur, sous-échantillonnage).
    *   **Early Stopping (Arrêt Précoce)** : Intégré via l'API native d'XGBoost. En surveillant la fonction de perte ($RMSE$) sur un échantillon de validation interne, l'algorithme stoppe automatiquement l'ajout de nouveaux arbres dès que la performance stagne sur **10 rounds**.
    """)
    
    st.markdown("#### ⚙️ Configuration des Hyperparamètres Optimaux")
    st.json({
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    })

with col_strat2:
    st.success("""
    **💡 Compromis Biais / Variance :**
    *   `max_depth: 8` : Capture les relations non linéaires complexes sans générer d'instabilité.
    *   `learning_rate: 0.05` : Assure une convergence progressive et stable.
    *   `subsample / colsample: 0.8` : Introduit de la régularisation par sous-extraction, limitant drastiquement le surapprentissage (*overfitting*).
    """)
    st.metric(label="Temps Global d'Entraînement", value="15.5 min", delta="Prédiction : 12s (pour 1M de lignes)")

# --- VISUALISATION DU COMPROMIS DE CONVERGENCE ---
epochs = np.arange(1, 101)
train_rmse = 140000 - 20000 * np.log10(epochs)
test_rmse = 142000 - 19500 * np.log10(epochs) + (epochs * 15 if epochs > 60 else 0)

fig_conv = go.Figure()
fig_conv.add_trace(go.Scatter(x=epochs, y=train_rmse, mode='lines', name='Perte Entraînement (Train)', line=dict(color='#1f77b4')))
fig_conv.add_trace(go.Scatter(x=epochs, y=test_rmse, mode='lines', name='Perte Évaluation (Test)', line=dict(color='#ef553b')))
fig_conv.add_vline(x=62, line_dash="dash", line_color="green", annotation_text="Early Stopping Triggered (Round 62)")
fig_conv.update_layout(title="Mécanisme de l'Early Stopping : Protection contre le Surapprentissage", xaxis_title="Nombre d'itérations (n_estimators)", yaxis_title="Erreur (RMSE - €)", height=350)
st.plotly_chart(fig_conv, use_container_width=True)

st.markdown("---")

# ==========================================
# SECTION 2 : LE PLAFOND DE VERRE STATISTIQUE
# ==========================================
st.header("2. Le « Plafond de Verre » Statistique & Gains Bruts")

col_glass1, col_glass2 = st.columns([2, 1])

with col_glass1:
    st.markdown("""
    Malgré l'intégration d'algorithmes hautement paramétrés, les performances sur l'ensemble de test ont convergé de manière asymptotique vers un niveau seuil ($R^2 \\approx 74,6\\%$ ; $MAPE \\approx 34,2\\%$). 
    
    > **Diagnostic de l'action corrective :** Ce plafond n'est pas lié à un sous-apprentissage algorithmique, mais à la **nature même de l'information disponible**. Les résidus non captés se concentrent sur la *variance intrinsèque du marché* (négociations de gré à gré, état intérieur du bien, prestations de standing, vue dégagée), autant de micro-facteurs absents du fichier national DVF original.
    """)
    
    st.markdown("#### 📊 Évolution des Métriques (XGBoost)")
    metrics_comp = {
        "Métrique": ["R²", "MAE (€)", "RMSE (€)", "MAPE (%)"],
        "Avant Optimisation": [0.7463, 60739, 125930, "34.20 %"],
        "Après Optimisation": [0.7466, 60598, 125864, "34.13 %"]
    }
    st.table(pd.DataFrame(metrics_comp))

with col_glass2:
    st.warning("""
    **🛡️ Arbitrage Stratégique :**
    Au vu de cette convergence asymptotique, nous avons arbitré **contre** une complexification algorithmique stérile (comme le stacking lourd ou les réseaux de neurones profonds), extrêmement gourmande en ressources pour des gains marginaux. 
    """)

st.markdown("---")

# ==========================================
# SECTION 3 : VERS L'EXPERTISE ENRICHIE
# ==========================================
st.header("3. Vers une Expertise Immobilière Augmentée (Données Exogènes)")
st.markdown("""
Pour dépasser la géographie brute, nous avons interconnecté (via le `code_commune` et des jointures spatiales géocodées) la base DVF à des sources externes majeures :
*   **Données Socio-Économiques (INSEE / Filosofi)** : Densité de population, Revenu médian par ménage.
*   **Modélisation de la Valeur d'Usage** : Proximité des infrastructures scolaires, pôles de santé, commerces et distance kilométrique réelle à la gare voyageurs la plus proche.
""")

# Fichiers pivots du pipeline enrichi
st.caption("📦 Fichiers sources du pipeline enrichi : `X_train_enriched.csv`, `X_test_enriched.csv`, `y_train_enriched.csv`, `y_test_enriched.csv` (Variable cible : `Valeur_fonciere`)")

# Tableau des résultats du modèle enrichi
st.markdown("#### 📈 Classement des Modèles Évalués sur Données Enrichies")
enriched_results = {
    "Rang": [1, 2, 3, 4, 5, 6],
    "Modèle Enrichi": ["XGBoost", "Random Forest", "ExtraTrees", "LightGBM", "Linear Regression", "Ridge Regression"],
    "R² Test": [0.7204, 0.7174, 0.7149, 0.7102, 0.4818, 0.4818],
    "MAE (€)": ["65 381,53", "65 461,42", "66 401,87", "66 746,18", "94 516,07", "94 512,15"],
    "RMSE (€)": ["113 282,21", "113 874,73", "114 381,59", "115 315,34", "154 206,24", "154 206,98"],
    "MAPE": ["36,85 %", "36,90 %", "37,61 %", "37,78 %", "51,86 %", "51,86 %"],
    "Temps Entraînement": ["11.34 min", "115.23 min", "83.74 min", "6.24 min", "0.48 min", "0.15 min"]
}
st.table(pd.DataFrame(enriched_results))

st.markdown("---")

# ==========================================
# SECTION 4 : CONFRONTATION & APPORT MÉTIER
# ==========================================
st.header("4. Analyse Comparative & Décryptage Métier")

col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    st.subheader("❓ Pourquoi les métriques brutes ont-elles légèrement baissé ?")
    st.markdown("""
    L'analyse comparative montre que l'ajout des données externes n'a pas amélioré la performance prédictive brute des modèles d'arbres avancés (le $R^2$ d'XGBoost passe de `0.7466` à `0.7204`). 
    
    *   **Dilution du signal** : L'introduction de variables macro-économiques (comme le revenu médian ou la démographie d'une commune complète) dilue temporairement le poids des signaux géographiques ultra-locaux et des caractéristiques structurelles pures de la parcelle.
    *   **Permanence des fondamentaux** : Les variables immobilières classiques (Surface réelle, type de bien, coordonnées GPS directes, prix moyen au m² sectoriel) demeurent les prédicteurs rois de la valeur transactionnelle brute.
    """)

with col_comp2:
    st.subheader("🎯 La véritable valeur ajoutée : L'Interprétation Métier")
    st.markdown("""
    Cette baisse de performance pure ne signifie pas que les variables sont inutiles. Elles basculent le projet d'un simple outil de calcul vers un **Compagnon Immobilier Auditable** capable d'expliquer l'attractivité territoriale :
    """)
    
    # Présentation propre sous forme de dictionnaire des apports métiers
    business_value = {
        "Variable": ["Revenu_median", "Nb_equipements_total", "Distance_gare_plus_proche_km", "Taux_cambriolages & Vols", "Evolution_pop_5_ans"],
        "Apport Métier Direct": ["Pouvoir d'achat local et solvabilité des futurs acquéreurs", "Niveau de services de proximité et vie de quartier", "Indicateur clé d'accessibilité aux bassins d'emploi", "Indicateurs de sécurité globale et de sérénité résidentielle", "Mesure fine de l'attractivité et de la dynamique démographique"]
    }
    st.dataframe(pd.DataFrame(business_value), use_container_width=True, hide_index=True)

# ==========================================
# SYNTHÈSE DU RENDU II
# ==========================================
st.info("""
🏆 **Conclusion Générale du Rendu II :** 
Ce pipeline complet de Data Science valide le passage des modèles linéaires limités ($R^2 \\approx 45,2\\%$) vers une architecture d'arbres non linéaires régularisés ($R^2 \\approx 74,6\\%$). En combinant la robustesse du modèle **XGBoost optimisé** à la richesse des explications **SHAP**, nous livrons une chaîne algorithmique mature, industrialisable, transparente et parfaitement exploitable pour l'aide à la décision sur le marché immobilier de masse.
""")