import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Preprocessing & Feature Engineering",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Phase II : Preprocessing & Feature Engineering")
st.markdown("""
Cette section détaille la valeur ajoutée de notre pipeline de données. Pour transformer des données brutes en variables hautement prédictives, 
nous avons résolu des paradoxes statistiques et fusionné **5 bases de données géographiques et démographiques externes**.
""")

PATH_DATA = r"C:\Users\carine\mars26_bds_immo\data\processed\dvf_clean_model_ready_optimized.csv"

@st.cache_data
def load_prep_data(path, n_samples=50000):
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        
        # Recréation ou sécurisation de l'indicateur has_terrain
        if 'has_terrain' not in df.columns and 'surface_terrain' in df.columns:
            df['has_terrain'] = df['surface_terrain'].apply(lambda x: 1 if x > 0 else 0)
        return df.sample(n_samples, random_state=42) if len(df) > n_samples else df
    except Exception as e:
        st.error(f"Erreur : {e}")
        return None

with st.spinner("Analyse du pipeline d'ingénierie..."):
    df_prep = load_prep_data(PATH_DATA)

if df_prep is not None:
    
    # ==========================================
    # SECTION 1 : LE PARADOXE DE L'EXTÉRIEUR
    # ==========================================
    st.header("1. Résolution du Paradoxe de l'Extérieur")
    
    col_vis, col_expl = st.columns([3, 2])
    
    with col_vis:
        st.subheader("Analyse des densités de probabilité")
        col_prix = 'prix_m2' if 'prix_m2' in df_prep.columns else ([c for c in df_prep.columns if 'prix' in c and 'm2' in c] + [None])[0]
        
        if col_prix:
            df_valid = df_prep[df_prep[col_prix] > 0].copy()
            df_valid['log_prix_m2'] = np.log10(df_valid[col_prix])
            
            t_col = 'has_terrain' if 'has_terrain' in df_valid.columns else 'surface_terrain'
            
            if t_col == 'surface_terrain':
                avec_t = df_valid[df_valid[t_col] > 0]['log_prix_m2'].dropna()
                sans_t = df_valid[df_valid[t_col] == 0]['log_prix_m2'].dropna()
            else:
                avec_t = df_valid[df_valid[t_col] == 1]['log_prix_m2'].dropna()
                sans_t = df_valid[df_valid[t_col] == 0]['log_prix_m2'].dropna()
                
            if not avec_t.empty and not sans_t.empty:
                fig_dist = ff.create_distplot(
                    [avec_t, sans_t],
                    group_labels=['Biens AVEC Terrain (Individuel)', 'Biens SANS Terrain (Collectif Urbain)'],
                    colors=['#1f77b4', '#ff7f0e'],
                    show_hist=False, show_rug=False
                )
                
                m_avec = avec_t.median()
                m_sans = sans_t.median()
                fig_dist.add_vline(x=m_avec, line_dash="dash", line_color="#1f77b4", annotation_text=f"Médiane: {m_avec:.2f}")
                fig_dist.add_vline(x=m_sans, line_dash="dash", line_color="#ff7f0e", annotation_text=f"Médiane: {m_sans:.2f}")
                
                fig_dist.update_layout(xaxis_title="Log10 (Prix au m²)", yaxis_title="Densité de distribution", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("Données de prix manquantes.")
            
    with col_expl:
        st.markdown("### L'explication statistique du paradoxe")
        st.error("**Le constat contre-intuitif** : Les statistiques descriptives affichent une valeur moyenne au m² supérieure pour les biens *sans extérieurs* (environ +1000€/m²).")
        st.markdown("""
        #### La Variable Cachée : La Densité Urbaine
        *   **Biens sans extérieur (Orange)** : Représentent les appartements situés en hyper-centre des métropoles économiques. La rareté foncière y dicte des prix au m² record.
        *   **Biens avec extérieur (Bleu)** : Représentent des structures individuelles situées en périphérie ou milieu rural, où le coût du m² bâti est mécaniquement plus faible.
        
        💡 **Action Feature Engineering** : Nous avons créé l'indicateur binaire `has_terrain`. Un modèle classique linéaire commettrait une erreur d'interprétation grave, tandis que nos architectures d'arbres (**XGBoost / LightGBM**) vont croiser nativement cette feature avec la localisation GPS pour rétablir la bonne prime de valeur.
        """)

    st.markdown("---")

    # ==========================================
    # SECTION 2 : TABLEAU D'ENRICHISSEMENT MULTI-SOURCES
    # ==========================================
    st.header("2. Architecture de l'Enrichissement Territorial")
    st.markdown("""
    Pour extraire de la valeur contextuelle, la base DVF de base a été fusionnée géographiquement à l'échelle de chaque commune avec **5 référentiels majeurs** de l'État :
    """)
    
    enrich_struct = {
        "Vecteur Territorial": ["📈 Démographie INSEE", "💰 Richesse Insee (Filosofi)", "🏪 Commerces & Services (BPE)", "🚨 Sécurité (Data.gouv)", "🚆 Accessibilité (SNCF)"],
        "Base Source d'Origine": [
            "DS_POPULATIONS_HISTORIQUES_data.csv",
            "DS_FILOSOFI_CC_2021_data.csv",
            "DS_BPE_2024_data.csv",
            "donnee-data.gouv-2025-geographie...",
            "liste-des-gares.csv"
        ],
        "Features Extraites & Injectées dans le Modèle": [
            "Historique de population (2013-2023) & Taux d'évolution (Attractivité)",
            "Revenu fiscal médian, Taux de pauvreté communal, Indice d'inégalité D9/D1",
            "Densité d'équipements de proximité par catégories (Santé, Éducation, Commerces)",
            "Indicateurs de criminalité 2024 (Taux de cambriolages, dégradations, atteintes aux personnes)",
            "Calcul des distances kilométriques via algorithme spatial BallTree (Indicateurs de proximité < 5km et 10km)"
        ]
    }
    
    st.table(pd.DataFrame(enrich_struct))
    
    # ==========================================
    # SECTION 3 : SÉCURITÉ ANTI DATA-LEAKAGE
    # ==========================================
    st.markdown("### 🔒 Protocole de Sécurisation du Pipeline")
    st.info("""
    **Alerte Data Leakage évitée :** Lors de la création de la feature de performance `commune_prix_m2`, le calcul du prix moyen de la commune a été effectué **strictement et uniquement sur le sous-ensemble d'Entraînement (Train Split)** après le fractionnement des données. 
    Ces valeurs moyennes ont ensuite été cartographiées sur le groupe de Test. Ce protocole strict empêche toute fuite d'information future vers le modèle, garantissant la fiabilité de nos métriques d'évaluation.
    """)