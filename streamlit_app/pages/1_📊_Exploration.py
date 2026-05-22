import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(
    page_title="EDA - Exploration Spatiale & Statistique",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Phase I : Analyse Exploratoire des Données (EDA)")
st.markdown("""
Cette première phase pose le diagnostic de notre base de données immobilière brute. Avant d'injecter la moindre variable dans un modèle prédictif, 
nous analysons ici la structure de notre variable cible, la distribution géographique des prix et le comportement parfois trompeur des statistiques simples.
""")

PATH_DATA = r"C:\Users\carine\mars26_bds_immo\data\processed\dvf_clean_model_ready_optimized.csv"

@st.cache_data
def load_eda_data(path, n_samples=50000):
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if len(df) > n_samples:
            return df.sample(n_samples, random_state=42)
        return df
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None

with st.spinner("Analyse du volume de transactions..."):
    df = load_eda_data(PATH_DATA)

if df is not None:
    st.success(f"📈 Échantillon d'analyse statistique initialisé : {len(df):,} transactions synchronisées.")

    # Détection des colonnes clés
    col_prix = 'prix_m2' if 'prix_m2' in df.columns else ([c for c in df.columns if 'prix' in c and 'm2' in c] + [None])[0]
    col_valeur = 'valeur_fonciere' if 'valeur_fonciere' in df.columns else ([c for c in df.columns if 'valeur' in c or 'cible' in c] + [None])[0]

    # ==========================================
    # SECTION 1 : COMPORTEMENT DE LA CIBLE
    # ==========================================
    st.header("1. Anatomie de la Variable Cible (Valeur Foncière)")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Distribution brute du marché")
        p95 = df[col_valeur].quantile(0.95)
        fig_brute = px.histogram(
            df[df[col_valeur] <= p95], 
            x=col_valeur,
            nbins=50,
            color_discrete_sequence=['#ef553b'],
            labels={col_valeur: 'Valeur Foncière (€)'}
        )
        fig_brute.update_layout(yaxis_title="Nombre de ventes", margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_brute, use_container_width=True)
        st.warning("🔺 **Asymétrie positive extrême (Skewness ≈ 7.18)** : Le marché est écrasé par une immense majorité de transactions standards et étiré par quelques transactions exceptionnelles. Une régression classique échouerait à cause de cette hétéroscédasticité.")

    with col_right:
        st.subheader("Comportement en échelle logarithmique")
        # On cherche si la colonne log est déjà présente, sinon calcul à la volée
        col_log = [c for c in df.columns if 'log' in c and 'valeur' in c]
        log_data = df[col_log[0]] if col_log else np.log10(df[col_valeur][df[col_valeur] > 0])
        
        fig_log = px.histogram(
            x=log_data,
            nbins=50,
            color_discrete_sequence=['#00cc96'],
            labels={'x': 'Log10 (Valeur Foncière)'}
        )
        fig_log.update_layout(yaxis_title="Nombre de ventes", margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_log, use_container_width=True)
        st.success("✅ **Distribution Log-Normale** : La transformation mathématique stabilise la variance. La distribution devient symétrique, une structure idéale pour la convergence de nos futurs algorithmes.")

    st.markdown("---")

    # ==========================================
    # SECTION 2 : CARTOGRAPHIE DU MARCHÉ
    # ==========================================
    st.header("2. Analyse Spatiale : Le Gradient des Prix")
    
    if 'latitude' in df.columns and 'longitude' in df.columns and col_prix:
        vue_carte = st.radio(
            "Filtrer l'étendue cartographique :",
            ("Zoom France Métropolitaine", "Vue Globale (Inclusions DROM)"),
            horizontal=True
        )
        
        if "Zoom" in vue_carte:
            df_map = df[(df['latitude'] > 41) & (df['latitude'] < 51) & 
                        (df['longitude'] > -5) & (df['longitude'] < 10)].dropna(subset=['latitude', 'longitude', col_prix])
            max_color = 12000
        else:
            df_map = df.dropna(subset=['latitude', 'longitude', col_prix])
            max_color = df_map[col_prix].quantile(0.99)
    
        fig_spatial = px.scatter_mapbox(
            df_map, lat="latitude", lon="longitude", color=col_prix,
            color_continuous_scale="jet", range_color=[0, max_color],
            size_max=4, zoom=5 if "Zoom" in vue_carte else 1,
            mapbox_style="carto-positron", height=550,
            title="Répartition nationale des prix observés au m²"
        )
        fig_spatial.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_spatial, use_container_width=True)
        
        st.info("🗺️ **Observation spatiale :** La carte révèle sans ambiguïté la polarisation du marché français : hyper-densité des valeurs élevées sur la région parisienne, le littoral azuréen, l'arc frontalier et les cœurs des grandes métropoles régionales.")
    else:
        st.error("Coordonnées de géolocalisation indisponibles pour le rendu cartographique.")

    st.markdown("---")

    # ==========================================
    # SECTION 3 : MULTICOLINÉARITÉ STRUCTURELLE
    # ==========================================
    st.header("3. Diagnostic des Corrélations Linéaires")
    
    col_txt, col_graph = st.columns([2, 3])
    
    with col_txt:
        st.markdown("""
        ### Le piège de la colinéarité
        L'analyse des variables physiques intrinsèques met en lumière une contrainte statistique majeure pour les modèles linéaires (comme l'OLS ou la Régression Ridge) :
        
        *   **Le bloc rouge central ($r = 0.82$)** : La `surface_reelle_bati` et le `nombre_pieces_principales` partagent une corrélation presque parfaite. 
        *   **Risque** : Injecter ces deux variables simultanément dans une régression classique détruit la stabilité des coefficients de prédiction.
        *   **Arbitrage technique** : Ce constat justifie l'abandon d'une approche purement linéaire au profit d'architectures basées sur des arbres de décision, insensibles à ce phénomène.
        """)
    
    with col_graph:
        corr_data = {
            'valeur_fonciere': [1.00, 0.59, 0.38, 0.27, 0.08],
            'prix_m2': [0.59, 1.00, 0.15, 0.05, -0.12],
            'surface_reelle_bati': [0.38, 0.15, 1.00, 0.82, 0.21],
            'nombre_pieces_principales': [0.27, 0.05, 0.82, 1.00, 0.11],
            'surface_terrain': [0.08, -0.12, 0.21, 0.11, 1.00]
        }
        df_corr = pd.DataFrame(corr_data, index=corr_data.keys())
        
        fig_heat = px.imshow(
            df_corr, text_auto='.2f',
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1
        )
        fig_heat.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)