import streamlit as st
import pandas as pd
import numpy as np
from src.cubature import calculate_cubature_tin, calculate_cubature_grid

# Configuration de la page
st.set_page_config(page_title="Calcul de Cubature MNT", layout="wide", page_icon="🏗️")

st.title("🏗️ Calculateur de Cubature MNT")
st.markdown("Importez vos fichiers de points topographiques (CSV) pour calculer automatiquement les déblais et remblais.")

# Mise en page en colonnes pour les uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Terrain Naturel (TN)")
    file_tn = st.file_uploader("Glissez-déposez le fichier TN ici", type=["csv", "txt"])
    if file_tn is not None:
        st.success(f"✅ {file_tn.name} chargé")

with col2:
    st.subheader("2. Terrain Projeté")
    file_proj = st.file_uploader("Glissez-déposez le fichier Projet ici", type=["csv", "txt"])
    if file_proj is not None:
        st.success(f"✅ {file_proj.name} chargé")

# Barre latérale pour les paramètres
st.sidebar.header("⚙️ Paramètres de calcul")
methode = st.sidebar.radio("Méthode de calcul", ("TIN (Maillage triangulé)", "Grille (Raster)"))

resolution = 1.0
if methode.startswith("Grille"):
    resolution = st.sidebar.slider("Résolution de la grille (mètres)", 0.5, 5.0, 1.0, 0.1)
    st.sidebar.info(f"Grille calculée avec des cellules de {resolution}m x {resolution}m.")

st.sidebar.header("🌍 Système de coordonnées")
crs_choice = st.sidebar.selectbox(
    "Système de vos fichiers d'entrée :",
    options=[
        "Lambert-93 (EPSG:2154) - Standard France", 
        "WGS84 (EPSG:4326) - (Latitude/Longitude)", 
        "Lambert-93 CC47 (EPSG:3947) - Conique Conforme zone 47"
    ]
)

# Fonction pour lire le CSV de manière robuste
def lire_csv_robuste(file):
    df = pd.read_csv(file, sep=',')
    if len(df.columns) == 1: # Si une seule colonne, le séparateur est probablement un point-virgule
        file.seek(0)
        df = pd.read_csv(file, sep=';')
    
    # Nettoyage des noms de colonnes
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Accepter différentes variantes de noms
    mapping = {
        'est': 'Est', 'x': 'Est', 'easting': 'Est', 'longitude': 'Est', 'lon': 'Est',
        'nord': 'Nord', 'y': 'Nord', 'northing': 'Nord', 'latitude': 'Nord', 'lat': 'Nord',
        'altitude': 'Altitude', 'z': 'Altitude', 'elev': 'Altitude', 'elevation': 'Altitude', 'h': 'Altitude'
    }
    df.rename(columns=mapping, inplace=True)
    return df

# Bouton de calcul
if st.button("🚀 Lancer le calcul", type="primary", disabled=not (file_tn and file_proj)):
    try:
        with st.spinner("Lecture et vérification des coordonnées en cours..."):
            df_tn = lire_csv_robuste(file_tn)
            df_proj = lire_csv_robuste(file_proj)
            
            # Vérification des colonnes
            required = ['Est', 'Nord', 'Altitude']
            if not all(c in df_tn.columns for c in required):
                st.error(f"❌ Colonnes introuvables dans le fichier TN. Attendu: {required}. Trouvé: {list(df_tn.columns)}")
                st.stop()
            if not all(c in df_proj.columns for c in required):
                st.error(f"❌ Colonnes introuvables dans le fichier Projet. Attendu: {required}. Trouvé: {list(df_proj.columns)}")
                st.stop()

            # --- GESTION INTELLIGENTE DES SYSTÈMES DE COORDONNÉES ---
            warnings = []
            x_mean = abs(df_tn['Est'].mean())
            y_mean = abs(df_tn['Nord'].mean())
            
            if crs_choice.startswith("WGS84"):
                if x_mean > 1000 or y_mean > 1000:
                    warnings.append("⚠️ **Conflit détecté** : Vos coordonnées semblent être en mètres (>1000), mais vous avez sélectionné 'WGS84 (degrés)'. Veuillez vérifier votre fichier ou changer le système.")
                else:
                    from pyproj import Transformer
                    # Conversion WGS84 (Lon, Lat) -> Lambert-93 (Est, Nord)
                    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
                    df_tn['Est'], df_tn['Nord'] = transformer.transform(df_tn['Est'].values, df_tn['Nord'].values)
                    df_proj['Est'], df_proj['Nord'] = transformer.transform(df_proj['Est'].values, df_proj['Nord'].values)
                    warnings.append("✅ **Conversion réussie** : Coordonnées GPS transformées automatiquement en Lambert-93 (mètres) pour un calcul de volume correct.")
            
            elif crs_choice.startswith("Lambert") or crs_choice.startswith("Autre"):
                if x_mean < 360 and y_mean < 180: 
                    warnings.append("⚠️ **Conflit détecté** : Vos coordonnées ressemblent à des degrés (ex: 2.35, 48.85), mais vous avez choisi un système métrique. Le calcul de volume sera faux. Vérifiez votre fichier ou sélectionnez 'WGS84'.")

            # Affichage des avertissements avant calcul
            for w in warnings:
                st.warning(w)

            # Calcul des volumes
            st.info("Calcul des volumes en cours...")
            if methode.startswith("TIN"):
                deblais, remblais = calculate_cubature_tin(df_tn, df_proj)
            else:
                deblais, remblais = calculate_cubature_grid(df_tn, df_proj, resolution)

                # Affichage des résultats
        st.success("✅ Calcul terminé avec succès !")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Volume de DÉBLAIS", f"{deblais:,.2f} m³", help="Terrain à enlever")
        kpi2.metric("Volume de REMBLAIS", f"{remblais:,.2f} m³", help="Terrain à ajouter")
        
        balance = deblais - remblais
        kpi3.metric("Balance", f"{balance:,.2f} m³", help="Positif = Excédent de déblais, Négatif = Besoin de remblais")

        st.divider()
        st.subheader("📥 Export des résultats pour rapport")
        
        # Création d'un DataFrame récapitulatif propre
        df_resultat = pd.DataFrame({
            "Poste": ["Volume de DÉBLAIS", "Volume de REMBLAIS", "Balance (Déblais - Remblais)"],
            "Volume (m³)": [f"{deblais:,.2f}", f"{remblais:,.2f}", f"{balance:,.2f}"],
            "Système de coordonnées": [crs_choice, crs_choice, crs_choice],
            "Méthode de calcul": [methode, methode, methode]
        })
        
        st.dataframe(df_resultat, hide_index=True, use_container_width=True)
        
        # Génération du fichier Excel en mémoire
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_resultat.to_excel(writer, index=False, sheet_name='Cubature')
            df_tn.head(100).to_excel(writer, index=False, sheet_name='Apercu_TN')
            df_proj.head(100).to_excel(writer, index=False, sheet_name='Apercu_Projet')
        
        # Bouton de téléchargement
        st.download_button(
            label="📥 Télécharger le rapport Excel (.xlsx)",
            data=output.getvalue(),
            file_name="Rapport_Cubature_MNT.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.divider()
        st.subheader("📊 Aperçu des données traitées (5 premières lignes)")
        tab1, tab2 = st.tabs(["Terrain Naturel", "Terrain Projeté"])
        with tab1: st.dataframe(df_tn.head())
        with tab2: st.dataframe(df_proj.head())

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")
        st.info("💡 Vérifiez que vos fichiers sont bien des CSV avec des colonnes Est/X, Nord/Y et Altitude/Z.")
