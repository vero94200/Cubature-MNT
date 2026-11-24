import streamlit as st
import pandas as pd
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
    st.sidebar.info(f"La grille sera calculée avec des cellules de {resolution}m x {resolution}m.")

# Fonction pour lire le CSV de manière robuste (gère virgule et point-virgule)
def lire_csv_robuste(file):
    df = pd.read_csv(file, sep=',')
    if len(df.columns) == 1: # Si une seule colonne, c'est que le séparateur est mauvais
        file.seek(0)
        df = pd.read_csv(file, sep=';')
    
    # Nettoyage des noms de colonnes
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Accepter différentes variantes de noms (Est, X, Nord, Y, Altitude, Z, Elev)
    mapping = {
        'est': 'Est', 'x': 'Est', 'easting': 'Est',
        'nord': 'Nord', 'y': 'Nord', 'northing': 'Nord',
        'altitude': 'Altitude', 'z': 'Altitude', 'elev': 'Altitude', 'elevation': 'Altitude'
    }
    df.rename(columns=mapping, inplace=True)
    return df

# Bouton de calcul
if st.button("🚀 Lancer le calcul", type="primary", disabled=not (file_tn and file_proj)):
    try:
        with st.spinner("Traitement en cours..."):
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

            # Calcul
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
        st.subheader("📊 Aperçu des données importées")
        tab1, tab2 = st.tabs(["Terrain Naturel", "Terrain Projeté"])
        with tab1: st.dataframe(df_tn.head(10))
        with tab2: st.dataframe(df_proj.head(10))

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")