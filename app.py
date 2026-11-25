import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata
from src.cubature import calculate_cubature_tin, calculate_cubature_grid

# ==========================================
# 1. 🔒 SYSTÈME D'AUTHENTIFICATION SIMPLE
# ==========================================
MOT_DE_PASSE_ATTENDU = ""  # <-- CHANGEZ CE MOT DE PASSE ICI

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Accès Restreint", page_icon="🔒")
    st.title("🔒 Accès Restreint - Cubature MNT")
    st.markdown("Veuillez entrer le mot de passe pour accéder à l'outil de calcul.")
    
    password = st.text_input("Mot de passe :", type="password")
    if st.button("Se connecter", type="primary"):
        if password == MOT_DE_PASSE_ATTENDU:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()

# ==========================================
# 2. APPLICATION PRINCIPALE
# ==========================================
st.set_page_config(page_title="Calcul de Cubature MNT", layout="wide", page_icon="🏗️")

st.title("🏗️ Calcul de Cubature MNT")

# --- SECTION FORMAT DES FICHIERS ---
with st.expander("📋 Format des fichiers CSV attendus (cliquez pour voir)", expanded=False):
    st.markdown("""
    ### 📄 Format requis pour vos fichiers CSV
    
    Vos fichiers doivent être au format **CSV** (séparateur virgule `,` ou point-virgule `;`) avec **3 colonnes obligatoires** :
    
    | Colonne | Noms acceptés | Description |
    |---------|---------------|-------------|
    | **Est** | `Est`, `X`, `Easting`, `Longitude`, `Lon` | Coordonnée Est (mètres ou degrés) |
    | **Nord** | `Nord`, `Y`, `Northing`, `Latitude`, `Lat` | Coordonnée Nord (mètres ou degrés) |
    | **Altitude** | `Altitude`, `Z`, `Elev`, `Elevation`, `H` | Altitude (mètres) |
    
    ### 📝 Exemple de fichier valide :
    ```csv
    Est,Nord,Altitude
    600000.50,6400000.20,150.50
    600005.00,6400002.10,151.20
    600010.30,6400005.80,149.80
    ```
    
    ### ✅ Compatibilité :
    - **Séparateurs** : virgule (`,`) ou point-virgule (`;`) détectés automatiquement
    - **En-têtes** : la première ligne doit contenir les noms de colonnes
    - **Décimales** : utiliser le point (`.`) pour les décimales (ex: `150.50`)
    - **Encodage** : UTF-8 recommandé
    
    ### 📥 Comment exporter depuis Excel :
    1. Ouvrez votre fichier Excel
    2. Cliquez sur **Fichier → Enregistrer sous**
    3. Choisissez le format **CSV (séparateur: point-virgule) (*.csv)**
    4. Assurez-vous que la première ligne contient les noms de colonnes
    """)

st.markdown("---")

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

# Barre latérale
st.sidebar.header("⚙️ Paramètres de calcul")
methode = st.sidebar.radio("Méthode de calcul", ("TIN (Maillage triangulé)", "Grille (Raster)"))

resolution = 1.0
if methode.startswith("Grille"):
    resolution = st.sidebar.slider("Résolution de la grille (m)", 0.5, 5.0, 1.0, 0.1)

st.sidebar.header("🌍 Système de coordonnées")
crs_choice = st.sidebar.selectbox(
    "Système de vos fichiers d'entrée :",
    options=["Lambert-93 (EPSG:2154)", "WGS84 (GPS - Latitude/Longitude)", "Lambert-93 CC47 (EPSG:3947)"]
)

st.sidebar.header("📐 Ajustement des volumes (Talutage / Foisonnement)")
foisonnement = st.sidebar.slider("Coeff. de foisonnement (Déblais)", 1.0, 1.5, 1.0, 0.05, help="1.3 = +30% de volume en camion")
compactage = st.sidebar.slider("Coeff. de compactage (Remblais)", 0.8, 1.0, 1.0, 0.05, help="0.9 = -10% de volume après compactage")

def lire_csv_robuste(file):
    df = pd.read_csv(file, sep=',')
    if len(df.columns) == 1:
        file.seek(0)
        df = pd.read_csv(file, sep=';')
    df.columns = [c.strip().lower() for c in df.columns]
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
        with st.spinner("Lecture, vérification et calcul en cours..."):
            df_tn = lire_csv_robuste(file_tn)
            df_proj = lire_csv_robuste(file_proj)
            
            required = ['Est', 'Nord', 'Altitude']
            if not all(c in df_tn.columns for c in required) or not all(c in df_proj.columns for c in required):
                st.error(f"❌ Colonnes introuvables. Attendu: {required}")
                st.stop()

            # Gestion des coordonnées
            warnings = []
            x_mean, y_mean = abs(df_tn['Est'].mean()), abs(df_tn['Nord'].mean())
            
            if crs_choice.startswith("WGS84"):
                if x_mean > 1000 or y_mean > 1000:
                    warnings.append("⚠️ Conflit : Coordonnées > 1000 mais système WGS84 sélectionné.")
                else:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
                    df_tn['Est'], df_tn['Nord'] = transformer.transform(df_tn['Est'].values, df_tn['Nord'].values)
                    df_proj['Est'], df_proj['Nord'] = transformer.transform(df_proj['Est'].values, df_proj['Nord'].values)
                    warnings.append("✅ Conversion GPS vers Lambert-93 réussie.")
            elif x_mean < 360 and y_mean < 180:
                warnings.append("⚠️ Conflit : Coordonnées ressemblent à des degrés, mais système métrique choisi.")

            for w in warnings:
                st.warning(w)

            # Calcul brut
            if methode.startswith("TIN"):
                deblais_brut, remblais_brut = calculate_cubature_tin(df_tn, df_proj)
            else:
                deblais_brut, remblais_brut = calculate_cubature_grid(df_tn, df_proj, resolution)

            # Application des coefficients
            deblais = deblais_brut * foisonnement
            remblais = remblais_brut * compactage
            balance = deblais - remblais

        # ==========================================
        # 3. AFFICHAGE DES RÉSULTATS
        # ==========================================
        st.success("✅ Calcul terminé avec succès !")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Volume de DÉBLAIS (ajusté)", f"{deblais:,.2f} m³", delta=f"{deblais_brut:,.2f} m³ (brut)" if foisonnement != 1.0 else None)
        kpi2.metric("Volume de REMBLAIS (ajusté)", f"{remblais:,.2f} m³", delta=f"{remblais_brut:,.2f} m³ (brut)" if compactage != 1.0 else None)
        kpi3.metric("Balance", f"{balance:,.2f} m³")

        # ==========================================
        # 4. GRAPHIQUES : PROFILS + CARTE
        # ==========================================
        st.subheader("📈 Profils en coupe (TN vs Projet)")
        st.info("Ces graphiques montrent les lignes du Terrain Naturel (bleu) et du Terrain Projeté (rouge) selon deux coupes.")
        
        # Préparation des données interpolées sur une grille
        x_min, x_max = df_tn['Est'].min(), df_tn['Est'].max()
        y_min, y_max = df_tn['Nord'].min(), df_tn['Nord'].max()
        
        res_map = max((x_max - x_min) / 100, (y_max - y_min) / 100, 1.0)
        xi = np.arange(x_min, x_max, res_map)
        yi = np.arange(y_min, y_max, res_map)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        zi_tn = griddata((df_tn['Est'], df_tn['Nord']), df_tn['Altitude'], (xi_grid, yi_grid), method='linear')
        zi_proj = griddata((df_proj['Est'], df_proj['Nord']), df_proj['Altitude'], (xi_grid, yi_grid), method='linear')
        
        # Coupe 1 : selon l'axe Est (à la valeur médiane de Nord)
        y_mid_idx = len(yi) // 2
        x_profile = xi
        z_tn_profile_1 = zi_tn[y_mid_idx, :]
        z_proj_profile_1 = zi_proj[y_mid_idx, :]
        
        # Coupe 2 : selon l'axe Nord (à la valeur médiane de Est)
        x_mid_idx = len(xi) // 2
        y_profile = yi
        z_tn_profile_2 = zi_tn[:, x_mid_idx]
        z_proj_profile_2 = zi_proj[:, x_mid_idx]
        
        # Affichage des deux profils côte à côte
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=x_profile, y=z_tn_profile_1, mode='lines', name='Terrain Naturel (TN)', line=dict(color='blue', width=2)))
            fig1.add_trace(go.Scatter(x=x_profile, y=z_proj_profile_1, mode='lines', name='Terrain Projeté', line=dict(color='red', width=2)))
            fig1.update_layout(
                title=f"Coupe Est-Ouest (à Nord = {yi[y_mid_idx]:.1f} m)",
                xaxis_title="Coordonnée Est (m)",
                yaxis_title="Altitude (m)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=y_profile, y=z_tn_profile_2, mode='lines', name='Terrain Naturel (TN)', line=dict(color='blue', width=2)))
            fig2.add_trace(go.Scatter(x=y_profile, y=z_proj_profile_2, mode='lines', name='Terrain Projeté', line=dict(color='red', width=2)))
            fig2.update_layout(
                title=f"Coupe Nord-Sud (à Est = {xi[x_mid_idx]:.1f} m)",
                xaxis_title="Coordonnée Nord (m)",
                yaxis_title="Altitude (m)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # --- CARTE DE RÉPARTITION ---
        st.subheader("🗺️ Carte de répartition des hauteurs (Déblai / Remblai)")
        
        diff = zi_tn - zi_proj
        
        df_map = pd.DataFrame({
            'Est': xi_grid.flatten(),
            'Nord': yi_grid.flatten(),
            'Différence (m)': diff.flatten()
        }).dropna()

        fig_map = px.scatter(df_map, x='Est', y='Nord', color='Différence (m)', 
                             color_continuous_scale='RdYlGn_r',
                             title="Visualisation des zones de Déblai (Rouge) et Remblai (Vert)",
                             labels={'Est': 'Coordonnée Est (m)', 'Nord': 'Coordonnée Nord (m)'})
        fig_map.update_traces(marker=dict(size=5, opacity=0.8))
        fig_map.update_layout(coloraxis_colorbar=dict(title="Hauteur (m)"))
        st.plotly_chart(fig_map, use_container_width=True)

        # --- EXPORT EXCEL ---
        st.divider()
        st.subheader("📥 Export des résultats pour rapport")
        
        df_resultat = pd.DataFrame({
            "Poste": ["Volume de DÉBLAIS (brut)", "Coeff. Foisonnement", "Volume de DÉBLAIS (ajusté)",
                      "Volume de REMBLAIS (brut)", "Coeff. Compactage", "Volume de REMBLAIS (ajusté)",
                      "Balance finale"],
            "Volume (m³)": [f"{deblais_brut:,.2f}", f"x {foisonnement}", f"{deblais:,.2f}",
                            f"{remblais_brut:,.2f}", f"x {compactage}", f"{remblais:,.2f}",
                            f"{balance:,.2f}"]
        })
        st.dataframe(df_resultat, hide_index=True, use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_resultat.to_excel(writer, index=False, sheet_name='Cubature')
            df_map.to_excel(writer, index=False, sheet_name='Maille_Carte')
            df_tn.head(100).to_excel(writer, index=False, sheet_name='Apercu_TN')
            df_proj.head(100).to_excel(writer, index=False, sheet_name='Apercu_Projet')
        
        st.download_button(
            label="📥 Télécharger le rapport Excel complet (.xlsx)",
            data=output.getvalue(),
            file_name="Rapport_Cubature_Pro.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")
