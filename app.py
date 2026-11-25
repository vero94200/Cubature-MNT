import csv
import io
import re

import numpy as np
import pandas as pd
import streamlit as st

from src.cubature import calculate_cubature_tin, calculate_cubature_grid

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Cubature MNT", layout="wide", page_icon="🏗️")

REQUIRED = ["Est", "Nord", "Altitude"]
MAPPING = {
    "est": "Est", "x": "Est", "easting": "Est", "e": "Est",
    "nord": "Nord", "y": "Nord", "northing": "Nord", "n": "Nord",
    "altitude": "Altitude", "z": "Altitude", "elev": "Altitude",
    "elevation": "Altitude", "alt": "Altitude", "h": "Altitude",
}

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 14px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Lecture et nettoyage des fichiers
# ----------------------------------------------------------------------------
def lire_points(file) -> tuple[pd.DataFrame, dict]:
    """Lit un fichier de points (séparateur et décimale détectés automatiquement).

    Retourne le DataFrame nettoyé (Est, Nord, Altitude) et un dict d'infos
    sur le nettoyage (lignes invalides, doublons...).
    """
    raw = file.getvalue().decode("utf-8-sig", errors="replace")
    lignes = [l for l in raw.splitlines() if l.strip()]
    if len(lignes) < 2:
        raise ValueError("Le fichier est vide ou ne contient qu'une ligne.")

    echantillon = "\n".join(lignes[:20])
    try:
        sep = csv.Sniffer().sniff(echantillon, delimiters=";,\t ").delimiter
    except csv.Error:
        sep = ";" if echantillon.count(";") > echantillon.count(",") else ","

    # Virgule décimale (ex. "12,5;45,2;101,3")
    decimal = "," if sep != "," and re.search(r"\d,\d", echantillon) else "."

    # Fichier sans en-tête ? (la 1re cellule est un nombre)
    premiere = lignes[0].split(sep)[0].strip().replace(",", ".")
    try:
        float(premiere)
        sans_entete = True
    except ValueError:
        sans_entete = False

    sep_regex = r"\s+" if sep == " " else sep
    if sans_entete:
        df = pd.read_csv(io.StringIO(raw), sep=sep_regex, decimal=decimal,
                         header=None, engine="python")
        if df.shape[1] < 3:
            raise ValueError("Au moins 3 colonnes sont nécessaires (Est, Nord, Altitude).")
        df = df.iloc[:, -3:]
        df.columns = REQUIRED
    else:
        df = pd.read_csv(io.StringIO(raw), sep=sep_regex, decimal=decimal, engine="python")
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns=MAPPING)

    manquantes = [c for c in REQUIRED if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes introuvables : {manquantes}. Colonnes trouvées : {list(df.columns)}"
        )

    df = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
    n_brut = len(df)
    df = df.dropna()
    n_invalides = n_brut - len(df)

    # Doublons XY : on moyenne les altitudes (évite les erreurs de triangulation)
    n_avant = len(df)
    df = df.groupby(["Est", "Nord"], as_index=False)["Altitude"].mean()
    n_doublons = n_avant - len(df)

    if len(df) < 3:
        raise ValueError("Moins de 3 points valides : impossible de calculer un MNT.")

    infos = {"lignes": n_brut, "invalides": n_invalides, "doublons": n_doublons,
             "separateur": repr(sep), "decimale": decimal}
    return df, infos


def emprise(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return df["Est"].min(), df["Nord"].min(), df["Est"].max(), df["Nord"].max()


def recouvrement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Part (0-1) de l'emprise du projet couverte par l'emprise du TN."""
    ax0, ay0, ax1, ay1 = emprise(a)
    bx0, by0, bx1, by1 = emprise(b)
    w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    h = max(0.0, min(ay1, by1) - max(ay0, by0))
    aire_b = (bx1 - bx0) * (by1 - by0)
    return (w * h) / aire_b if aire_b > 0 else 0.0


@st.cache_data(show_spinner=False)
def calculer(df_tn, df_proj, methode, resolution):
    if methode == "tin":
        return calculate_cubature_tin(df_tn, df_proj)
    return calculate_cubature_grid(df_tn, df_proj, resolution)


def carte_points(df: pd.DataFrame):
    """Nuage de points coloré par altitude (échantillonné si volumineux)."""
    if len(df) > 5000:
        df = df.sample(5000, random_state=0)
    st.scatter_chart(df, x="Est", y="Nord", color="Altitude", height=320)


def resume(df: pd.DataFrame, infos: dict):
    x0, y0, x1, y1 = emprise(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points valides", f"{len(df):,}".replace(",", " "))
    c2.metric("Alt. min", f"{df['Altitude'].min():.2f} m")
    c3.metric("Alt. max", f"{df['Altitude'].max():.2f} m")
    c4.metric("Emprise", f"{x1 - x0:.0f} × {y1 - y0:.0f} m")
    if infos["invalides"] or infos["doublons"]:
        st.caption(
            f"⚠️ {infos['invalides']} ligne(s) invalide(s) ignorée(s), "
            f"{infos['doublons']} doublon(s) XY fusionné(s)."
        )


# ----------------------------------------------------------------------------
# Barre latérale
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Paramètres")
choix = st.sidebar.radio(
    "Méthode de calcul",
    ("TIN (maillage triangulé)", "Grille (raster)"),
    help="TIN : plus précis, suit la topographie réelle. Grille : plus rapide sur de gros jeux de données.",
)
methode = "tin" if choix.startswith("TIN") else "grille"

resolution = 1.0
if methode == "grille":
    resolution = st.sidebar.slider("Résolution de la grille (m)", 0.5, 5.0, 1.0, 0.1)
    st.sidebar.info(f"Cellules de {resolution} m × {resolution} m.")

st.sidebar.subheader("Coefficients")
foisonnement = st.sidebar.number_input(
    "Foisonnement des déblais (%)", 0.0, 60.0, 0.0, 1.0,
    help="Augmentation de volume des terres une fois excavées (ex. 15 % pour de l'argile).",
)
tassement = st.sidebar.number_input(
    "Tassement des remblais (%)", 0.0, 30.0, 0.0, 1.0,
    help="Réduction de volume des remblais après compactage.",
)
prix_m3 = st.sidebar.number_input("Coût de transport (€/m³, optionnel)", 0.0, 500.0, 0.0, 0.5)

# ----------------------------------------------------------------------------
# En-tête et import
# ----------------------------------------------------------------------------
st.title("🏗️ Calculateur de Cubature MNT")
st.caption("Importez le terrain naturel et le terrain projeté pour obtenir déblais, remblais et balance.")

col1, col2 = st.columns(2)
donnees = {}

for col, cle, titre in ((col1, "tn", "1. Terrain naturel (TN)"),
                        (col2, "proj", "2. Terrain projeté")):
    with col:
        st.subheader(titre)
        f = st.file_uploader(f"Fichier {titre}", type=["csv", "txt"], key=cle,
                             label_visibility="collapsed")
        if f is not None:
            try:
                df, infos = lire_points(f)
                donnees[cle] = (f.name, df, infos)
                st.success(f"✅ {f.name}")
                resume(df, infos)
                carte_points(df)
            except Exception as e:
                st.error(f"❌ {f.name} : {e}")

pret = "tn" in donnees and "proj" in donnees

if not pret:
    st.info("📂 Chargez les deux fichiers (colonnes Est/X, Nord/Y, Altitude/Z) pour activer le calcul.")
    with st.expander("Format attendu"):
        st.code("Est;Nord;Altitude\n651234,50;6861234,25;102,35\n651235,50;6861234,25;102,41", language="text")
        st.caption("Séparateur `,` `;` tabulation ou espace, virgule ou point décimal : détectés automatiquement. "
                   "Les fichiers sans en-tête sont acceptés (3 dernières colonnes = Est, Nord, Altitude).")

# ----------------------------------------------------------------------------
# Contrôles de cohérence
# ----------------------------------------------------------------------------
if pret:
    _, df_tn, _ = donnees["tn"]
    _, df_proj, _ = donnees["proj"]
    taux = recouvrement(df_tn, df_proj)
    if taux < 0.05:
        st.error("❌ Les deux emprises ne se recouvrent pas : vérifiez le système de coordonnées.")
    elif taux < 0.9:
        st.warning(f"⚠️ Le TN ne couvre que {taux:.0%} de l'emprise du projet : les volumes seront partiels.")

# ----------------------------------------------------------------------------
# Calcul
# ----------------------------------------------------------------------------
if st.button("🚀 Lancer le calcul", type="primary", disabled=not pret):
    try:
        with st.spinner("Calcul en cours..."):
            deb, rem = calculer(df_tn, df_proj, methode, resolution)
        st.session_state["resultat"] = {
            "deblais": float(deb), "remblais": float(rem),
            "methode": choix, "resolution": resolution,
            "tn": donnees["tn"][0], "proj": donnees["proj"][0],
        }
    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")

# ----------------------------------------------------------------------------
# Résultats (conservés dans la session : pas de perte au changement de paramètre)
# ----------------------------------------------------------------------------
res = st.session_state.get("resultat")
if res:
    st.divider()
    st.subheader("📊 Résultats")

    deb = res["deblais"]
    rem = res["remblais"]
    deb_f = deb * (1 + foisonnement / 100)
    rem_t = rem * (1 + tassement / 100)  # volume à approvisionner
    balance = deb_f - rem_t

    k1, k2, k3 = st.columns(3)
    k1.metric("Déblais", f"{deb:,.2f} m³".replace(",", " "), help="Terrain à enlever (foisonné ci-dessous)")
    k2.metric("Remblais", f"{rem:,.2f} m³".replace(",", " "), help="Terrain à ajouter")
    k3.metric("Balance", f"{balance:,.2f} m³".replace(",", " "),
              delta="Excédent à évacuer" if balance > 0 else "Apport extérieur nécessaire",
              delta_color="off",
              help="Déblais foisonnés − remblais tassés")

    if foisonnement or tassement:
        c1, c2 = st.columns(2)
        c1.metric("Déblais foisonnés", f"{deb_f:,.2f} m³".replace(",", " "))
        c2.metric("Remblais à approvisionner", f"{rem_t:,.2f} m³".replace(",", " "))

    if prix_m3 > 0:
        st.metric("Coût de transport estimé", f"{abs(balance) * prix_m3:,.0f} €".replace(",", " "),
                  help="Calculé sur le volume absolu de la balance.")

    chart_df = pd.DataFrame({"Volume (m³)": [deb, rem]}, index=["Déblais", "Remblais"])
    st.bar_chart(chart_df, height=240)

    rapport = pd.DataFrame(
        {
            "Paramètre": ["Fichier TN", "Fichier projet", "Méthode", "Résolution (m)",
                          "Déblais (m³)", "Remblais (m³)", "Foisonnement (%)", "Tassement (%)",
                          "Déblais foisonnés (m³)", "Remblais à approvisionner (m³)", "Balance (m³)"],
            "Valeur": [res["tn"], res["proj"], res["methode"],
                       res["resolution"] if res["methode"].startswith("Grille") else "—",
                       round(deb, 2), round(rem, 2), foisonnement, tassement,
                       round(deb_f, 2), round(rem_t, 2), round(balance, 2)],
        }
    )
    st.download_button(
        "📥 Télécharger le rapport (CSV)",
        rapport.to_csv(index=False, sep=";").encode("utf-8-sig"),
        file_name="rapport_cubature.csv",
        mime="text/csv",
    )

    if pret:
        with st.expander("Aperçu des données importées"):
            t1, t2 = st.tabs(["Terrain naturel", "Terrain projeté"])
            t1.dataframe(df_tn.head(50), use_container_width=True)
            t2.dataframe(df_proj.head(50), use_container_width=True)
