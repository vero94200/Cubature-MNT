import csv
import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cubature import (
    calculate_cubature_tin,
    calculate_cubature_grid,
    compute_difference_grid,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Cubature MNT", layout="wide", page_icon="🏗️")

REQUIRED = ["Est", "Nord", "Altitude"]
MAPPING = {
    "est": "Est", "x": "Est", "easting": "Est", "e": "Est",
    "nord": "Nord", "y": "Nord", "northing": "Nord", "n": "Nord",
    "lon": "Est", "long": "Est", "lng": "Est", "longitude": "Est",
    "lat": "Nord", "latitude": "Nord",
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
        padding: 10px 12px;
    }
    /* Métriques : texte réduit et retour à la ligne au lieu de la coupure "..." */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        font-size: 1.1rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 0.78rem !important;
    }
    /* Zone d'import : textes et bouton plus petits, sur plusieurs lignes */
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] {
        flex-wrap: wrap;
        gap: 8px;
        padding: 10px 12px;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        font-size: 0.8rem !important;
        white-space: normal !important;
    }
    div[data-testid="stFileUploaderDropzone"] button {
        font-size: 0.8rem !important;
        white-space: normal !important;
        height: auto;
        padding: 4px 10px;
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


def semble_wgs84(df: pd.DataFrame) -> bool:
    """Heuristique : valeurs dans les bornes lon/lat ET emprise de quelques degrés au plus."""
    est, nord = df["Est"], df["Nord"]
    return bool(
        est.between(-180, 180).all() and nord.between(-90, 90).all()
        and (est.max() - est.min()) < 5 and (nord.max() - nord.min()) < 5
    )


def epsg_utm(lon: float, lat: float) -> int:
    """Code EPSG du fuseau UTM (WGS84) contenant le point."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def wgs84_vers_metres(df: pd.DataFrame, epsg_cible: int) -> pd.DataFrame:
    """Convertit Est=longitude / Nord=latitude (EPSG:4326) vers un CRS projeté en mètres."""
    from pyproj import Transformer

    tr = Transformer.from_crs(4326, epsg_cible, always_xy=True)
    x, y = tr.transform(df["Est"].to_numpy(), df["Nord"].to_numpy())
    out = df.copy()
    out["Est"], out["Nord"] = x, y
    return out


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


@st.cache_data(show_spinner=False)
def calculer_ecarts(df_tn, df_proj, resolution):
    return compute_difference_grid(df_tn, df_proj, resolution)


def resolution_carte(df_tn, df_proj, methode, resolution):
    """Pas de la grille de la carte : celui de la grille, sinon ~400 cellules sur le grand côté."""
    if methode == "grille":
        return resolution
    w = min(df_tn["Est"].max(), df_proj["Est"].max()) - max(df_tn["Est"].min(), df_proj["Est"].min())
    h = min(df_tn["Nord"].max(), df_proj["Nord"].max()) - max(df_tn["Nord"].min(), df_proj["Nord"].min())
    return max(max(w, h, 1.0) / 400, 0.05)


JAUNE = np.array([255, 221, 0], dtype=float)   # déblais
ROUGE = np.array([220, 30, 30], dtype=float)   # remblais
NEUTRE = np.array([130, 130, 130], dtype=float)
FOND = np.array([30, 34, 40], dtype=np.uint8)  # hors données


def figure_carte(xi, yi, diff: np.ndarray, seuil: float) -> go.Figure:
    """Carte interactive (zoom, déplacement, survol) : jaune = déblais, rouge = remblais, gris = négligeable.

    La couleur est plus vive quand l'épaisseur à déblayer/remblayer est grande.
    """
    # Sous-échantillonnage si trop dense (fluidité de l'affichage)
    pas = int(np.ceil(max(diff.shape) / 800))
    if pas > 1:
        diff, xi, yi = diff[::pas, ::pas], xi[::pas], yi[::pas]

    valid = ~np.isnan(diff)
    vmax = float(np.percentile(np.abs(diff[valid]), 95)) if valid.any() else 1.0
    vmax = max(vmax, 1e-6, seuil * 1.001)

    t = min(seuil / (2 * vmax), 0.499)
    a, b = 0.5 - t, 0.5 + t

    def rgb(c, k=1.0):
        return "rgb({},{},{})".format(*(int(v * k) for v in c))

    colorscale = [
        [0.0, rgb(ROUGE)], [a, rgb(ROUGE, 0.55)],
        [a, rgb(NEUTRE)], [b, rgb(NEUTRE)],
        [b, rgb(JAUNE, 0.55)], [1.0, rgb(JAUNE)],
    ]

    fig = go.Figure(
        go.Heatmap(
            x=xi, y=yi, z=diff,
            zmin=-vmax, zmax=vmax, colorscale=colorscale,
            hoverongaps=False,
            hovertemplate="Est : %{x:.1f} m<br>Nord : %{y:.1f} m<br>Écart : %{z:+.2f} m<extra></extra>",
            colorbar=dict(title="Écart (m)", thickness=14),
        )
    )
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor=rgb(FOND),
        xaxis=dict(title="Est (m)", showgrid=False, tickformat=".0f"),
        yaxis=dict(title="Nord (m)", showgrid=False, tickformat=".0f",
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def legende_carte(seuil: float):
    st.markdown(
        f"""
        <div style="display:flex;gap:22px;flex-wrap:wrap;font-size:0.9rem">
          <span><span style="display:inline-block;width:14px;height:14px;background:rgb(255,221,0);border-radius:3px;vertical-align:middle"></span> Déblais (TN au-dessus du projet)</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:rgb(220,30,30);border-radius:3px;vertical-align:middle"></span> Remblais (TN sous le projet)</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:rgb(130,130,130);border-radius:3px;vertical-align:middle"></span> Écart &lt; {seuil:.2f} m</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:rgb(30,34,40);border-radius:3px;vertical-align:middle"></span> Hors données</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def surface(df: pd.DataFrame) -> float:
    """Surface (m²) de l'enveloppe convexe des points ; emprise rectangulaire à défaut."""
    try:
        from scipy.spatial import ConvexHull

        return float(ConvexHull(df[["Est", "Nord"]].to_numpy()).volume)  # en 2D, .volume = aire
    except Exception:
        x0, y0, x1, y1 = emprise(df)
        return float((x1 - x0) * (y1 - y0))


def resume(df: pd.DataFrame, infos: dict):
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("Points", f"{len(df):,}".replace(",", " "))
    c2.metric("Alt. min", f"{df['Altitude'].min():.2f} m")
    c3.metric("Alt. max", f"{df['Altitude'].max():.2f} m")
    c4.metric("Surface", f"{surface(df):,.0f} m²".replace(",", " "),
              help="Aire de l'enveloppe convexe des points.")
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

st.sidebar.subheader("Coordonnées")
mode_crs = st.sidebar.selectbox(
    "Système des fichiers",
    ("Détection automatique", "Projeté (mètres)", "WGS84 lon/lat (EPSG:4326)"),
    index=2,  # défaut : WGS84 lon/lat (EPSG:4326)
    help="Le calcul de volumes exige des mètres : les coordonnées WGS84 (degrés) sont "
         "converties automatiquement vers une projection métrique.",
)
inverser_latlon = st.sidebar.checkbox(
    "Colonnes en ordre latitude, longitude",
    help="À cocher si vos fichiers WGS84 sans en-tête sont au format lat, lon, altitude.",
)
choix_proj = st.sidebar.selectbox(
    "Projection de sortie (WGS84 → mètres)",
    ("UTM automatique", "UTM 30N (EPSG:32630)", "UTM 31N (EPSG:32631)",
     "UTM 32N (EPSG:32632)", "Autre code EPSG…"),
    index=1,  # défaut : UTM 30N (EPSG:32630)
    help="Automatique : fuseau UTM calculé depuis le centre du fichier TN. "
         "Choisissez 30, 31 ou 32 pour l'imposer (utile près d'une limite de fuseau).",
)
epsg_saisi = ""
if choix_proj.startswith("Autre"):
    epsg_saisi = st.sidebar.text_input(
        "Code EPSG", "",
        help="Exemples : 2154 (Lambert-93), 3812 (Lambert 2008, Belgique), 32633 (UTM 33N).",
    )

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

st.sidebar.subheader("Carte")
seuil = st.sidebar.number_input(
    "Seuil d'écart négligeable (m)", 0.0, 1.0, 0.02, 0.01,
    help="Les écarts inférieurs à ce seuil apparaissent en gris sur la carte (n'influence pas les volumes).",
)

# ----------------------------------------------------------------------------
# En-tête et import
# ----------------------------------------------------------------------------
st.title("🏗️ Calcul de Cubature MNT")
st.caption("Importez le terrain naturel et le terrain projeté pour obtenir déblais, remblais et balance.")

col1, col2 = st.columns(2)
donnees = {}
cols = (("tn", col1, "1. Terrain naturel (TN)"), ("proj", col2, "2. Terrain projeté"))

# Phase 1 : upload et lecture brute
bruts = {}
fichiers = {}
for cle, col, titre in cols:
    with col:
        st.subheader(titre)
        fichiers[cle] = st.file_uploader(f"Fichier {titre}", type=["csv", "txt"], key=cle,
                                         label_visibility="collapsed")
        if fichiers[cle] is not None:
            try:
                bruts[cle] = lire_points(fichiers[cle])
            except Exception as e:
                st.error(f"❌ {fichiers[cle].name} : {e}")

# Phase 2 : détection WGS84 et conversion vers une projection commune (même CRS pour les 2 fichiers)
def est_wgs84(df):
    if mode_crs.startswith("WGS84"):
        return True
    if mode_crs.startswith("Projeté"):
        return False
    return semble_wgs84(df)


epsg_cible = None
conversions = {c: est_wgs84(b[0]) for c, b in bruts.items()}
if any(conversions.values()):
    if inverser_latlon:
        for c, b in bruts.items():
            if conversions[c]:
                b[0].rename(columns={"Est": "Nord", "Nord": "Est"}, inplace=True)
    if "EPSG:" in choix_proj:
        epsg_cible = int(choix_proj.split("EPSG:")[1].rstrip(")"))
    elif epsg_saisi.strip().isdigit():
        epsg_cible = int(epsg_saisi.strip())
    else:
        ref = bruts["tn"][0] if conversions.get("tn") else next(
            b[0] for c, b in bruts.items() if conversions[c])
        epsg_cible = epsg_utm(ref["Est"].mean(), ref["Nord"].mean())

# Phase 3 : affichage
for cle, col, titre in cols:
    if cle not in bruts:
        continue
    df, infos = bruts[cle]
    nom = fichiers[cle].name
    with col:
        try:
            if conversions[cle]:
                df = wgs84_vers_metres(df, epsg_cible)
                st.success(f"✅ {nom} — WGS84 converti en EPSG:{epsg_cible}")
            else:
                st.success(f"✅ {nom}")
            donnees[cle] = (nom, df, infos)
            resume(df, infos)
            st.dataframe(
                df[REQUIRED],
                use_container_width=True,
                height=250,
                column_config={
                    "Est": st.column_config.NumberColumn("Est", format="%.3f"),
                    "Nord": st.column_config.NumberColumn("Nord", format="%.3f"),
                    "Altitude": st.column_config.NumberColumn("Altitude", format="%.2f"),
                },
            )
        except ImportError:
            st.error("❌ La conversion WGS84 nécessite la bibliothèque `pyproj` (pip install pyproj).")
        except Exception as e:
            st.error(f"❌ {nom} : {e}")

pret = "tn" in donnees and "proj" in donnees

if not pret:
    with st.expander("Format attendu"):
        st.markdown("**Coordonnées projetées (mètres)**")
        st.code("Est;Nord;Altitude\n651234,50;6861234,25;102,35\n651235,50;6861234,25;102,41", language="text")
        st.markdown("**Coordonnées WGS84 (degrés)**")
        st.code("lon;lat;alt\n2,352200;48,856600;35,20\n2,352210;48,856600;35,26", language="text")
        st.code("lat;lon;alt\n48,856600;2,352200;35,20\n48,856600;2,352210;35,26", language="text")
        st.caption("Séparateur `,` `;` tabulation ou espace, virgule ou point décimal : détectés automatiquement. "
                   "Les colonnes sont reconnues par leur nom (Est/X/lon, Nord/Y/lat, Altitude/Z/alt) : "
                   "l'ordre lat, lon fonctionne donc avec un en-tête. "
                   "Les fichiers sans en-tête sont acceptés (3 dernières colonnes = Est/lon, Nord/lat, Altitude) ; "
                   "s'ils sont au format lat, lon, altitude, cochez « Colonnes en ordre latitude, longitude » dans la barre latérale.")

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
            try:
                res_carte = resolution_carte(df_tn, df_proj, methode, resolution)
                ecarts = (*calculer_ecarts(df_tn, df_proj, res_carte), res_carte)
            except Exception as e_carte:
                ecarts = None
                st.warning(f"Carte indisponible : {e_carte}")
        st.session_state["resultat"] = {
            "deblais": float(deb), "remblais": float(rem), "ecarts": ecarts,
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

    ecarts = res.get("ecarts")
    if ecarts is not None:
        xi, yi, diff, res_carte = ecarts
        valid = ~np.isnan(diff)
        cell = res_carte ** 2

        st.subheader("🗺️ Carte des déblais / remblais")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Surface en déblai", f"{np.count_nonzero(valid & (diff > seuil)) * cell:,.0f} m²".replace(",", " "))
        m2.metric("Surface en remblai", f"{np.count_nonzero(valid & (diff < -seuil)) * cell:,.0f} m²".replace(",", " "))
        m3.metric("Déblai max", f"{max(0.0, float(np.nanmax(diff))):.2f} m" if valid.any() else "—")
        m4.metric("Remblai max", f"{max(0.0, -float(np.nanmin(diff))):.2f} m" if valid.any() else "—")

        st.plotly_chart(figure_carte(xi, yi, diff, seuil), use_container_width=True)
        legende_carte(seuil)
        st.caption(
            f"Est : {xi.min():.1f} → {xi.max():.1f} m · Nord : {yi.min():.1f} → {yi.max():.1f} m · "
            f"cellules de {res_carte:.2f} m · molette pour zoomer, glisser pour se déplacer, survol pour lire l'écart · plus la couleur est vive, plus l'épaisseur est grande."
        )

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
