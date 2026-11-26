import csv
import io
import re
from urllib.parse import quote

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
# Zones Conique Conforme (RGF93 / CC42 à CC50 = EPSG:3942 à 3950)
ZONES_CC = {f"CC{z} (EPSG:{3900 + z})": 3900 + z for z in range(42, 51)}
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
def _est_nombre(cellule: str) -> bool:
    try:
        float(cellule.strip().strip('"\'').replace(",", "."))
        return True
    except ValueError:
        return False


def _nom_colonne(nom) -> str:
    """Normalise un nom de colonne : minuscules, sans guillemets, sans unité « (m) » / « [m] »."""
    nom = re.sub(r"[\(\[].*?[\)\]]", "", str(nom))
    return nom.strip().strip('"\'').strip().lower()


def lire_points(file) -> tuple[pd.DataFrame, dict]:
    """Lit un fichier de points CSV/TXT (séparateur, décimale et en-tête détectés automatiquement) ou KML.

    L'en-tête est facultatif. S'il est présent, les colonnes sont reconnues par leur nom
    (Est/X/lon, Nord/Y/lat, Altitude/Z/alt...) ; si les noms ne sont pas reconnus, les 3
    dernières colonnes sont utilisées (Est, Nord, Altitude).

    Retourne le DataFrame nettoyé (Est, Nord, Altitude) et un dict d'infos
    sur le nettoyage (lignes invalides, doublons...).
    """
    if file.name.lower().endswith(".kml"):
        return lire_kml(file)

    raw = file.getvalue().decode("utf-8-sig", errors="replace")
    lignes = [l for l in raw.splitlines() if l.strip()]
    if lignes and lignes[0].lstrip().startswith("#"):
        lignes[0] = lignes[0].lstrip().lstrip("#").strip()  # en-tête commenté : « #Est;Nord;Altitude »
    lignes = [lignes[0]] + [l for l in lignes[1:] if not l.lstrip().startswith("#")] if lignes else lignes
    if len(lignes) < 2:
        raise ValueError("Le fichier est vide ou ne contient qu'une ligne.")
    raw = "\n".join(lignes)

    echantillon = "\n".join(lignes[:20])
    sep = None
    for cand in (";", "\t"):  # séparateur présent le même nombre de fois sur chaque ligne
        nb = {l.count(cand) for l in lignes[:20]}
        if len(nb) == 1 and min(nb) >= 2:
            sep = cand
            break
    if sep is None:
        try:
            sep = csv.Sniffer().sniff(echantillon, delimiters=";,\t ").delimiter
        except csv.Error:
            sep = ";" if echantillon.count(";") > echantillon.count(",") else ","

    # Virgule décimale (ex. "12,5;45,2;101,3")
    decimal = "," if sep != "," and re.search(r"\d,\d", echantillon) else "."

    sep_regex = r"\s+" if sep == " " else sep

    # En-tête présent ? Oui si l'une des 3 dernières cellules de la 1re ligne n'est pas un nombre
    cellules = re.split(sep_regex, lignes[0].strip()) if sep == " " else lignes[0].split(sep)
    avec_entete = any(not _est_nombre(c) for c in cellules[-3:])

    infos_entete = None
    if not avec_entete:
        df = pd.read_csv(io.StringIO(raw), sep=sep_regex, decimal=decimal,
                         header=None, engine="python")
        if df.shape[1] < 3:
            raise ValueError("Au moins 3 colonnes sont nécessaires (Est, Nord, Altitude).")
        df = df.iloc[:, -3:]
        df.columns = REQUIRED
    else:
        df = pd.read_csv(io.StringIO(raw), sep=sep_regex, decimal=decimal, engine="python")
        noms = list(df.columns)
        df.columns = [_nom_colonne(c) for c in df.columns]
        df = df.rename(columns=MAPPING)
        if any(c not in df.columns for c in REQUIRED) or df.columns.duplicated().any():
            # Noms non reconnus : on retombe sur les 3 dernières colonnes
            if len(noms) < 3:
                raise ValueError(
                    f"Au moins 3 colonnes sont nécessaires (Est, Nord, Altitude). Colonnes trouvées : {noms}"
                )
            df = pd.read_csv(io.StringIO(raw), sep=sep_regex, decimal=decimal, engine="python")
            df = df.iloc[:, -3:]
            infos_entete = f"En-tête non reconnu ({', '.join(map(str, noms[-3:]))}) : colonnes lues dans l'ordre Est, Nord, Altitude."
            df.columns = REQUIRED

    return _finaliser(df, {"separateur": repr(sep), "decimale": decimal, "entete": avec_entete,
                           "avertissement": infos_entete, "kml": False, "alt_nulle": False,
                           "entete_reconnu": avec_entete and infos_entete is None})


def _finaliser(df: pd.DataFrame, extra: dict) -> tuple[pd.DataFrame, dict]:
    """Nettoyage commun (valeurs invalides, doublons XY) + dict d'infos."""
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

    infos = {"lignes": n_brut, "invalides": n_invalides, "doublons": n_doublons, **extra}
    return df, infos


def lire_kml(file) -> tuple[pd.DataFrame, dict]:
    """Lit un fichier KML : tous les points, sommets de lignes et de polygones.

    Le KML est toujours en WGS84 (longitude, latitude, altitude). Sans altitude, elle vaut 0.
    """
    import xml.etree.ElementTree as ET

    try:
        racine = ET.fromstring(file.getvalue())
    except ET.ParseError as e:
        raise ValueError(f"Fichier KML invalide : {e}")

    pts = []
    for el in racine.iter():
        balise = el.tag.split("}")[-1]
        if not el.text:
            continue
        if balise == "coordinates":        # "lon,lat[,alt] lon,lat[,alt] ..."
            triplets = [t.split(",") for t in el.text.split()]
        elif balise == "coord":            # gx:coord : "lon lat [alt]"
            triplets = [el.text.split()]
        else:
            continue
        for v in triplets:
            if len(v) >= 2:
                try:
                    pts.append((float(v[0]), float(v[1]), float(v[2]) if len(v) >= 3 else 0.0))
                except ValueError:
                    pass

    if not pts:
        raise ValueError("Aucune coordonnée trouvée dans le fichier KML.")
    df = pd.DataFrame(pts, columns=REQUIRED)
    return _finaliser(df, {"separateur": "—", "decimale": ".", "entete": False,
                           "avertissement": None, "kml": True,
                           "alt_nulle": bool((df["Altitude"] == 0).all())})


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


def transformer_points(df: pd.DataFrame, epsg_source: int, epsg_cible: int) -> pd.DataFrame:
    """Transforme les colonnes Est/Nord d'un système vers un autre (l'altitude n'est pas modifiée)."""
    from pyproj import Transformer

    tr = Transformer.from_crs(epsg_source, epsg_cible, always_xy=True)
    x, y = tr.transform(df["Est"].to_numpy(), df["Nord"].to_numpy())
    out = df.copy()
    out["Est"], out["Nord"] = x, y
    return out


def wgs84_vers_metres(df: pd.DataFrame, epsg_cible: int) -> pd.DataFrame:
    """Convertit Est=longitude / Nord=latitude (EPSG:4326) vers un CRS projeté en mètres."""
    return transformer_points(df, 4326, epsg_cible)


def centre_wgs84(df: pd.DataFrame, epsg_source):
    """Centre (lat, lon) WGS84 de l'emprise des points ; epsg_source = 4326 si déjà en lon/lat.
    Retourne None si le système source est inconnu ou si le résultat est incohérent."""
    if epsg_source is None or df is None or len(df) == 0:
        return None
    try:
        from pyproj import Transformer

        est = (float(df["Est"].min()) + float(df["Est"].max())) / 2
        nord = (float(df["Nord"].min()) + float(df["Nord"].max())) / 2
        lon, lat = Transformer.from_crs(epsg_source, 4326, always_xy=True).transform(est, nord)
    except Exception:
        return None
    if not (np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return float(lat), float(lon)


def liens_carte(lat: float, lon: float, libelle: str) -> str:
    """Liens Markdown (Google Maps, Plans d'Apple, Google Earth) vers un point WGS84."""
    ll = f"{lat:.6f},{lon:.6f}"
    gmaps = f"https://www.google.com/maps/search/?api=1&query={ll}"
    apple = f"https://maps.apple.com/?ll={ll}&q={quote(libelle)}&z=17"
    earth = f"https://earth.google.com/web/search/{ll}"
    return f"[Google Maps]({gmaps}) · [Plans]({apple}) · [Google Earth]({earth})"


def en_france(centre) -> bool:
    """Position plausible en France métropolitaine (zone large, Corse comprise)."""
    return bool(centre) and 41 <= centre[0] <= 52 and -6 <= centre[1] <= 10


def candidats_position(df: pd.DataFrame, wgs84_probable: bool, epsg_choisi) -> list:
    """Interprétations plausibles de la position d'un fichier, indépendamment du choix des coordonnées.

    Retourne une liste de (libellé du système supposé, (lat, lon)). Pour des coordonnées en mètres,
    on essaie Lambert-93 et les zones CC42 à CC50 et on ne garde que les positions en France ;
    la projection actuellement choisie est toujours proposée en premier si elle donne un point valide."""
    cands = []
    if wgs84_probable:
        c = centre_wgs84(df, 4326)
        if c:
            cands.append(("WGS84 lon, lat", c))
        inv = centre_wgs84(df.rename(columns={"Est": "Nord", "Nord": "Est"}), 4326)
        if en_france(inv) and not en_france(c):
            cands.insert(0, ("WGS84 lat, lon (colonnes inversées)", inv))
        return cands
    essais = ([("Lambert-93", 2154)] + [(n.split(" ")[0], e) for n, e in ZONES_CC.items()]
              + [("UTM 30N", 32630), ("UTM 31N", 32631), ("UTM 32N", 32632)])
    if epsg_choisi and epsg_choisi not in [e for _, e in essais]:
        essais.append((f"EPSG:{epsg_choisi}", epsg_choisi))
    for lib, e in essais:
        c = centre_wgs84(df, e)
        if c and (en_france(c) or e == epsg_choisi):
            cands.append((lib + (" · choix actuel" if e == epsg_choisi else ""), c))
    cands.sort(key=lambda t: "choix actuel" not in t[0])
    return cands


def afficher_liens(libelle: str, centre, detail: str = "") -> None:
    if centre is None:
        st.markdown(f"**{libelle}** : position non calculable — vérifiez le système de coordonnées.")
    else:
        lat, lon = centre
        st.markdown(f"**{libelle}**" + (f" — {detail}" if detail else ""))
        st.code(f"{lat:.6f}, {lon:.6f}", language=None)  # copiable (bouton de copie), collable dans Google Maps
        st.markdown(liens_carte(lat, lon, libelle))


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
    if infos.get("alt_nulle"):
        st.caption("⚠️ Aucune altitude (ou toutes à 0) dans le KML : utilisez l'option « Altitudes » "
                   "sous le fichier (valeur constante ou saisie manuelle).")
    if infos.get("avertissement"):
        st.caption(f"ℹ️ {infos['avertissement']}")
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

st.sidebar.subheader("Système de coordonnées")
bloc_verif = st.sidebar.container()  # rempli plus bas, une fois les fichiers lus
mode_crs = st.sidebar.selectbox(
    "Coordonnées de vos fichiers",
    ("Détection automatique", "Projeté en mètres (Lambert-93, CC47, UTM…)",
     "WGS84 lon/lat (EPSG:4326)"),
    index=2,  # défaut : WGS84 lon/lat (EPSG:4326)
    help="Dans quel système sont exprimées les coordonnées de vos fichiers TN et projet. "
         "Le calcul de volumes exige des mètres : les coordonnées WGS84 (degrés) sont "
         "converties automatiquement vers la projection choisie ci-dessous. "
         "Des fichiers déjà en Lambert-93, CC47 ou UTM sont utilisés tels quels.",
)
inverser_latlon = st.sidebar.checkbox(
    "Colonnes en ordre latitude, longitude",
    disabled=mode_crs.startswith("Projeté"),
    help="À cocher si vos fichiers WGS84 sans en-tête (ou à en-tête non reconnu) sont au format "
         "lat, lon, altitude. Sans effet sur les KML et sur les fichiers dont l'en-tête est reconnu.",
)
choix_proj = st.sidebar.selectbox(
    "Projection métrique (calcul / vos fichiers)",
    ("UTM automatique",
     "Lambert-93 (EPSG:2154)",
     "CC47 – RGF93 / Conique conforme zone 7 (EPSG:3947)",
     "UTM 30N (EPSG:32630)", "UTM 31N (EPSG:32631)",
     "UTM 32N (EPSG:32632)", "Autre code EPSG…"),
    index=3,  # défaut : UTM 30N (EPSG:32630)
    help="Fichiers WGS84 : projection métrique vers laquelle ils sont convertis pour le calcul. "
         "Fichiers déjà en mètres : projection d'origine de vos fichiers (nécessaire pour les liens "
         "de vérification sur carte). Lambert-93 : France entière. CC47 : bandes CC (zone 47°N, "
         "ex. Vendée). UTM automatique : fuseau calculé depuis le centre du fichier TN (fichiers WGS84 "
         "uniquement).",
)
epsg_saisi = ""
if choix_proj.startswith("Autre"):
    epsg_saisi = st.sidebar.text_input(
        "Code EPSG", "",
        help="Exemples : 2154 (Lambert-93), 3947 (CC47), 3812 (Lambert 2008, Belgique), 32633 (UTM 33N).",
    )

bloc_transfo = st.sidebar.container()  # rempli plus bas (transformation Lambert-93 ⇄ CC)

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

with bloc_transfo.expander("🔄 Transformation Lambert-93 ⇄ Conique conforme (CC42 à CC50)"):
    sens = st.radio("Sens de la transformation", ("Lambert-93 → CC", "CC → Lambert-93"),
                    horizontal=True, key="tr_sens")
    zone_cc = st.selectbox("Zone Conique conforme", list(ZONES_CC), index=5, key="tr_zone",
                           help="CC47 (EPSG:3947) couvre les latitudes 46° à 48° N, dont la Vendée.")
    epsg_cc = ZONES_CC[zone_cc]
    nom_cc = zone_cc.split(" ")[0]
    if sens.startswith("Lambert"):
        ep_src, ep_dst, nom_src, nom_dst = 2154, epsg_cc, "Lambert-93", nom_cc
    else:
        ep_src, ep_dst, nom_src, nom_dst = epsg_cc, 2154, nom_cc, "Lambert-93"
    st.caption("Même datum (RGF93) : simple changement de projection, sans perte de précision. "
               "L'altitude n'est pas modifiée.")
    t_pt, t_fic = st.tabs(["Un point", "Fichier de points"])
    with t_pt:
        ce, cn = st.columns(2)
        e_in = ce.number_input(f"Est {nom_src} (m)", value=0.0, step=1.0, format="%.3f", key="tr_e")
        n_in = cn.number_input(f"Nord {nom_src} (m)", value=0.0, step=1.0, format="%.3f", key="tr_n")
        if e_in or n_in:
            try:
                r = transformer_points(pd.DataFrame({"Est": [e_in], "Nord": [n_in]}), ep_src, ep_dst)
                x_out, y_out = float(r["Est"].iloc[0]), float(r["Nord"].iloc[0])
                if np.isfinite(x_out) and np.isfinite(y_out):
                    ro, rn = st.columns(2)
                    ro.markdown(f"**Est {nom_dst} (m)**")
                    ro.code(f"{x_out:.3f}", language=None)
                    rn.markdown(f"**Nord {nom_dst} (m)**")
                    rn.code(f"{y_out:.3f}", language=None)
                else:
                    st.warning("Coordonnées hors du domaine de validité de la projection.")
            except ImportError:
                st.error("❌ La transformation nécessite la bibliothèque `pyproj` (pip install pyproj).")
            except Exception as e:
                st.error(f"❌ {e}")
    with t_fic:
        f_tr = st.file_uploader("Fichier Est;Nord;Altitude (CSV ou TXT)", type=["csv", "txt"], key="tr_fic")
        if f_tr is not None:
            try:
                d_in, i_in = lire_points(f_tr)
                d_out = transformer_points(d_in, ep_src, ep_dst)
                st.success(f"✅ {len(d_out)} point(s) transformé(s) : {nom_src} → {nom_dst}")
                if i_in["invalides"] or i_in["doublons"]:
                    st.caption(f"⚠️ {i_in['invalides']} ligne(s) invalide(s) ignorée(s), "
                               f"{i_in['doublons']} doublon(s) XY fusionné(s).")
                st.dataframe(
                    d_out[REQUIRED], use_container_width=True, height=250,
                    column_config={
                        "Est": st.column_config.NumberColumn("Est", format="%.3f"),
                        "Nord": st.column_config.NumberColumn("Nord", format="%.3f"),
                        "Altitude": st.column_config.NumberColumn("Altitude", format="%.2f"),
                    },
                )
                nom_sortie = f_tr.name.rsplit(".", 1)[0] + f"_{nom_dst}.csv"
                st.download_button(
                    f"📥 Télécharger en {nom_dst} (CSV)",
                    d_out[REQUIRED].to_csv(index=False, sep=";", float_format="%.3f").encode("utf-8-sig"),
                    file_name=nom_sortie, mime="text/csv",
                )
            except ImportError:
                st.error("❌ La transformation nécessite la bibliothèque `pyproj` (pip install pyproj).")
            except Exception as e:
                st.error(f"❌ {f_tr.name} : {e}")

col1, col2 = st.columns(2)
donnees = {}
cols = (("tn", col1, "1. Terrain naturel (TN)"), ("proj", col2, "2. Terrain projeté"))

# Phase 1 : upload, choix de l'origine des altitudes et lecture brute
MAX_EDITION = 5000  # au-delà, la saisie manuelle point par point n'est plus proposée
bruts = {}
fichiers = {}
for cle, col, titre in cols:
    with col:
        st.subheader(titre)
        fichiers[cle] = st.file_uploader(f"Fichier {titre}", type=["csv", "txt", "kml"], key=cle,
                                         label_visibility="collapsed")
        mode_alt = st.radio(
            "Altitudes", ("Du fichier", "Valeur constante", "Saisie manuelle"),
            key=f"altmode_{cle}", horizontal=True,
            help="Du fichier : altitudes lues telles quelles. Valeur constante : une même altitude pour tous "
                 "les points (plateforme plane, KML sans altitude). Saisie manuelle : modifier les altitudes "
                 "point par point dans un tableau.",
        )
        valeur_alt = 0.0
        if mode_alt == "Valeur constante":
            valeur_alt = st.number_input("Altitude appliquée à tous les points (m)", value=0.0,
                                         step=0.1, format="%.2f", key=f"altval_{cle}")
        if fichiers[cle] is not None:
            try:
                df0, infos0 = lire_points(fichiers[cle])
                if mode_alt == "Valeur constante":
                    df0 = df0.assign(Altitude=float(valeur_alt))
                    infos0 = {**infos0, "alt_nulle": False}
                elif mode_alt == "Saisie manuelle":
                    if len(df0) > MAX_EDITION:
                        st.warning(f"Saisie manuelle limitée à {MAX_EDITION} points "
                                   f"(fichier : {len(df0)}). Altitudes du fichier conservées.")
                    else:
                        with st.expander("Saisir / modifier les altitudes", expanded=True):
                            edite = st.data_editor(
                                df0[REQUIRED], use_container_width=True, height=250,
                                disabled=["Est", "Nord"], num_rows="fixed",
                                key=f"ed_{cle}_{fichiers[cle].name}_{fichiers[cle].size}",
                                column_config={
                                    "Est": st.column_config.NumberColumn("Est", format="%.6f"),
                                    "Nord": st.column_config.NumberColumn("Nord", format="%.6f"),
                                    "Altitude": st.column_config.NumberColumn("Altitude (m)", format="%.2f"),
                                },
                            )
                        df0 = edite.dropna(subset=["Altitude"])
                        if len(df0) < 3:
                            raise ValueError("Moins de 3 points avec une altitude : impossible de calculer un MNT.")
                        infos0 = {**infos0, "alt_nulle": False}
                bruts[cle] = (df0, infos0)
            except Exception as e:
                st.error(f"❌ {fichiers[cle].name} : {e}")

# Phase 2 : détection WGS84 et conversion vers une projection commune (même CRS pour les 2 fichiers)
def est_wgs84(df, infos):
    if infos.get("kml"):
        return True  # un KML est toujours en WGS84
    if mode_crs.startswith("WGS84"):
        return True
    if mode_crs.startswith("Projeté"):
        return False
    return semble_wgs84(df)


epsg_cible = None
conversions = {c: est_wgs84(b[0], b[1]) for c, b in bruts.items()}
if any(conversions.values()):
    if inverser_latlon:
        for c, b in bruts.items():
            # Un KML est toujours lon, lat ; un en-tête reconnu fixe déjà l'ordre des colonnes
            if conversions[c] and not b[1].get("kml") and not b[1].get("entete_reconnu"):
                b[0].rename(columns={"Est": "Nord", "Nord": "Est"}, inplace=True)
    if "EPSG:" in choix_proj:
        epsg_cible = int(choix_proj.split("EPSG:")[1].rstrip(")"))
    elif epsg_saisi.strip().isdigit():
        epsg_cible = int(epsg_saisi.strip())
    else:
        ref = bruts["tn"][0] if conversions.get("tn") else next(
            b[0] for c, b in bruts.items() if conversions[c])
        epsg_cible = epsg_utm(ref["Est"].mean(), ref["Nord"].mean())

if epsg_cible is None:  # fichiers déjà en mètres : la projection choisie est leur système d'origine
    if "EPSG:" in choix_proj:
        epsg_cible = int(choix_proj.split("EPSG:")[1].rstrip(")"))
    elif epsg_saisi.strip().isdigit():
        epsg_cible = int(epsg_saisi.strip())

# Vérification de l'emplacement en amont : indépendante du choix ci-dessous (on teste les systèmes
# plausibles), pour voir sur carte où tombent les fichiers AVANT de choisir le système de coordonnées
with bloc_verif:
    if not bruts:
        st.markdown("[Google Maps](https://www.google.com/maps), "
                    "[Plans](https://maps.apple.com/) et "
                    "[Google Earth](https://earth.google.com/web/)")
    for cle, _, _titre in cols:
        if cle not in bruts:
            continue
        df_b, infos_b = bruts[cle]
        nom_c = "TN" if cle == "tn" else "Projet"
        wgs_probable = bool(infos_b.get("kml")) or semble_wgs84(df_b)
        cands = candidats_position(df_b, wgs_probable, None if wgs_probable else epsg_cible)
        if not cands:
            st.markdown(f"**{nom_c}** : aucune position plausible trouvée — vérifiez le format du fichier.")
        for lib, centre in cands:
            afficher_liens(nom_c, centre, lib)
    if bruts:
        st.write("Choisissez ensuite le système de coordonnées correspondant au bon emplacement.")

# Phase 3 : affichage
for cle, col, titre in cols:
    if cle not in bruts:
        continue
    df, infos = bruts[cle]
    nom = fichiers[cle].name
    with col:
        try:
            origine = "détecté" if mode_crs.startswith("Détection") else "choisi"
            if conversions[cle]:
                df = wgs84_vers_metres(df, epsg_cible)
                st.success(f"✅ {nom}\n\n"
                           f"Système {origine} : WGS84 lon/lat (EPSG:4326) → "
                           f"converti en mètres : EPSG:{epsg_cible}")
            else:
                src = f"EPSG:{epsg_cible}" if epsg_cible else "projection non précisée"
                st.success(f"✅ {nom}\n\n"
                           f"Système {origine} : mètres ({src}) → utilisé tel quel, sans conversion")
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
        st.caption("Les fichiers KML (points, lignes ou polygones) sont aussi acceptés : toujours en WGS84 "
                   "lon, lat, altitude ; sans altitude, elle vaut 0 → utilisez l'option « Altitudes » "
                   "(valeur constante ou saisie manuelle).")
        st.caption("Séparateur `,` `;` tabulation ou espace, virgule ou point décimal : détectés automatiquement. "
                   "Les colonnes sont reconnues par leur nom (Est/X/lon, Nord/Y/lat, Altitude/Z/alt) : "
                   "l'ordre lat, lon fonctionne donc avec un en-tête. "
                   "L'en-tête est facultatif (une ligne de titres, éventuellement précédée de #) : s'il n'est pas reconnu, "
                   "ou s'il n'y en a pas, les 3 dernières colonnes sont lues comme Est/lon, Nord/lat, Altitude ; "
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
            "centres": {
                "TN": centre_wgs84(donnees["tn"][1], epsg_cible),
                "Projet": centre_wgs84(donnees["proj"][1], epsg_cible),
            },
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

    if res.get("centres"):
        st.markdown("##### 📍 Vérifier l'emplacement")
        lc = st.columns(len(res["centres"]))
        for colonne, (nom_c, centre) in zip(lc, res["centres"].items()):
            with colonne:
                afficher_liens(nom_c, centre)

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
