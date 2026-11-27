# 🏗️ Cubature MNT

Calcul de cubatures (déblais / remblais) entre deux modèles numériques de terrain, par TIN ou grille — application web Streamlit, en français.

**🚀 Application en ligne** : [cubature-mnt.streamlit.app](https://cubature-mnt-ywqyaueuwtjfzc8jtto92z.streamlit.app/)

![Aperçu de l'application](captures/apercu.png)

## Import des données

- **CSV / TXT / KML** — séparateur, décimale et en-tête détectés automatiquement (en-tête facultatif, même commenté par `#`)
- Colonnes reconnues par leur nom (Est/X/lon…, Nord/Y/lat…, Altitude/Z/h…) ; sinon les 3 dernières colonnes sont lues Est, Nord, Altitude
- Altitudes : du fichier, valeur constante, ou saisie manuelle (≤ 5 000 points)

## Systèmes de coordonnées

- Détection automatique WGS84 lon/lat vs mètres, ou choix manuel
- Conversion WGS84 → UTM automatique, Lambert-93, CC47, UTM 30N/31N/32N ou tout code EPSG
- **Transformation France** : un point ou un fichier entre Lambert-93, CC42–CC50, anciens Lambert NTF (I à IV, II étendu), UTM et WGS84 — avec suggestion de la zone CC adaptée et alerte sur les changements de datum NTF ⇄ RGF93
- **Vérification avant calcul** : liens Google Maps / Plans / Google Earth sur la position supposée de chaque fichier

## Altitudes IGN

Si un fichier WGS84 est sans altitude, l'application les récupère auprès du service altimétrique de l'IGN, au choix :

- **LiDAR HD – sans interpolation** (défaut) : mesure la plus proche
- **LiDAR HD – avec interpolation** : valeur interpolée
- **RGE ALTI (MNT 1 m)** : MNT de référence

Requêtes par paquets de 100 points (limite service : 5/s) ; points hors couverture signalés.

## Calcul et résultats

- **TIN** (précis) ou **grille** (rapide, pas de 0,5 à 5 m)
- **Épaisseur de couche** : ajout (+) ou retranchement (−) aux altitudes du TN et/ou du projet (ex. décapage de terre végétale)
- Foisonnement, tassement, coût de transport
- Déblais, remblais, balance (« Volume à évacuer » / « Volume à apporter »), surface réellement calculée
- **Carte interactive** déblais/remblais avec seuil d'écart réglable
- **Rapport CSV** : paramètres, modes d'altitudes, emprises, altitudes avec/sans épaisseur, et les 4 tables de points complètes

## Formats

```text
Est;Nord;Altitude            → mètres (projeté)
651234,50;6861234,25;102,35

lon;lat;alt                  → WGS84 en degrés (ou lat;lon;alt à cocher)
2,352200;48,856600;35,20
```

KML : WGS84 lon, lat, altitude (0 si absente → récupération IGN possible).

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.9+ ; dépendances : streamlit, pandas, numpy, plotly, scipy, pyproj.

## Structure

```text
app.py            # application web Streamlit
src/cubature.py   # calculs (TIN, grille, différences)
main.py           # interface en ligne de commande
requirements.txt
```

Déploiement : [Streamlit Community Cloud](https://share.streamlit.io/) depuis ce dépôt (point d'entrée `app.py`).

## Licence

[MIT](LICENSE) — réutilisation libre, y compris commerciale, en conservant la mention de copyright.
