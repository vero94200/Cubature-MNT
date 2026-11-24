# Calcul de Cubature MNT

Ce projet permet de calculer les volumes de déblais et de remblais entre deux Modèles Numériques de Terrain (MNT) définis par des nuages de points topographiques (Est, Nord, Altitude).

## Prérequis

Python 3.8+ est requis. Installez les dépendances :

```bash
pip install -r requirements.txt
```

## Format des données

Les fichiers d'entrée doivent être au format CSV avec exactement ces trois colonnes (l'ordre n'importe pas, mais les noms doivent correspondre) :

```csv
Est,Nord,Altitude
350000.50,6400000.20,150.50
350005.00,6400002.10,151.20
```

## Utilisation

Lancez le script `main.py` en spécifiant les chemins vers vos fichiers CSV :

### Méthode TIN (Recommandée pour la précision)
La méthode TIN (Triangulated Irregular Network) crée un maillage triangulé respectant exactement vos points d'entrée.

```bash
python main.py --tn data/terrain_naturel.csv --projet data/terrain_projet.csv --methode tin
```

### Méthode Grille (Recommandée pour la rapidité)
La méthode Grille interpole les données sur une grille régulière. Idéale pour les très gros jeux de données.

```bash
python main.py --tn data/terrain_naturel.csv --projet data/terrain_projet.csv --methode grille --resolution 2.0
```
*(Ici, la résolution est de 2m x 2m).*

## Résultats

Le script affiche les volumes de déblais, de remblais et la balance globale en unités cubiques (généralement des m³ si vos coordonnées sont en mètres).
