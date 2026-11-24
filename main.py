import pandas as pd
import argparse
from src.cubature import calculate_cubature_tin, calculate_cubature_grid

def main():
    parser = argparse.ArgumentParser(description="Calcul de cubature entre deux MNT (TN et Projet).")
    parser.add_argument("--tn", required=True, help="Chemin vers le fichier CSV du Terrain Naturel (Est, Nord, Altitude)")
    parser.add_argument("--projet", required=True, help="Chemin vers le fichier CSV du Terrain Projeté (Est, Nord, Altitude)")
    parser.add_argument("--methode", choices=["tin", "grille"], default="tin", help="Méthode de calcul (défaut: tin)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Résolution de la grille si méthode 'grille' (défaut: 1.0)")
    
    args = parser.parse_args()
    
    print(f"Chargement des données...")
    df_tn = pd.read_csv(args.tn)
    df_projet = pd.read_csv(args.projet)
    
    print(f" -> TN : {len(df_tn)} points")
    print(f" -> Projet : {len(df_projet)} points")
    
    print(f"\nCalcul en cours avec la méthode : {args.methode.upper()}...")
    
    if args.methode == "tin":
        deblais, remblais = calculate_cubature_tin(df_tn, df_projet)
    else:
        deblais, remblais = calculate_cubature_grid(df_tn, df_projet, resolution=args.resolution)
        
    print("\n" + "="*40)
    print("RÉSULTATS DE LA CUBATURE")
    print("="*40)
    print(f"Volume de DÉBLAIS  : {deblais:,.2f} m3")
    print(f"Volume de REMBLAIS : {remblais:,.2f} m3")
    print(f"Balance (Déblais - Remblais) : {deblais - remblais:,.2f} m3")
    print("="*40)

if __name__ == "__main__":
    main()