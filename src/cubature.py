import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, griddata

def _validate_dataframes(df_tn, df_proj):
    """Vérifie que les DataFrames ont les bonnes colonnes."""
    required_cols = ['Est', 'Nord', 'Altitude']
    for df, name in zip([df_tn, df_proj], ['TN', 'Projet']):
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Le DataFrame {name} doit contenir les colonnes : {required_cols}")

def calculate_cubature_tin(df_tn, df_proj):
    """
    Calcule la cubature par la méthode TIN (Triangulated Irregular Network).
    Plus précis, respecte exactement les points topographiques.
    
    Retourne: (déblais, remblais) en unités cubiques (ex: m3).
    """
    _validate_dataframes(df_tn, df_proj)
    
    points_tn = df_tn[['Est', 'Nord']].values
    z_tn = df_tn['Altitude'].values
    
    # 1. Triangulation de Delaunay basée sur le MNT TN
    tri = Delaunay(points_tn)
    
    # 2. Création de l'interpolateur pour le MNT Projet
    points_proj = df_proj[['Est', 'Nord']].values
    z_proj = df_proj['Altitude'].values
    interp_proj = LinearNDInterpolator(points_proj, z_proj)
    
    total_deblais = 0.0
    total_remblais = 0.0
    
    # 3. Calcul du volume pour chaque triangle
    for simplex in tri.simplices:
        p1, p2, p3 = points_tn[simplex]
        z_tn1, z_tn2, z_tn3 = z_tn[simplex]
        
        # Interpolation des altitudes du MNT Projet sur les sommets du triangle TN
        z_proj_interp = interp_proj([p1, p2, p3]).flatten()
        
        # Ignorer les triangles hors de l'emprise du MNT Projet
        if np.any(np.isnan(z_proj_interp)):
            continue
            
        z_proj1, z_proj2, z_proj3 = z_proj_interp
        
        # Différences d'altitude (h = Z_TN - Z_Projet)
        # h > 0 : le TN est au-dessus -> Déblai
        # h < 0 : le TN est en-dessous -> Remblai
        h1 = z_tn1 - z_proj1
        h2 = z_tn2 - z_proj2
        h3 = z_tn3 - z_proj3
        
        # Surface du triangle (produit vectoriel en 2D)
        area = 0.5 * np.abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        
        # Volume du prisme triangulaire
        volume = area * (h1 + h2 + h3) / 3.0
        
        if volume > 0:
            total_deblais += volume
        else:
            total_remblais += abs(volume)
            
    return total_deblais, total_remblais

def calculate_cubature_grid(df_tn, df_proj, resolution=1.0):
    """
    Calcule la cubature par la méthode Grille (Raster).
    Plus rapide, interpole sur une grille régulière.
    
    resolution: pas de la grille en unités du terrain (ex: 1.0 pour 1m).
    Retourne: (déblais, remblais) en unités cubiques.
    """
    _validate_dataframes(df_tn, df_proj)
    
    # Définition de l'emprise commune
    x_min = min(df_tn['Est'].min(), df_proj['Est'].min())
    x_max = max(df_tn['Est'].max(), df_proj['Est'].max())
    y_min = min(df_tn['Nord'].min(), df_proj['Nord'].min())
    y_max = max(df_tn['Nord'].max(), df_proj['Nord'].max())
    
    xi = np.arange(x_min, x_max, resolution)
    yi = np.arange(y_min, y_max, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolation des deux MNT sur la grille
    zi_tn = griddata((df_tn['Est'], df_tn['Nord']), df_tn['Altitude'], (xi_grid, yi_grid), method='linear')
    zi_proj = griddata((df_proj['Est'], df_proj['Nord']), df_proj['Altitude'], (xi_grid, yi_grid), method='linear')
    
    # Différence et masquage des zones sans données (NaN)
    diff = zi_tn - zi_proj
    valid_mask = ~np.isnan(diff)
    diff_valid = diff[valid_mask]
    
    # Calcul des volumes
    cell_area = resolution * resolution
    deblais = np.sum(diff_valid[diff_valid > 0]) * cell_area
    remblais = np.sum(np.abs(diff_valid[diff_valid < 0])) * cell_area
    
    return deblais, remblais
