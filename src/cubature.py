import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata
from scipy.spatial import ConvexHull

# Garde-fou : nombre maximal de cellules pour la méthode grille / la carte
MAX_CELLS = 25_000_000


def _validate_dataframes(df_tn, df_proj):
    """Vérifie que les DataFrames ont les bonnes colonnes."""
    required_cols = ['Est', 'Nord', 'Altitude']
    for df, name in zip([df_tn, df_proj], ['TN', 'Projet']):
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Le DataFrame {name} doit contenir les colonnes : {required_cols}")


def _split_volumes(h, area):
    """Volumes de déblai (V+) et de remblai (V-) de prismes triangulaires.

    h    : (m, 3) écarts Z_TN - Z_Projet aux sommets de chaque triangle.
    area : (m,)   surface en plan de chaque triangle.

    L'écart variant linéairement dans le triangle, la ligne d'écart nul le
    coupe en deux zones. Chaque zone est une pyramide ou un tronc, ce qui
    donne une formule exacte (pas de compensation déblai/remblai interne).
    """
    net = area * h.sum(axis=1) / 3.0
    npos = (h > 0).sum(axis=1)
    hs = np.sort(h, axis=1)  # croissant

    vp = np.zeros(len(h))
    vm = np.zeros(len(h))

    # Triangle entièrement en déblai / en remblai
    m3 = npos == 3
    vp[m3] = net[m3]
    m0 = npos == 0
    vm[m0] = -net[m0]

    # Un seul sommet en déblai : petite pyramide de déblai
    m1 = npos == 1
    a, b, c = hs[m1, 2], hs[m1, 0], hs[m1, 1]
    vp[m1] = area[m1] * a**3 / (3.0 * (a - b) * (a - c))
    vm[m1] = vp[m1] - net[m1]

    # Un seul sommet en remblai : petite pyramide de remblai
    m2 = npos == 2
    c, a, b = hs[m2, 0], hs[m2, 1], hs[m2, 2]
    vm[m2] = area[m2] * (-c)**3 / (3.0 * (a - c) * (b - c))
    vp[m2] = net[m2] + vm[m2]

    return vp, vm


def _clip_convex(subject, clip):
    """Sutherland-Hodgman : découpe le polygone `subject` par le polygone convexe `clip` (sens antihoraire)."""
    out = [tuple(p) for p in subject]
    n = len(clip)
    for i in range(n):
        if not out:
            break
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % n]
        inp, out = out, []
        s = inp[-1]
        ss = (bx - ax) * (s[1] - ay) - (by - ay) * (s[0] - ax)
        for e in inp:
            se = (bx - ax) * (e[1] - ay) - (by - ay) * (e[0] - ax)
            if se >= 0:
                if ss < 0:
                    t = ss / (ss - se)
                    out.append((s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])))
                out.append(e)
            elif ss >= 0:
                t = ss / (ss - se)
                out.append((s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])))
            s, ss = e, se
    return out


def _hull(p):
    """Enveloppe convexe (sommets en sens antihoraire)."""
    try:
        return p[ConvexHull(p).vertices]
    except Exception as e:  # points colinéaires ou trop peu nombreux
        raise ValueError("Il faut au moins 3 points non alignés par fichier.") from e


def _triangle_areas(pts):
    return 0.5 * np.abs((pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1])
                        - (pts[:, 2, 0] - pts[:, 0, 0]) * (pts[:, 1, 1] - pts[:, 0, 1]))


def calculate_cubature_tin(df_tn, df_proj):
    """
    Calcule la cubature par la méthode TIN (Triangulated Irregular Network).

    La triangulation de Delaunay est construite sur l'ensemble des points
    (TN + projet), ce qui capte aussi les ruptures de pente du projet. Chaque
    surface est interpolée linéairement sur les sommets, puis le volume de
    chaque triangle est découpé exactement entre déblai et remblai.

    Seule l'emprise commune (intersection des deux enveloppes convexes) est
    comptée : les triangles qui la traversent sont découpés exactement, ceux
    qui sont entièrement dehors sont ignorés.

    Retourne: (déblais, remblais) en unités cubiques (ex: m3).
    """
    _validate_dataframes(df_tn, df_proj)

    p_tn = df_tn[['Est', 'Nord']].to_numpy(dtype=float)
    z_tn = df_tn['Altitude'].to_numpy(dtype=float)
    p_pr = df_proj[['Est', 'Nord']].to_numpy(dtype=float)
    z_pr = df_proj['Altitude'].to_numpy(dtype=float)

    # Recentrage des coordonnées (bien meilleur conditionnement numérique
    # avec des coordonnées type Lambert 93 ou UTM)
    origin = np.vstack([p_tn, p_pr]).min(axis=0)
    p_tn = p_tn - origin
    p_pr = p_pr - origin

    # Emprise commune = intersection des deux enveloppes convexes
    poly = np.array(_clip_convex(_hull(p_tn), _hull(p_pr)))
    if len(poly) < 3:
        return 0.0, 0.0
    x, y = poly[:, 0], poly[:, 1]
    if 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) <= 0:
        return 0.0, 0.0

    lin_tn, nn_tn = LinearNDInterpolator(p_tn, z_tn), NearestNDInterpolator(p_tn, z_tn)
    lin_pr, nn_pr = LinearNDInterpolator(p_pr, z_pr), NearestNDInterpolator(p_pr, z_pr)

    def z_at(lin, nn, pts):
        # Interpolation linéaire ; sur le bord exact de l'enveloppe, repli sur le plus proche voisin
        z = lin(pts)
        bad = np.isnan(z)
        if bad.any():
            z[bad] = nn(pts[bad])
        return z

    # Points uniques de la triangulation commune
    uniq, inv = np.unique(np.vstack([p_tn, p_pr]), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n = len(p_tn)

    z1 = z_at(lin_tn, nn_tn, uniq)
    z1[inv[:n]] = z_tn
    z2 = z_at(lin_pr, nn_pr, uniq)
    z2[inv[n:]] = z_pr

    simplices = Delaunay(uniq).simplices
    P = uniq[simplices]  # (m, 3, 2)

    # Triangles entièrement dans l'emprise commune / entièrement dehors / à découper
    span = max(float(np.ptp(uniq[:, 0])), float(np.ptp(uniq[:, 1])), 1.0)
    tol = 1e-9 * span * span
    inside = np.ones(len(simplices), dtype=bool)
    outside = np.zeros(len(simplices), dtype=bool)
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        d = (bx - ax) * (P[:, :, 1] - ay) - (by - ay) * (P[:, :, 0] - ax)
        inside &= (d >= -tol).all(axis=1)
        outside |= (d < -tol).all(axis=1)
    cross = ~inside & ~outside

    h_parts = [(z1 - z2)[simplices[inside]]]
    area_parts = [_triangle_areas(P[inside])]

    # Triangles qui traversent le bord : découpe exacte puis re-triangulation en éventail
    sub = []
    for k in np.flatnonzero(cross):
        q = _clip_convex(P[k], poly)
        for j in range(1, len(q) - 1):
            sub.append((q[0], q[j], q[j + 1]))
    if sub:
        sub = np.array(sub, dtype=float)  # (s, 3, 2)
        flat = sub.reshape(-1, 2)
        hz = (z_at(lin_tn, nn_tn, flat) - z_at(lin_pr, nn_pr, flat)).reshape(-1, 3)
        ar = _triangle_areas(sub)
        keep = ar > 1e-12 * span * span
        h_parts.append(hz[keep])
        area_parts.append(ar[keep])

    h = np.vstack(h_parts)
    area = np.concatenate(area_parts)
    if len(area) == 0:
        return 0.0, 0.0

    vp, vm = _split_volumes(h, area)
    return float(vp.sum()), float(vm.sum())


def compute_difference_grid(df_tn, df_proj, resolution=1.0):
    """
    Grille des écarts Z_TN - Z_Projet sur l'emprise commune.

    Retourne (xi, yi, diff) : coordonnées des centres de cellules (Est, Nord)
    et matrice (len(yi), len(xi)). diff > 0 : déblai ; diff < 0 : remblai ;
    NaN : hors données.
    """
    _validate_dataframes(df_tn, df_proj)
    if resolution <= 0:
        raise ValueError("La résolution doit être strictement positive.")

    # Emprise commune aux deux jeux de points
    x_min = max(df_tn['Est'].min(), df_proj['Est'].min())
    x_max = min(df_tn['Est'].max(), df_proj['Est'].max())
    y_min = max(df_tn['Nord'].min(), df_proj['Nord'].min())
    y_max = min(df_tn['Nord'].max(), df_proj['Nord'].max())
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Les emprises du TN et du projet ne se recouvrent pas.")

    nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
    ny = max(1, int(np.ceil((y_max - y_min) / resolution)))
    if nx * ny > MAX_CELLS:
        raise ValueError(
            f"Grille trop grande ({nx * ny:,} cellules) : augmentez la résolution."
        )

    # Centres de cellules (échantillonnage plus juste qu'aux coins)
    xi = x_min + (np.arange(nx) + 0.5) * resolution
    yi = y_min + (np.arange(ny) + 0.5) * resolution
    xg, yg = np.meshgrid(xi - x_min, yi - y_min)

    def _interp(df):
        pts = np.column_stack([df['Est'].to_numpy(float) - x_min,
                               df['Nord'].to_numpy(float) - y_min])
        return griddata(pts, df['Altitude'].to_numpy(float), (xg, yg), method='linear')

    diff = _interp(df_tn) - _interp(df_proj)
    return xi, yi, diff


def calculate_cubature_grid(df_tn, df_proj, resolution=1.0):
    """
    Calcule la cubature par la méthode Grille (Raster).
    Plus rapide, interpole sur une grille régulière (emprise commune).

    resolution: pas de la grille en unités du terrain (ex: 1.0 pour 1m).
    Retourne: (déblais, remblais) en unités cubiques.
    """
    _, _, diff = compute_difference_grid(df_tn, df_proj, resolution)

    diff_valid = diff[~np.isnan(diff)]
    cell_area = resolution * resolution
    deblais = np.sum(diff_valid[diff_valid > 0]) * cell_area
    remblais = np.sum(np.abs(diff_valid[diff_valid < 0])) * cell_area

    return float(deblais), float(remblais)
