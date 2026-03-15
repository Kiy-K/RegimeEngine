#!/usr/bin/env python3
"""
Generate a real map image of Western Europe (British Isles + France + Benelux)
for the Air Strip One GUI background.

Uses matplotlib with Natural Earth shapefiles (auto-downloaded).
Renders coastlines, country borders, and water in a dark military style.
Output: gui/assets/map_western_europe.png (700×740 pixels)
"""

import os
import sys
import json
import numpy as np

# Ensure project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import shape, MultiPolygon, Polygon

# Map bounds: Western Europe focused on British Isles + France + Benelux
# lon: -12 to 8, lat: 43 to 60
LON_MIN, LON_MAX = -12.0, 8.5
LAT_MIN, LAT_MAX = 43.0, 60.5

# Output size
WIDTH_PX = 700
HEIGHT_PX = 740

# Real geographic coordinates for each sector (lon, lat)
SECTOR_GEO = {
    # Oceania — South England
    0:  (-0.12, 51.51),   # London
    1:  (1.31, 51.13),    # Dover
    2:  (-1.09, 50.80),   # Portsmouth
    3:  (-1.40, 50.90),   # Southampton
    4:  (1.08, 51.28),    # Canterbury
    5:  (-0.14, 50.82),   # Brighton
    # Oceania — Southwest + Wales
    6:  (-2.59, 51.45),   # Bristol
    7:  (-4.14, 50.38),   # Plymouth
    8:  (-3.18, 51.48),   # Cardiff
    # Oceania — Midlands + North
    9:  (-1.90, 52.48),   # Birmingham
    10: (-2.24, 53.48),   # Manchester
    11: (-2.98, 53.41),   # Liverpool
    12: (-1.55, 53.80),   # Leeds
    # Oceania — East Anglia
    13: (1.30, 52.63),    # Norwich
    # Oceania — Scotland
    14: (-3.19, 55.95),   # Edinburgh
    15: (-4.25, 55.86),   # Glasgow
    # Oceania — Ireland
    16: (-6.26, 53.35),   # Dublin
    17: (-5.93, 54.60),   # Belfast
    # Eurasia — Channel Front
    18: (1.86, 50.95),    # Calais
    19: (2.38, 51.03),    # Dunkirk
    20: (0.11, 49.49),    # Le Havre
    21: (-1.62, 49.63),   # Cherbourg
    # Eurasia — Northern France
    22: (2.30, 49.89),    # Amiens
    23: (1.10, 49.44),    # Rouen
    24: (3.06, 50.63),    # Lille
    # Eurasia — Benelux
    25: (4.35, 50.85),    # Brussels
    26: (4.40, 51.22),    # Antwerp
    27: (4.48, 51.92),    # Rotterdam
    28: (4.90, 52.37),    # Amsterdam
    29: (6.13, 49.61),    # Luxembourg
    # Eurasia — Central France
    30: (2.35, 48.86),    # Paris
    31: (1.90, 47.90),    # Orleans
    32: (4.83, 45.76),    # Lyon
    # Eurasia — Atlantic France
    33: (-4.49, 48.39),   # Brest
    34: (-0.58, 44.84),   # Bordeaux
}


def geo_to_pixel(lon, lat):
    """Convert geographic coordinates to pixel coordinates."""
    x = (lon - LON_MIN) / (LON_MAX - LON_MIN) * WIDTH_PX
    y = (1.0 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * HEIGHT_PX
    return int(x), int(y)


def download_natural_earth():
    """Download Natural Earth 110m coastline + countries if not cached."""
    import urllib.request
    import zipfile
    
    cache_dir = os.path.join(_ROOT, "gui", "assets", "ne_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    coast_shp = os.path.join(cache_dir, "ne_110m_land.shp")
    if os.path.exists(coast_shp):
        return cache_dir
    
    url = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
    zip_path = os.path.join(cache_dir, "ne_110m_land.zip")
    
    print(f"Downloading Natural Earth coastlines...")
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(cache_dir)
    
    os.remove(zip_path)
    print(f"Cached at {cache_dir}")
    return cache_dir


def generate_map():
    """Generate the map image."""
    try:
        import shapefile  # pyshp
    except ImportError:
        print("Installing pyshp...")
        os.system(f"{sys.executable} -m pip install pyshp")
        import shapefile
    
    cache_dir = download_natural_earth()
    
    # Read shapefile
    sf = shapefile.Reader(os.path.join(cache_dir, "ne_110m_land"))
    
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH_PX/100, HEIGHT_PX/100), dpi=100)
    fig.patch.set_facecolor('#141820')
    ax.set_facecolor('#1a2540')  # dark ocean
    
    # Draw land masses
    for shape_rec in sf.shapeRecords():
        geom = shape(shape_rec.shape.__geo_interface__)
        if isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        elif isinstance(geom, Polygon):
            polys = [geom]
        else:
            continue
        
        for poly in polys:
            # Check if polygon intersects our view
            minx, miny, maxx, maxy = poly.bounds
            if maxx < LON_MIN or minx > LON_MAX or maxy < LAT_MIN or miny > LAT_MAX:
                continue
            
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color='#2a3020', edgecolor='#3a4535', linewidth=0.5)
    
    # Set bounds
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    out_path = os.path.join(_ROOT, "gui", "assets", "map_western_europe.png")
    fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"Map saved: {out_path}")
    print(f"Size: {os.path.getsize(out_path)} bytes")
    
    # Also save the pixel positions for each sector
    positions = {}
    for sid, (lon, lat) in SECTOR_GEO.items():
        px, py = geo_to_pixel(lon, lat)
        positions[sid] = (px, py)
    
    pos_path = os.path.join(_ROOT, "gui", "assets", "sector_positions.json")
    with open(pos_path, 'w') as f:
        json.dump(positions, f, indent=2)
    
    print(f"Sector positions saved: {pos_path}")
    print(f"Sectors: {len(positions)}")
    
    # Print positions for verification
    for sid, (px, py) in sorted(positions.items()):
        name = list(SECTOR_GEO.keys())[sid] if sid < len(SECTOR_GEO) else "?"
        print(f"  {sid:2d}: ({px:4d}, {py:4d})")
    
    return out_path, positions


if __name__ == "__main__":
    generate_map()
