"""Writes proj_reference.json: PROJ's projections of sample positions for
every EPSG code and descriptor kind orb-crs knows. Run with pyproj (PROJ 9):

    python3 -I crates/orb-crs/tests/proj_reference.py > crates/orb-crs/tests/proj_reference.json
"""

import json
import warnings

import pyproj
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup

warnings.simplefilter("ignore")

EPSG = [3857, 4326, 2193, 32601, 32633, 32660, 32701, 32760, 25832, 26910, 3006, 3067,
        3413, 3976, 3031, 3995, 5041, 5042, 32661, 32761, 2154, 3034, 3577, 5070, 3338,
        3035, 6931, 6932, 3573, 8857]

# Codes on other datums, from the datum their Helmert transformation starts
# at (ETRS89, or WGS 84 for Britain's), through that transformation even
# where PROJ has a better grid, with GRS 1980 read as WGS 84 as orb-crs
# reads it (a tenth of a millimetre apart).
SHIFTED = [(28992, 4258), (27700, 4326), (2056, 4258), (21781, 4258)]

# Descriptor kinds as PROJ strings on the ellipsoid or sphere orb-crs uses.
DESCRIPTORS = [
    ({"type": "equal-earth"}, "+proj=eqearth +ellps=WGS84", (-180, -89, 180, 89)),
    ({"type": "natural-earth", "lon_0": 10}, "+proj=natearth +R=6378137 +lon_0=10", (-170, -89, 190, 89)),
    ({"type": "equirectangular"}, "+proj=eqc +R=6378137", (-180, -90, 180, 90)),
    ({"type": "lcc", "lat_1": 33, "lat_2": 45, "lon_0": -96, "lat_0": 39},
     "+proj=lcc +lat_1=33 +lat_2=45 +lon_0=-96 +lat_0=39 +ellps=WGS84", (-130, 20, -60, 55)),
    ({"type": "albers", "lat_1": -18, "lat_2": -36, "lon_0": 132},
     "+proj=aea +lat_1=-18 +lat_2=-36 +lon_0=132 +ellps=WGS84", (110, -45, 155, -10)),
    ({"type": "transverse-mercator", "lon_0": 15, "lat_0": 40, "k_0": 0.9999, "x_0": 200000, "y_0": 100000},
     "+proj=tmerc +lon_0=15 +lat_0=40 +k_0=0.9999 +x_0=200000 +y_0=100000 +ellps=WGS84", (10, 30, 20, 60)),
    ({"type": "utm", "zone": 19, "south": 1}, "+proj=utm +zone=19 +south +ellps=WGS84", (-72, -60, -66, 0)),
    ({"type": "laea", "lat_0": 45, "lon_0": -100}, "+proj=laea +lat_0=45 +lon_0=-100 +ellps=WGS84", (-150, 0, -50, 85)),
    ({"type": "polar-stereographic", "lat_ts": -71, "lon_0": 0}, "+proj=stere +lat_0=-90 +lat_ts=-71 +ellps=WGS84", (-180, -90, 180, -50)),
]


def samples(west, south, east, north):
    for i in range(5):
        for j in range(5):
            yield (west + (east - west) * (i + 0.37) / 5, south + (north - south) * (j + 0.61) / 5)


def projection_only(crs):
    """The code's projection on WGS 84, so that no datum transformation
    (such as NAD83's) enters the reference: orb-crs treats these datums as
    WGS 84."""
    if crs.is_geographic:
        return crs
    proj = [p for p in crs.to_proj4().split() if not p.startswith(("+datum", "+ellps", "+towgs84", "+nadgrids", "+type"))]
    if not any(p.startswith(("+a=", "+R=")) for p in proj):
        proj.append("+ellps=WGS84")
    return CRS.from_proj4(" ".join(proj))


def entry(descriptor, crs, area, t=None):
    t = t or Transformer.from_crs(CRS.from_user_input("OGC:CRS84"), crs, always_xy=True)
    points = []
    for lon, lat in samples(*area):
        x, y = t.transform(lon, lat)
        points.append([lon, lat, x, y])
    return {"descriptor": descriptor, "points": points}


out = {"proj": pyproj.proj_version_str, "cases": []}
for code in EPSG:
    crs = CRS.from_epsg(code)
    a = crs.area_of_use
    area = (a.west, max(a.south, -89.5), a.east if a.east > a.west else a.east + 360, min(a.north, 89.5))
    if code == 3857:
        area = (-180, -85, 180, 85)
    out["cases"].append(entry({"type": f"EPSG:{code}"}, projection_only(crs), area))
for code, source in SHIFTED:
    crs = CRS.from_epsg(code)
    a = crs.area_of_use
    helmert = [t for t in TransformerGroup(source, crs, always_xy=True).transformers
               if "helmert" in t.definition][0]
    helmert = Transformer.from_pipeline(helmert.definition.replace("ellps=GRS80", "ellps=WGS84"))
    out["cases"].append(entry({"type": f"EPSG:{code}"}, crs, (a.west, a.south, a.east, a.north), helmert))
for descriptor, proj, area in DESCRIPTORS:
    out["cases"].append(entry(descriptor, CRS.from_proj4(proj), area))
print(json.dumps(out, indent=None, separators=(",", ":")))
