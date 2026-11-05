# 0032: Raster tiles make room like vector tiles

On the globe demo (Liberty on the v8 globe), the ocean showed tile-shaped
gaps, a hard straight edge or most of it missing, "coming in and out with
zoom". At low zooms Liberty's ocean is its `natural_earth` shaded-relief
raster. After some zooming in and out at 2560×1440 in headless Chrome,
z1–z5 raster tiles failed with "out of scene memory; the tile is not drawn"
(over a hundred in three rounds) and drew nothing until their cache
entries left.

## Cause

Decision 0027 taught vector tiles to make room: evict unused tiles, wait
while other tiles load, then give up for a coarser tile and say so. Raster
tiles kept the old path: one try, then empty. On the globe a view covers
1.6–2.5× Mercator's tiles and fills are subdivided for the sphere, so the
cached vector tiles filled the arenas (16.69 M of 16.78 M vertex words,
14.4 M of 16.8 M index words) and a raster tile's quad, a few words, had no
room.

## What changed

Raster and DEM tiles check that their meshes fit, evict unused tiles of any
source (freeing their texture layers too) until they do, wait while other
tiles load, and only then give up: an `error` event and a warning, with the
tile marked missing so its ancestor stands in rather than nothing.

## Measurements

Three rounds of easing between about z1 and z5.6 around the owner's three
views (#2.70/36.36/135.40, #3.12/35.62/135.24, #2.84/38.35/152.54) at
2560×1440, devicePixelRatio 2, main thread: over a hundred raster tiles
dropped before, none after, and each view draws the same tiles every round
(66, 132 and 86).
