"""
Geography/grid constants with no heavy dependencies - split out from
grid_generator.py (which needs osmnx/shapely for actual grid creation, a
dependency chain the deployed webapp never touches - see Finding #2 in the
plan) so consumers that only need these plain values don't pull that
import chain in along with them. Currently used by webapp/simulation_runner.py
and scripts/bake_basemap.py; grid_generator.py itself imports from here too,
so there's exactly one place these values are defined.
"""
AKUSE_CENTER_POINT = (6.0993, 0.12821)
AKUSE_BOUNDARY_COORDS = [
    (0.1118, 6.1135),
    (0.1661, 6.1135),
    (0.1661, 6.0677),
    (0.1118, 6.0677),
]
GRID_SIZE = 100
