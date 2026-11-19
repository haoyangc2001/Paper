"""Isolated synthetic-fixture terrain config for actual IsaacLab 0.47.2.

The installed importer always samples generator origins. Its plane branch
uses a spaced grid; no newer use_terrain_origins option is assumed here.
Original benchmark and repetition terrain configurations stay unchanged.
"""
import math


def configure_fixture_plane(cfg, spacing):
    spacing=float(spacing)
    if not math.isfinite(spacing) or spacing<=0:
        raise ValueError('fixture clone spacing must be finite and positive')
    cfg.scene.env_spacing=spacing
    terrain=cfg.scene.terrain
    terrain.terrain_type='plane'
    terrain.terrain_generator=None
    terrain.env_spacing=spacing
    return {'terrain_type':'plane','terrain_generator':None,'env_spacing_m':spacing,
            'scope':'isolated controlled fixtures only; no benchmark/repetition terrain change',
            'expected_installed_branch':'TerrainImporter plane -> configure_env_origins() -> grid'}
