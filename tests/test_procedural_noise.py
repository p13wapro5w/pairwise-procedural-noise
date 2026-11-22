import pytest
import render.procedural_noise as mod

CASES = [
    ("screen_frequency", (2.0, 1.0)),
    ("world_scale", (0.5, 2.0)),
    ("normalize", ([0.5, 0.7, 0.9],)),
    ("fixed_seed", (42, 7)),
    ("analytic_derivative", (1.0, 2.0)),
    ("layer_budget", (8, 4)),
    ("lod_cache", (1, 2)),
    ("precision", (0.5, 0.25)),
    ("triplanar_weight", (0.5, 0.2)),
    ("filter_fit", (1, 2)),
    ("noise_type", (1, 2)),
    ("octave_count", (4, 8)),
    ("cache_key", (1, 2)),
    ("tile_size", (64, 128)),
    ("output_range", (0.8, 1.0)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
