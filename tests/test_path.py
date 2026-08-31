import numpy as np

from vehicle_sim.path import Path2D


def test_open_path_projection_has_signed_error() -> None:
    path = Path2D(np.array([[0.0, 0.0], [10.0, 0.0]]), loop=False)
    projection = path.project(np.array([4.0, 2.0]))
    assert np.allclose(projection.point, [4.0, 0.0])
    assert np.isclose(projection.s, 4.0)
    assert np.isclose(projection.cross_track_error, 2.0)


def test_loop_path_lookahead_wraps() -> None:
    path = Path2D(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        loop=True,
    )
    projection = path.project(np.array([0.1, 0.0]))
    target = path.lookahead(projection, path.length + 0.5)
    assert np.allclose(target, path.lookahead(projection, 0.5))
