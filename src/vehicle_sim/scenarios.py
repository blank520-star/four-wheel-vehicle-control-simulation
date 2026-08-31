"""Small deterministic paths used by examples and regression tests."""

from __future__ import annotations

import numpy as np

from .path import Path2D


def make_circle_path(radius: float = 20.0, count: int = 400) -> Path2D:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if count < 16:
        raise ValueError("count must be at least 16")
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Path2D(np.column_stack((radius * np.cos(angle), radius * np.sin(angle))))


def make_sine_path(
    length: float = 80.0,
    amplitude: float = 3.0,
    wavelength: float = 20.0,
    count: int = 800,
) -> Path2D:
    if length <= 0.0 or wavelength <= 0.0:
        raise ValueError("length and wavelength must be positive")
    if count < 4:
        raise ValueError("count must be at least 4")
    x = np.linspace(0.0, length, count)
    y = amplitude * np.sin(2.0 * np.pi * x / wavelength)
    return Path2D(np.column_stack((x, y)), loop=False)


def make_oval_path(
    half_length: float = 30.0,
    half_width: float = 10.0,
    count: int = 500,
) -> Path2D:
    if half_length <= half_width or half_width <= 0.0:
        raise ValueError("half_length must be greater than half_width")
    if count < 32:
        raise ValueError("count must be at least 32")
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Path2D(np.column_stack((half_length * np.cos(angle), half_width * np.sin(angle))))
