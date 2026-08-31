"""Lightweight, dependency-free 2D path utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathProjection:
    """Projection of a vehicle position onto a path."""

    point: np.ndarray
    tangent: np.ndarray
    s: float
    distance: float
    cross_track_error: float
    segment_index: int


class Path2D:
    """Polyline path with vectorized nearest-point and lookahead queries."""

    def __init__(self, points: np.ndarray, loop: bool = True) -> None:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if len(points) < 2:
            raise ValueError("at least two path points are required")
        if not np.all(np.isfinite(points)):
            raise ValueError("path points must be finite")

        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10
        points = points[keep]
        if loop and len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-10:
            points = points[:-1]
        if len(points) < (3 if loop else 2):
            raise ValueError("path has too few unique points")

        self.points = points.copy()
        self.loop = bool(loop)
        if self.loop:
            self._ends = np.roll(self.points, -1, axis=0)
        else:
            self._ends = self.points[1:].copy()
        self._starts = self.points if self.loop else self.points[:-1]
        self._segments = self._ends - self._starts
        self._segment_lengths = np.linalg.norm(self._segments, axis=1)
        if np.any(self._segment_lengths <= 1e-10):
            raise ValueError("path contains a zero-length segment")
        self._cumulative_s = np.concatenate(
            ([0.0], np.cumsum(self._segment_lengths, dtype=float))
        )
        self.length = float(self._cumulative_s[-1])

    @classmethod
    def from_csv(cls, file_path: str, loop: bool = True, delimiter: str = ",") -> "Path2D":
        """Load a path from a CSV file containing x and y columns."""

        values = np.genfromtxt(file_path, delimiter=delimiter, names=True)
        if values.dtype.names is not None and {"x", "y"}.issubset(values.dtype.names):
            points = np.column_stack((values["x"], values["y"]))
        else:
            raw = np.genfromtxt(file_path, delimiter=delimiter)
            if raw.ndim != 2 or raw.shape[1] < 2:
                raise ValueError("CSV must contain at least two numeric columns")
            points = raw[:, :2]
        return cls(points, loop=loop)

    def _segment_at(self, s: float) -> tuple[int, float]:
        if self.loop:
            s = float(s % self.length)
        else:
            s = float(np.clip(s, 0.0, self.length))
        index = int(np.searchsorted(self._cumulative_s, s, side="right") - 1)
        index = min(max(index, 0), len(self._segment_lengths) - 1)
        fraction = (s - self._cumulative_s[index]) / self._segment_lengths[index]
        return index, float(np.clip(fraction, 0.0, 1.0))

    def point_at(self, s: float) -> np.ndarray:
        index, fraction = self._segment_at(s)
        return self._starts[index] + fraction * self._segments[index]

    def tangent_at(self, s: float) -> np.ndarray:
        index, _ = self._segment_at(s)
        return self._segments[index] / self._segment_lengths[index]

    def project(self, point: np.ndarray) -> PathProjection:
        """Project a point onto the closest path segment."""

        point = np.asarray(point, dtype=float).reshape(-1)
        if point.shape != (2,):
            raise ValueError("point must contain x and y")
        offsets = point[None, :] - self._starts
        segment_squared = self._segment_lengths**2
        fraction = np.sum(offsets * self._segments, axis=1) / segment_squared
        fraction = np.clip(fraction, 0.0, 1.0)
        projections = self._starts + fraction[:, None] * self._segments
        distances = np.linalg.norm(point[None, :] - projections, axis=1)
        index = int(np.argmin(distances))
        projected_point = projections[index]
        tangent = self._segments[index] / self._segment_lengths[index]
        error_vector = point - projected_point
        signed_error = float(tangent[0] * error_vector[1] - tangent[1] * error_vector[0])
        s = float(self._cumulative_s[index] + fraction[index] * self._segment_lengths[index])
        if self.loop and s >= self.length:
            s = 0.0
        return PathProjection(
            point=projected_point.copy(),
            tangent=tangent.copy(),
            s=s,
            distance=float(distances[index]),
            cross_track_error=signed_error,
            segment_index=index,
        )

    def lookahead(self, projection: PathProjection, distance: float) -> np.ndarray:
        if distance < 0.0:
            raise ValueError("lookahead distance cannot be negative")
        return self.point_at(projection.s + distance)

    def sample(self, count: int = 200) -> np.ndarray:
        if count < 2:
            raise ValueError("count must be at least two")
        if self.loop:
            samples = np.linspace(0.0, self.length, count, endpoint=False)
        else:
            samples = np.linspace(0.0, self.length, count)
        return np.vstack([self.point_at(s) for s in samples])
