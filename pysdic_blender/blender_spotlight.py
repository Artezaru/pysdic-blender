"""
pysdic-blender - Rendering of synthetic images for pysdic with Blender
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import json
from numbers import Real
from typing import Any, Dict, Optional

import numpy
from numpy.typing import ArrayLike

from ._validation import _as_array, _validate_real
from .blender_camera import _rotation_to_quaternion


class BlenderSpotLight(object):
    r"""
    A spot light of Blender: a cone of light emitted from a position along a direction.

    - ``energy``: the power in watts the light would emit without the cone (Blender convention);
    - ``spot_size``: the full opening angle of the cone, in radians in :math:`[0, \pi]`;
    - ``spot_blend``: the fraction of the cone with a smooth edge, in :math:`[0, 1]` (0: sharp
      edge).

    A Blender light emits along its local :math:`-z` axis: its rotation maps :math:`-z` to
    ``direction`` (the rotation around the direction does not matter for a spot light). The
    object is immutable.

    Parameters
    ----------
    position : ArrayLike
        The position :math:`(3,)` of the light in the world.

    direction : ArrayLike
        The direction :math:`(3,)` of the axis of the cone (normalized).

    energy : Real, optional (default: ``1000.0``)
        The power in watts (non-negative).

    spot_size : Real, optional (default: ``pi / 4``)
        The full opening angle in radians.

    spot_blend : Real, optional (default: ``0.15``)
        The fraction of the cone with a smooth edge.

    Examples
    --------
    .. code-block:: python

        from pysdic_blender import BlenderSpotLight

        light = BlenderSpotLight.looking_at([0.0, -200.0, 600.0], [0.0, 0.0, 0.0], energy=5e5)

    """

    __slots__ = ("_position", "_direction", "_energy", "_spot_size", "_spot_blend")

    def __init__(
        self,
        position: ArrayLike,
        direction: ArrayLike,
        *,
        energy: Real = 1000.0,
        spot_size: Real = numpy.pi / 4,
        spot_blend: Real = 0.15,
    ) -> None:
        self._position = _as_array(position, "position", numpy.float64, shape=(3,), finite=True).copy()
        direction = _as_array(direction, "direction", numpy.float64, shape=(3,), finite=True)
        norm = numpy.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("Input 'direction' must be a non-zero vector.")
        self._direction = direction / norm
        self._energy = _validate_real(energy, "energy", non_negative=True, finite=True)
        self._spot_size = _validate_real(spot_size, "spot_size", non_negative=True, finite=True)
        if self._spot_size > numpy.pi:
            raise ValueError(f"Input 'spot_size' must be in [0, pi] radians. Got {spot_size}.")
        self._spot_blend = _validate_real(spot_blend, "spot_blend", non_negative=True, finite=True)
        if self._spot_blend > 1.0:
            raise ValueError(f"Input 'spot_blend' must be in [0, 1]. Got {spot_blend}.")
        self._position.setflags(write=False)
        self._direction.setflags(write=False)

    @classmethod
    def looking_at(cls, position: ArrayLike, target: ArrayLike, **kwargs: Any) -> BlenderSpotLight:
        r"""Create a spot light at ``position`` pointing to ``target`` (other inputs as the constructor)."""
        position = _as_array(position, "position", numpy.float64, shape=(3,), finite=True)
        target = _as_array(target, "target", numpy.float64, shape=(3,), finite=True)
        return cls(position, target - position, **kwargs)

    def __repr__(self) -> str:
        return (
            f"BlenderSpotLight(position={self._position.tolist()}, direction={numpy.round(self._direction, 4).tolist()}, "
            f"energy={self._energy}, spot_size={self._spot_size:.4f}, spot_blend={self._spot_blend})"
        )

    @property
    def position(self) -> numpy.ndarray:
        r"""The position :math:`(3,)` (read-only)."""
        return self._position

    @property
    def direction(self) -> numpy.ndarray:
        r"""The unit direction :math:`(3,)` of the axis of the cone (read-only)."""
        return self._direction

    @property
    def energy(self) -> float:
        r"""The power in watts."""
        return self._energy

    @property
    def spot_size(self) -> float:
        r"""The full opening angle in radians."""
        return self._spot_size

    @property
    def spot_blend(self) -> float:
        r"""The fraction of the cone with a smooth edge."""
        return self._spot_blend

    @property
    def quaternion(self) -> numpy.ndarray:
        r"""The rotation ``(w, x, y, z)`` of the Blender light object (local :math:`-z` along ``direction``)."""
        z_axis = -self._direction
        helper = numpy.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else numpy.array([0.0, 1.0, 0.0])
        x_axis = numpy.cross(helper, z_axis)
        x_axis /= numpy.linalg.norm(x_axis)
        y_axis = numpy.cross(z_axis, x_axis)
        return _rotation_to_quaternion(numpy.stack((x_axis, y_axis, z_axis), axis=1))

    def to_dict(self, description: Optional[str] = None) -> Dict[str, Any]:
        r"""Export the light to a dictionary (``position``, ``direction``, ``energy``, ``spot_size``, ``spot_blend``)."""
        data: Dict[str, Any] = {
            "type": "BlenderSpotLight",
            "position": self._position.tolist(),
            "direction": self._direction.tolist(),
            "energy": self._energy,
            "spot_size": self._spot_size,
            "spot_blend": self._spot_blend,
        }
        if description is not None:
            if not isinstance(description, str):
                raise TypeError(f"Input 'description' must be a string. Got type {type(description)}.")
            data["description"] = description
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlenderSpotLight:
        r"""Create a light from a dictionary of :meth:`to_dict`."""
        if not isinstance(data, dict):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        return cls(
            data["position"], data["direction"],
            energy=data.get("energy", 1000.0), spot_size=data.get("spot_size", numpy.pi / 4),
            spot_blend=data.get("spot_blend", 0.15),
        )

    def to_json(self, path: str, description: Optional[str] = None) -> None:
        r"""Export the light to a JSON file (see :meth:`to_dict`)."""
        with open(path, "w") as file:
            json.dump(self.to_dict(description), file, indent=4)

    @classmethod
    def from_json(cls, path: str) -> BlenderSpotLight:
        r"""Create a light from a JSON file (see :meth:`from_dict`)."""
        with open(path, "r") as file:
            return cls.from_dict(json.load(file))
