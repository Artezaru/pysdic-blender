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


# =============================================================================
# Internal helpers
# =============================================================================
def _light_axes(direction: numpy.ndarray, up: Optional[numpy.ndarray] = None) -> numpy.ndarray:
    # Rotation (3, 3) of a Blender light object, columns = local axes in the world: the light
    # emits along its local -z (z = -direction), its local y is `up` made orthogonal to the
    # direction (by default an arbitrary deterministic axis)
    z_axis = -direction
    if up is None:
        helper = numpy.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else numpy.array([0.0, 1.0, 0.0])
        x_axis = numpy.cross(helper, z_axis)
        x_axis /= numpy.linalg.norm(x_axis)
        y_axis = numpy.cross(z_axis, x_axis)
    else:
        y_axis = up - numpy.dot(up, z_axis) * z_axis
        norm = numpy.linalg.norm(y_axis)
        if norm < 1e-9 * numpy.linalg.norm(up):
            raise ValueError("Input 'up' must not be parallel to the direction of the light.")
        y_axis = y_axis / norm
        x_axis = numpy.cross(y_axis, z_axis)
    return numpy.stack((x_axis, y_axis, z_axis), axis=1)


# =============================================================================
# Spot light
# =============================================================================
class BlenderSpotLight(object):
    r"""
    A spot light of Blender: a cone of light emitted from a position along a direction.

    - ``energy``: the power in watts the light would emit without the cone (Blender convention);
    - ``spot_size``: the full opening angle of the cone, in radians in :math:`[0, \pi]`;
    - ``spot_blend``: the fraction of the cone with a smooth edge, in :math:`[0, 1]` (0: sharp
      edge);
    - ``color``: the linear RGB color of the light (multiplies the emitted power on each
      channel, white by default);
    - ``radius``: the radius of the emitting sphere (``shadow_soft_size`` of Blender), 0 for a
      point source.

    **Radius.** Cycles renders the light as a sphere of radius ``radius`` emitting the same
    total power. A point source (``radius = 0``, default) gives sharp shadows and tiny,
    very intense specular highlights on a glossy surface; a real projector (a few centimeters)
    gives soft shadows (penumbra of width about ``2 radius`` times the ratio of the distances
    occluder-surface / light-occluder) and larger, less saturated highlights. The sphere is
    physically correct (the "soft falloff" of Blender is disabled).

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

    color : ArrayLike, optional (default: ``(1.0, 1.0, 1.0)``)
        The RGB color :math:`(3,)` of the light, non-negative linear values (scene-linear
        Rec.709, as all the colors of Blender). The power of the channel :math:`c` is
        ``energy * color[c]``.

    radius : Real, optional (default: ``0.0``)
        The radius of the emitting sphere, in the unit of the scene (non-negative, 0 for a point
        source). Not the default 0.25 of Blender, whose meaning depends on the unit of the scene.

    Examples
    --------
    .. code-block:: python

        from pysdic_blender import BlenderSpotLight

        light = BlenderSpotLight.looking_at([0.0, -200.0, 600.0], [0.0, 0.0, 0.0], energy=5e5)

    """

    __slots__ = ("_position", "_direction", "_energy", "_spot_size", "_spot_blend", "_color", "_radius")

    def __init__(
        self,
        position: ArrayLike,
        direction: ArrayLike,
        *,
        energy: Real = 1000.0,
        spot_size: Real = numpy.pi / 4,
        spot_blend: Real = 0.15,
        color: ArrayLike = (1.0, 1.0, 1.0),
        radius: Real = 0.0,
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
        self._color = _as_array(color, "color", numpy.float64, shape=(3,), finite=True).copy()
        if numpy.any(self._color < 0.0):
            raise ValueError(f"Input 'color' must be non-negative. Got {self._color.tolist()}.")
        self._radius = _validate_real(radius, "radius", non_negative=True, finite=True)
        self._position.setflags(write=False)
        self._direction.setflags(write=False)
        self._color.setflags(write=False)

    @classmethod
    def looking_at(cls, position: ArrayLike, target: ArrayLike, **kwargs: Any) -> BlenderSpotLight:
        r"""Create a spot light at ``position`` pointing to ``target`` (other inputs as the constructor)."""
        position = _as_array(position, "position", numpy.float64, shape=(3,), finite=True)
        target = _as_array(target, "target", numpy.float64, shape=(3,), finite=True)
        return cls(position, target - position, **kwargs)

    def __repr__(self) -> str:
        return (
            f"BlenderSpotLight(position={self._position.tolist()}, direction={numpy.round(self._direction, 4).tolist()}, "
            f"energy={self._energy}, spot_size={self._spot_size:.4f}, spot_blend={self._spot_blend}, "
            f"color={self._color.tolist()}, radius={self._radius})"
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
    def color(self) -> numpy.ndarray:
        r"""The linear RGB color :math:`(3,)` of the light (read-only)."""
        return self._color

    @property
    def radius(self) -> float:
        r"""The radius of the emitting sphere (``shadow_soft_size`` of Blender), 0 for a point source."""
        return self._radius

    @property
    def quaternion(self) -> numpy.ndarray:
        r"""The rotation ``(w, x, y, z)`` of the Blender light object (local :math:`-z` along ``direction``)."""
        return _rotation_to_quaternion(_light_axes(self._direction))

    def to_dict(self, description: Optional[str] = None) -> Dict[str, Any]:
        r"""
        Export the light to a dictionary (``position``, ``direction``, ``energy``, ``spot_size``,
        ``spot_blend``, ``color``, ``radius``).
        """
        data: Dict[str, Any] = {
            "type": "BlenderSpotLight",
            "position": self._position.tolist(),
            "direction": self._direction.tolist(),
            "energy": self._energy,
            "spot_size": self._spot_size,
            "spot_blend": self._spot_blend,
            "color": self._color.tolist(),
            "radius": self._radius,
        }
        if description is not None:
            if not isinstance(description, str):
                raise TypeError(f"Input 'description' must be a string. Got type {type(description)}.")
            data["description"] = description
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlenderSpotLight:
        r"""
        Create a light from a dictionary of :meth:`to_dict` (the default values of the
        constructor for the missing optional fields, e.g. a white light without ``color``).
        """
        if not isinstance(data, dict):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        return cls(
            data["position"], data["direction"],
            energy=data.get("energy", 1000.0), spot_size=data.get("spot_size", numpy.pi / 4),
            spot_blend=data.get("spot_blend", 0.15), color=data.get("color", (1.0, 1.0, 1.0)),
            radius=data.get("radius", 0.0),
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