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

from ._validation import _as_array, _validate_choice, _validate_real
from .blender_camera import _rotation_to_quaternion
from .blender_spotlight import _light_axes


class BlenderAreaLight(object):
    r"""
    An area light of Blender: a flat emitting surface (square, rectangle, disk or ellipse), e.g.
    an LED panel or a diffuser.

    - ``shape``: ``"SQUARE"`` or ``"DISK"`` (one dimension ``size``), ``"RECTANGLE"`` or
      ``"ELLIPSE"`` (dimensions ``size`` along the local :math:`x` axis and ``size_y`` along the
      local :math:`y` axis);
    - ``energy``: the total power in watts emitted by the surface (Blender convention);
    - ``color``: the linear RGB color of the light (white by default);
    - ``spread``: the angle of the cone of emission of each point of the surface, in radians in
      :math:`]0, \pi]` (:math:`\pi`: a Lambertian emitter; smaller: a directional panel, like a
      honeycomb grid).

    The surface is centered on ``position`` and emits from **one side only**, along
    ``direction`` (the local :math:`-z` axis of the Blender light). Its local :math:`y` axis
    (along ``size_y``) is ``up`` made orthogonal to ``direction``: it orients a rectangle or an
    ellipse around the direction. The object is immutable.

    Parameters
    ----------
    position : ArrayLike
        The center :math:`(3,)` of the surface in the world.

    direction : ArrayLike
        The direction :math:`(3,)` of emission, normal to the surface (normalized).

    size : Real
        The side (``"SQUARE"``, ``"RECTANGLE"``) or the diameter (``"DISK"``, ``"ELLIPSE"``)
        along the local :math:`x` axis, in the unit of the scene (positive).

    size_y : Optional[Real], optional
        The dimension along the local :math:`y` axis, required for ``"RECTANGLE"`` and
        ``"ELLIPSE"``, forbidden for ``"SQUARE"`` and ``"DISK"``.

    shape : str, optional (default: ``"SQUARE"``)
        ``"SQUARE"``, ``"RECTANGLE"``, ``"DISK"`` or ``"ELLIPSE"``.

    up : Optional[ArrayLike], optional
        The direction :math:`(3,)` of the local :math:`y` axis of the surface (made orthogonal to
        ``direction``, must not be parallel to it). By default, an arbitrary deterministic axis
        (see :attr:`up`): give it for a rectangle, an ellipse or a square whose orientation
        matters.

    energy : Real, optional (default: ``1000.0``)
        The power in watts (non-negative).

    color : ArrayLike, optional (default: ``(1.0, 1.0, 1.0)``)
        The RGB color :math:`(3,)`, non-negative linear values.

    spread : Real, optional (default: ``pi``)
        The angle of emission in radians, in :math:`]0, \pi]`.

    Examples
    --------
    .. code-block:: python

        from pysdic_blender import BlenderAreaLight

        panel = BlenderAreaLight.looking_at(
            [0.0, -300.0, 500.0], [0.0, 0.0, 0.0], size=200.0, size_y=100.0, shape="RECTANGLE",
            up=[1.0, 0.0, 0.0], energy=2e5,
        )

    """

    __slots__ = ("_position", "_direction", "_up", "_shape", "_size", "_size_y", "_energy", "_color", "_spread")

    def __init__(
        self,
        position: ArrayLike,
        direction: ArrayLike,
        *,
        size: Real,
        size_y: Optional[Real] = None,
        shape: str = "SQUARE",
        up: Optional[ArrayLike] = None,
        energy: Real = 1000.0,
        color: ArrayLike = (1.0, 1.0, 1.0),
        spread: Real = numpy.pi,
    ) -> None:
        self._position = _as_array(position, "position", numpy.float64, shape=(3,), finite=True).copy()
        direction = _as_array(direction, "direction", numpy.float64, shape=(3,), finite=True)
        norm = numpy.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("Input 'direction' must be a non-zero vector.")
        self._direction = direction / norm
        if up is not None:
            up = _as_array(up, "up", numpy.float64, shape=(3,), finite=True)
            if numpy.linalg.norm(up) == 0.0:
                raise ValueError("Input 'up' must be a non-zero vector.")
        self._up = _light_axes(self._direction, up)[:, 1].copy()  # The actual local y axis

        self._shape = _validate_choice(shape, "shape", ("SQUARE", "RECTANGLE", "DISK", "ELLIPSE"))
        self._size = _validate_real(size, "size", positive=True, finite=True)
        if self._shape in ("RECTANGLE", "ELLIPSE"):
            if size_y is None:
                raise ValueError(f"Input 'size_y' is required for the shape {self._shape!r}.")
            self._size_y = _validate_real(size_y, "size_y", positive=True, finite=True)
        else:
            if size_y is not None:
                raise ValueError(
                    f"Input 'size_y' must be None for the shape {self._shape!r} (one dimension 'size'): use "
                    "'RECTANGLE' or 'ELLIPSE' for two dimensions."
                )
            self._size_y = None

        self._energy = _validate_real(energy, "energy", non_negative=True, finite=True)
        self._color = _as_array(color, "color", numpy.float64, shape=(3,), finite=True).copy()
        if numpy.any(self._color < 0.0):
            raise ValueError(f"Input 'color' must be non-negative. Got {self._color.tolist()}.")
        self._spread = _validate_real(spread, "spread", positive=True, finite=True)
        if self._spread > numpy.pi:
            raise ValueError(f"Input 'spread' must be in ]0, pi] radians. Got {spread}.")
        for array in (self._position, self._direction, self._up, self._color):
            array.setflags(write=False)

    @classmethod
    def looking_at(cls, position: ArrayLike, target: ArrayLike, **kwargs: Any) -> BlenderAreaLight:
        r"""Create an area light centered at ``position`` emitting towards ``target`` (other inputs as the constructor)."""
        position = _as_array(position, "position", numpy.float64, shape=(3,), finite=True)
        target = _as_array(target, "target", numpy.float64, shape=(3,), finite=True)
        return cls(position, target - position, **kwargs)

    def __repr__(self) -> str:
        size = f"{self._size}" if self._size_y is None else f"{self._size}x{self._size_y}"
        return (
            f"BlenderAreaLight({self._shape} {size}, position={self._position.tolist()}, "
            f"direction={numpy.round(self._direction, 4).tolist()}, energy={self._energy}, "
            f"color={self._color.tolist()}, spread={self._spread:.4f})"
        )

    @property
    def position(self) -> numpy.ndarray:
        r"""The center :math:`(3,)` of the surface (read-only)."""
        return self._position

    @property
    def direction(self) -> numpy.ndarray:
        r"""The unit direction :math:`(3,)` of emission, normal to the surface (read-only)."""
        return self._direction

    @property
    def up(self) -> numpy.ndarray:
        r"""
        The unit local :math:`y` axis :math:`(3,)` of the surface, orthogonal to
        :attr:`direction` (along ``size_y``, read-only): ``up`` of the constructor made
        orthogonal, or the default axis chosen when ``up`` is None.
        """
        return self._up

    @property
    def shape(self) -> str:
        r"""The shape: ``"SQUARE"``, ``"RECTANGLE"``, ``"DISK"`` or ``"ELLIPSE"``."""
        return self._shape

    @property
    def size(self) -> float:
        r"""The side or the diameter along the local :math:`x` axis."""
        return self._size

    @property
    def size_y(self) -> Optional[float]:
        r"""The dimension along the local :math:`y` axis (:attr:`up`), None for ``"SQUARE"`` and ``"DISK"``."""
        return self._size_y

    @property
    def area(self) -> float:
        r"""The area of the emitting surface."""
        if self._shape == "SQUARE":
            return self._size ** 2
        if self._shape == "RECTANGLE":
            return self._size * self._size_y
        if self._shape == "DISK":
            return numpy.pi * self._size ** 2 / 4
        return numpy.pi * self._size * self._size_y / 4

    @property
    def energy(self) -> float:
        r"""The total power in watts."""
        return self._energy

    @property
    def color(self) -> numpy.ndarray:
        r"""The linear RGB color :math:`(3,)` of the light (read-only)."""
        return self._color

    @property
    def spread(self) -> float:
        r"""The angle of emission in radians (:math:`\pi`: Lambertian)."""
        return self._spread

    @property
    def quaternion(self) -> numpy.ndarray:
        r"""
        The rotation ``(w, x, y, z)`` of the Blender light object: local :math:`-z` along
        :attr:`direction`, local :math:`y` along :attr:`up`.
        """
        return _rotation_to_quaternion(_light_axes(self._direction, self._up))

    def to_dict(self, description: Optional[str] = None) -> Dict[str, Any]:
        r"""
        Export the light to a dictionary (``position``, ``direction``, ``up``, ``shape``,
        ``size``, ``size_y``, ``energy``, ``color``, ``spread``). ``up`` is the actual local
        :math:`y` axis, so the orientation is kept exactly.
        """
        data: Dict[str, Any] = {
            "type": "BlenderAreaLight",
            "position": self._position.tolist(),
            "direction": self._direction.tolist(),
            "up": self._up.tolist(),
            "shape": self._shape,
            "size": self._size,
            "size_y": self._size_y,
            "energy": self._energy,
            "color": self._color.tolist(),
            "spread": self._spread,
        }
        if description is not None:
            if not isinstance(description, str):
                raise TypeError(f"Input 'description' must be a string. Got type {type(description)}.")
            data["description"] = description
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlenderAreaLight:
        r"""
        Create a light from a dictionary of :meth:`to_dict` (``position``, ``direction`` and
        ``size`` required, the default values of the constructor for the other fields).
        """
        if not isinstance(data, dict):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        return cls(
            data["position"], data["direction"],
            size=data["size"], size_y=data.get("size_y"), shape=data.get("shape", "SQUARE"), up=data.get("up"),
            energy=data.get("energy", 1000.0), color=data.get("color", (1.0, 1.0, 1.0)),
            spread=data.get("spread", numpy.pi),
        )

    def to_json(self, path: str, description: Optional[str] = None) -> None:
        r"""Export the light to a JSON file (see :meth:`to_dict`)."""
        with open(path, "w") as file:
            json.dump(self.to_dict(description), file, indent=4)

    @classmethod
    def from_json(cls, path: str) -> BlenderAreaLight:
        r"""Create a light from a JSON file (see :meth:`from_dict`)."""
        with open(path, "r") as file:
            return cls.from_dict(json.load(file))