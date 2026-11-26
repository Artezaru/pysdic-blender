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

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy


# Field of the material -> (input of the Principled BSDF node of Blender, number of values or None for a scalar)
_PRINCIPLED_INPUTS: Dict[str, Tuple[str, Optional[int]]] = {
    "base_color": ("Base Color", 4),
    "metallic": ("Metallic", None),
    "roughness": ("Roughness", None),
    "IOR": ("IOR", None),
    "alpha": ("Alpha", None),
    "normal": ("Normal", 3),
    "weight": ("Weight", None),
    "subsurface_weight": ("Subsurface Weight", None),
    "subsurface_radius": ("Subsurface Radius", 3),
    "subsurface_scale": ("Subsurface Scale", None),
    "subsurface_IOR": ("Subsurface IOR", None),
    "subsurface_anisotropy": ("Subsurface Anisotropy", None),
    "specular_IOR_level": ("Specular IOR Level", None),
    "specular_tint": ("Specular Tint", 4),
    "anisotropic": ("Anisotropic", None),
    "anisotropic_rotation": ("Anisotropic Rotation", None),
    "tangent": ("Tangent", 3),
    "transmission_weight": ("Transmission Weight", None),
    "coat_weight": ("Coat Weight", None),
    "coat_roughness": ("Coat Roughness", None),
    "coat_IOR": ("Coat IOR", None),
    "coat_tint": ("Coat Tint", 4),
    "coat_normal": ("Coat Normal", 3),
    "sheen_weight": ("Sheen Weight", None),
    "sheen_roughness": ("Sheen Roughness", None),
    "sheen_tint": ("Sheen Tint", 4),
    "emission_color": ("Emission Color", 4),
    "emission_strength": ("Emission Strength", None),
    "thin_film_thickness": ("Thin Film Thickness", None),
    "thin_film_IOR": ("Thin Film IOR", None),
}


@dataclass(frozen=True)
class BlenderMaterialBSDF:
    r"""
    The parameters of the Principled BSDF shader of Blender for the material of a mesh.

    Each field is an input of the shader (see the `Blender manual
    <https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html>`_);
    a field left to None keeps the default value of Blender. The colors are RGBA tuples, the
    vectors (normals, tangent, radius) XYZ tuples.

    With a texture (see :class:`BlenderMesh`), the reflectance of the texture is multiplied by
    ``base_color`` (white by default).

    The object is immutable: use :func:`dataclasses.replace` to change a field, e.g.
    ``dataclasses.replace(get_steel_material(), roughness=0.4)``.

    Raises
    ------
    ValueError
        If a field has a wrong number of values or non-finite values.

    """

    base_color: Optional[Tuple[float, float, float, float]] = None
    metallic: Optional[float] = None
    roughness: Optional[float] = None
    IOR: Optional[float] = None
    alpha: Optional[float] = None
    normal: Optional[Tuple[float, float, float]] = None
    weight: Optional[float] = None
    subsurface_weight: Optional[float] = None
    subsurface_radius: Optional[Tuple[float, float, float]] = None
    subsurface_scale: Optional[float] = None
    subsurface_IOR: Optional[float] = None
    subsurface_anisotropy: Optional[float] = None
    specular_IOR_level: Optional[float] = None
    specular_tint: Optional[Tuple[float, float, float, float]] = None
    anisotropic: Optional[float] = None
    anisotropic_rotation: Optional[float] = None
    tangent: Optional[Tuple[float, float, float]] = None
    transmission_weight: Optional[float] = None
    coat_weight: Optional[float] = None
    coat_roughness: Optional[float] = None
    coat_IOR: Optional[float] = None
    coat_tint: Optional[Tuple[float, float, float, float]] = None
    coat_normal: Optional[Tuple[float, float, float]] = None
    sheen_weight: Optional[float] = None
    sheen_roughness: Optional[float] = None
    sheen_tint: Optional[Tuple[float, float, float, float]] = None
    emission_color: Optional[Tuple[float, float, float, float]] = None
    emission_strength: Optional[float] = None
    thin_film_thickness: Optional[float] = None
    thin_film_IOR: Optional[float] = None

    def __post_init__(self) -> None:
        # Normalize the values: floats or tuples of floats, checked
        for name, (_, size) in _PRINCIPLED_INPUTS.items():
            value = getattr(self, name)
            if value is None:
                continue
            array = numpy.asarray(value, dtype=numpy.float64)
            if size is None and array.shape != ():
                raise ValueError(f"Field '{name}' must be a number. Got shape {array.shape}.")
            if size is not None and array.shape != (size,):
                raise ValueError(f"Field '{name}' must have {size} values. Got shape {array.shape}.")
            if not numpy.all(numpy.isfinite(array)):
                raise ValueError(f"Field '{name}' must be finite. Got {value}.")
            object.__setattr__(self, name, float(array) if size is None else tuple(float(v) for v in array))

    def principled_inputs(self) -> Dict[str, Any]:
        r"""
        The values given to the inputs of the Principled BSDF node: ``{input name: value}``
        for the fields that are not None.
        """
        return {
            _PRINCIPLED_INPUTS[name][0]: getattr(self, name)
            for name in _PRINCIPLED_INPUTS
            if getattr(self, name) is not None
        }

    def to_dict(self, description: Optional[str] = None) -> Dict[str, Any]:
        r"""
        Export the material to a dictionary: ``{"type": "BlenderMaterialBSDF", "description":
        ..., <field>: <value>}`` with the fields that are not None (lists for the tuples).
        """
        data: Dict[str, Any] = {"type": "BlenderMaterialBSDF"}
        if description is not None:
            if not isinstance(description, str):
                raise TypeError(f"Input 'description' must be a string. Got type {type(description)}.")
            data["description"] = description
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None:
                data[field.name] = list(value) if isinstance(value, tuple) else value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlenderMaterialBSDF:
        r"""
        Create a material from a dictionary of :meth:`to_dict` (``"type"`` and
        ``"description"`` are ignored, an unknown key raises a ``KeyError``).
        """
        if not isinstance(data, dict):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        values = {key: value for key, value in data.items() if key not in ("type", "description")}
        unknown = sorted(set(values) - set(_PRINCIPLED_INPUTS))
        if unknown:
            raise KeyError(f"Unknown fields of a BlenderMaterialBSDF: {unknown}.")
        return cls(**values)

    def to_json(self, path: str, description: Optional[str] = None) -> None:
        r"""Export the material to a JSON file (see :meth:`to_dict`)."""
        with open(path, "w") as file:
            json.dump(self.to_dict(description), file, indent=4)

    @classmethod
    def from_json(cls, path: str) -> BlenderMaterialBSDF:
        r"""Create a material from a JSON file (see :meth:`from_dict`)."""
        with open(path, "r") as file:
            return cls.from_dict(json.load(file))


# =============================================================================
# Default materials
# =============================================================================
def get_mirror_material() -> BlenderMaterialBSDF:
    r"""A perfect mirror: white, fully metallic, no roughness."""
    return BlenderMaterialBSDF(base_color=(1.0, 1.0, 1.0, 1.0), metallic=1.0, roughness=0.0, IOR=1.5)


def get_steel_material() -> BlenderMaterialBSDF:
    r"""Polished steel: neutral gray, fully metallic, slightly rough."""
    return BlenderMaterialBSDF(base_color=(0.55, 0.55, 0.55, 1.0), metallic=1.0, roughness=0.2, IOR=2.5)


def get_titanium_material() -> BlenderMaterialBSDF:
    r"""Titanium: slightly bluish gray, fully metallic, slightly rough."""
    return BlenderMaterialBSDF(base_color=(0.5, 0.52, 0.6, 1.0), metallic=1.0, roughness=0.25, IOR=2.5)


def get_iron_material() -> BlenderMaterialBSDF:
    r"""Iron: dark gray, fully metallic, rougher than polished steel."""
    return BlenderMaterialBSDF(base_color=(0.3, 0.3, 0.3, 1.0), metallic=1.0, roughness=0.35, IOR=2.9)


def get_copper_material() -> BlenderMaterialBSDF:
    r"""Copper: reddish orange, fully metallic, slightly rough."""
    return BlenderMaterialBSDF(base_color=(0.955, 0.637, 0.538, 1.0), metallic=1.0, roughness=0.2, IOR=2.7)


def get_matte_paint_material() -> BlenderMaterialBSDF:
    r"""
    A matte white paint (diffuse, not metallic, rough): the usual surface of a speckle
    pattern, whose gray levels barely depend on the point of view.
    """
    return BlenderMaterialBSDF(base_color=(1.0, 1.0, 1.0, 1.0), metallic=0.0, roughness=0.9)
