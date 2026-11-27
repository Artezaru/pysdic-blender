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
from typing import Any, Dict, Optional, Tuple, Union

import numpy
from numpy.typing import ArrayLike

from pysdic import Image, Mesh
from ._validation import _as_array, _validate_bool, _validate_choice


# =============================================================================
# Internal helpers
# =============================================================================
def _as_float_array(value: Any, name: str, shape: Tuple[Optional[int], ...]) -> numpy.ndarray:
    # Finite float64 array with a shape where None is any size
    array = numpy.asarray(value, dtype=numpy.float64)
    if array.ndim != len(shape) or any(s is not None and n != s for n, s in zip(array.shape, shape)):
        expected = "(" + ", ".join("N" if s is None else str(s) for s in shape) + ")"
        raise ValueError(f"Input '{name}' must have the shape {expected}. Got shape {array.shape}.")
    if not numpy.all(numpy.isfinite(array)):
        raise ValueError(f"Input '{name}' must have finite values.")
    return array


def _as_faces(value: Any, n_vertices: int) -> numpy.ndarray:
    # Triangles (N_e, 3) of integer indices in [0, N_v)
    array = numpy.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Input 'faces' must have the shape (N_e, 3). Got shape {array.shape}.")
    if array.size > 0 and not numpy.issubdtype(array.dtype, numpy.integer):
        raise TypeError(f"Input 'faces' must contain integers. Got dtype {array.dtype}.")
    array = array.astype(numpy.int64)
    if array.size > 0 and (array.min() < 0 or array.max() >= n_vertices):
        raise ValueError(
            f"Input 'faces' must contain indices in [0, {n_vertices}). Got [{array.min()}, {array.max()}]."
        )
    return array


# =============================================================================
# Mesh
# =============================================================================
class BlenderMesh(object):
    r"""
    The geometry and the texture of a surface to render with Blender, extracted from a
    :class:`pysdic.Mesh`.

    The mesh must be a 3D ``triangle_3`` mesh: a linear triangle is planar, so the surface
    rendered by Blender is exactly the surface of the mesh (a non-planar quadrangle would be
    split by Blender in its own way). Convert the other meshes first, e.g. with
    :func:`pysdic.quadrangle_4_to_triangle_3`.

    The extracted data:

    - ``coordinates`` :math:`(N_v, 3)`: the vertices, displaced by ``displacement``;
    - ``faces`` :math:`(N_e, 3)`: the connectivity (each triangle is a polygon of Blender, its
      loops in the order of its vertices);
    - ``uv`` :math:`(N_e, 3, 2)`: the UV coordinates of the vertices of each triangle, from
      :attr:`pysdic.Mesh.uvmap` (OpenGL convention :math:`(u, v) \in [0, 1]^2`, :math:`v`
      upwards, as in Blender);
    - ``albedo`` :math:`(H_t, W_t)`: the texture mapped to the reflectance in :math:`[0, 1]`,
      :math:`\rho = (g - g_{min}) / (g_{max} - g_{min})` with ``texture_range`` :math:`(g_{min},
      g_{max})`, clipped to :math:`[0, 1]`. The first row of the texture is :math:`v = 1`
      (same convention as :func:`pysdic.generate_synthetic_image`).

    The texture is multiplied by the base color of the material of the mesh (see
    :class:`BlenderMaterialBSDF`). The object is immutable (read-only copies).

    **Export.** :meth:`to_dict` / :meth:`to_json` save the extracted data above (not the
    :class:`pysdic.Mesh`); :meth:`from_dict` / :meth:`from_json` rebuild the Blender mesh from
    them. With a texture, the file contains all the :math:`H_t \times W_t` reflectances (a large
    JSON file for a large texture).

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        A 3D ``triangle_3`` mesh, with a UV map if a texture is given.

    displacement : Optional[ArrayLike], optional
        The displacement :math:`(N_v, 3)` added to the vertices (e.g. the displacement of a
        frame). By default, the vertices of the mesh.

    texture : Optional[Union[:class:`pysdic.Image`, ArrayLike]], optional
        The grayscale texture :math:`(H_t, W_t)` (or :math:`(H_t, W_t, 1)`), e.g. from
        :func:`pysdic.compute_wave_speckle`. By default, no texture (uniform base color).

    texture_range : Tuple[float, float], optional (default: ``(0.0, 255.0)``)
        The gray levels mapped to the reflectances 0 and 1.

    interpolation : str, optional (default: ``"Linear"``)
        The interpolation of the texture by Blender: ``"Closest"``, ``"Linear"`` or ``"Cubic"``.

    smooth : bool, optional (default: ``False``)
        If True, the shading interpolates the normals of the vertices (smooth surface); else each
        triangle is shaded with its own normal (exact geometry of the mesh).

    Raises
    ------
    TypeError
        If ``mesh`` is not a :class:`pysdic.Mesh`.

    ValueError
        If the mesh is not a 3D ``triangle_3`` mesh, has non-finite vertices, has no UV map with
        a texture, or an input has a wrong shape.

    Examples
    --------
    .. code-block:: python

        from pysdic_blender import BlenderMesh

        blender_mesh = BlenderMesh(mesh, displacement=analysis.get_displacement(1), texture=speckle)
        blender_mesh.to_json("mesh_frame_1.json")
        same_mesh = BlenderMesh.from_json("mesh_frame_1.json")

    """

    __slots__ = ("_coordinates", "_faces", "_uv", "_albedo", "_interpolation", "_smooth")

    def __init__(
        self,
        mesh: Mesh,
        *,
        displacement: Optional[ArrayLike] = None,
        texture: Optional[Union[Image, ArrayLike]] = None,
        texture_range: Tuple[float, float] = (0.0, 255.0),
        interpolation: str = "Linear",
        smooth: bool = False,
    ) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
        if mesh.element_type != "triangle_3" or mesh.n_dimensions != 3:
            raise ValueError(
                f"The mesh must be a 3D 'triangle_3' mesh (exact planar faces in Blender). Got a {mesh.n_dimensions}D "
                f"{mesh.element_type!r} mesh: convert it first (e.g. pysdic.quadrangle_4_to_triangle_3)."
            )
        coordinates = numpy.array(mesh.coordinates, dtype=numpy.float64)
        if displacement is not None:
            coordinates = coordinates + _as_array(
                displacement, "displacement", numpy.float64, shape=(mesh.n_vertices, 3), finite=True
            )
        if not numpy.all(numpy.isfinite(coordinates)):
            raise ValueError("The vertices of the mesh (with the displacement) must be finite.")

        uv = None
        if mesh.uvmap is not None:
            uv = numpy.array(mesh.uvmap, dtype=numpy.float64).reshape(mesh.n_elements, 3, 2)

        albedo = None
        if texture is not None:
            if uv is None:
                raise ValueError("The mesh must have a UV map ('mesh.uvmap') to be rendered with a texture.")
            data = texture.image if isinstance(texture, Image) else numpy.asarray(texture)
            data = numpy.asarray(data, dtype=numpy.float64)
            if data.ndim == 3 and data.shape[2] == 1:
                data = data[:, :, 0]
            if data.ndim != 2:
                raise ValueError(f"Input 'texture' must be a grayscale image (H_t, W_t). Got shape {data.shape}.")
            g_min, g_max = _as_array(texture_range, "texture_range", numpy.float64, shape=(2,), finite=True)
            if not g_max > g_min:
                raise ValueError(f"Input 'texture_range' must satisfy min < max. Got ({g_min}, {g_max}).")
            if not numpy.all(numpy.isfinite(data)):
                raise ValueError("Input 'texture' must have finite values.")
            albedo = numpy.clip((data - g_min) / (g_max - g_min), 0.0, 1.0)

        self._setup(coordinates, mesh.elements, uv, albedo, interpolation, smooth)

    def _setup(
        self,
        coordinates: ArrayLike,
        faces: ArrayLike,
        uv: Optional[ArrayLike],
        albedo: Optional[ArrayLike],
        interpolation: str,
        smooth: bool,
    ) -> None:
        # Validate and store the data (shared by __init__ and from_dict), as read-only copies
        self._coordinates = _as_float_array(coordinates, "coordinates", (None, 3)).copy()
        self._faces = _as_faces(faces, self._coordinates.shape[0]).copy()
        n_faces = self._faces.shape[0]
        self._uv = None if uv is None else _as_float_array(uv, "uv", (n_faces, 3, 2)).copy()
        self._albedo = None
        if albedo is not None:
            if self._uv is None:
                raise ValueError("A mesh with a texture ('albedo') must have UV coordinates ('uv').")
            self._albedo = _as_float_array(albedo, "albedo", (None, None)).copy()
            if self._albedo.size == 0:
                raise ValueError("Input 'albedo' must not be empty.")
            if self._albedo.min() < 0.0 or self._albedo.max() > 1.0:
                raise ValueError(
                    f"Input 'albedo' must be in [0, 1]. Got [{self._albedo.min()}, {self._albedo.max()}]."
                )
        self._interpolation = _validate_choice(interpolation, "interpolation", ("Closest", "Linear", "Cubic"))
        self._smooth = _validate_bool(smooth, "smooth")
        for array in (self._coordinates, self._faces, self._uv, self._albedo):
            if array is not None:
                array.setflags(write=False)

    def __repr__(self) -> str:
        texture = "no texture" if self._albedo is None else f"texture {self._albedo.shape[1]}x{self._albedo.shape[0]}"
        return f"BlenderMesh({self.n_vertices} vertices, {self.n_faces} triangles, {texture})"

    # ===================================================================
    # Data
    # ===================================================================
    @property
    def coordinates(self) -> numpy.ndarray:
        r"""
        The vertices :math:`(N_v, 3)` given to Blender (float64, read-only): the vertices of the
        :class:`pysdic.Mesh` plus ``displacement``, in the unit of the scene (the unit of the
        cameras and the lights).
        """
        return self._coordinates

    @property
    def faces(self) -> numpy.ndarray:
        r"""
        The triangles :math:`(N_e, 3)` (int64, read-only): the indices (from 0) in
        :attr:`coordinates` of the vertices of each triangle, in the order of
        :attr:`pysdic.Mesh.elements`.

        Each triangle is a polygon of Blender whose loops follow this order: the normal of the
        triangle is given by the right-hand rule, :math:`(\vec{X}_1 - \vec{X}_0) \times
        (\vec{X}_2 - \vec{X}_0)`. Blender renders both sides of the faces.
        """
        return self._faces

    @property
    def uv(self) -> Optional[numpy.ndarray]:
        r"""
        The UV coordinates :math:`(N_e, 3, 2)` (float64, read-only), or None without UV map.

        ``uv[i, j]`` is the :math:`(u, v)` of the vertex ``faces[i, j]`` in the triangle
        :math:`i` (a value per loop of Blender, so a vertex can have different UV coordinates in
        different triangles, e.g. along a seam). OpenGL convention: :math:`(u, v) \in [0, 1]^2`,
        :math:`u` to the right and :math:`v` upwards in the texture.
        """
        return self._uv

    @property
    def albedo(self) -> Optional[numpy.ndarray]:
        r"""
        The reflectance :math:`(H_t, W_t)` of the texture in :math:`[0, 1]` (float64,
        read-only), or None without texture.

        :math:`\rho = \text{clip}((g - g_{min}) / (g_{max} - g_{min}), 0, 1)` with
        ``texture_range``. The first row is :math:`v = 1` (pysdic convention): the rows are
        flipped when the texture is given to Blender (first row at :math:`v = 0`). It is
        **not** multiplied by the base color of the material, which is applied in
        :meth:`BlenderExperiment.add_mesh`.
        """
        return self._albedo

    @property
    def interpolation(self) -> str:
        r"""
        The interpolation of the texture by Blender (``interpolation`` of the Image Texture
        node): ``"Closest"`` (nearest texel), ``"Linear"`` (bilinear) or ``"Cubic"`` (bicubic).
        Outside :math:`[0, 1]^2`, the texture is extended with its border texels.
        """
        return self._interpolation

    @property
    def smooth(self) -> bool:
        r"""
        Whether the shading interpolates the normals of the vertices (``use_smooth`` of the
        polygons of Blender). Only the normal used for the shading changes, not the geometry: the
        silhouette and the shadows stay those of the flat triangles.
        """
        return self._smooth

    @property
    def n_vertices(self) -> int:
        r"""The number of vertices :math:`N_v`."""
        return self._coordinates.shape[0]

    @property
    def n_faces(self) -> int:
        r"""The number of triangles :math:`N_e`."""
        return self._faces.shape[0]

    def has_same_topology(self, other: BlenderMesh) -> bool:
        r"""
        Check if another mesh has the same triangles and UV coordinates (only the vertices can
        differ), i.e. if it can replace this one in a scene (see
        :meth:`BlenderExperiment.update_mesh`).
        """
        if not isinstance(other, BlenderMesh):
            raise TypeError(f"Input 'other' must be a BlenderMesh. Got type {type(other)}.")
        same_uv = (self._uv is None and other._uv is None) or (
            self._uv is not None and other._uv is not None and numpy.array_equal(self._uv, other._uv)
        )
        return self.n_vertices == other.n_vertices and numpy.array_equal(self._faces, other._faces) and same_uv

    # ===================================================================
    # Export
    # ===================================================================
    def to_dict(self, description: Optional[str] = None) -> Dict[str, Any]:
        r"""
        Export the mesh to a dictionary: ``{"type": "BlenderMesh", "description": ...,
        "coordinates", "faces", "uv", "albedo", "interpolation", "smooth"}`` (nested lists;
        ``uv`` and ``albedo`` only if not None).

        The data are those given to Blender (displaced vertices, reflectances of the texture),
        not the :class:`pysdic.Mesh` nor the gray levels of the texture.
        """
        data: Dict[str, Any] = {"type": "BlenderMesh"}
        if description is not None:
            if not isinstance(description, str):
                raise TypeError(f"Input 'description' must be a string. Got type {type(description)}.")
            data["description"] = description
        data["coordinates"] = self._coordinates.tolist()
        data["faces"] = self._faces.tolist()
        if self._uv is not None:
            data["uv"] = self._uv.tolist()
        if self._albedo is not None:
            data["albedo"] = self._albedo.tolist()
        data["interpolation"] = self._interpolation
        data["smooth"] = self._smooth
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlenderMesh:
        r"""
        Create a mesh from a dictionary of :meth:`to_dict`.

        ``coordinates`` and ``faces`` are required; ``uv`` and ``albedo`` are None if absent,
        ``interpolation`` is ``"Linear"`` and ``smooth`` is False if absent. ``albedo`` is the
        reflectance in :math:`[0, 1]` (no ``texture_range``). ``type`` and ``description`` are
        ignored, an unknown key raises a ``KeyError``.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        allowed = {"type", "description", "coordinates", "faces", "uv", "albedo", "interpolation", "smooth"}
        missing = [key for key in ("coordinates", "faces") if key not in data]
        if missing:
            raise KeyError(f"Missing fields of a BlenderMesh: {missing}.")
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise KeyError(f"Unknown fields of a BlenderMesh: {unknown}.")
        mesh = cls.__new__(cls)
        mesh._setup(
            data["coordinates"],
            data["faces"],
            data.get("uv"),
            data.get("albedo"),
            data.get("interpolation", "Linear"),
            data.get("smooth", False),
        )
        return mesh

    def to_json(self, path: str, description: Optional[str] = None) -> None:
        r"""Export the mesh to a JSON file (see :meth:`to_dict`)."""
        with open(path, "w") as file:
            json.dump(self.to_dict(description), file)

    @classmethod
    def from_json(cls, path: str) -> BlenderMesh:
        r"""Create a mesh from a JSON file (see :meth:`from_dict`)."""
        with open(path, "r") as file:
            return cls.from_dict(json.load(file))