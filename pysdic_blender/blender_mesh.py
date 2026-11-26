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

from typing import Optional, Tuple, Union

import numpy
from numpy.typing import ArrayLike

from pysdic import Image, Mesh
from ._validation import _as_array, _validate_bool, _validate_choice


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
        self._coordinates = coordinates
        self._faces = numpy.array(mesh.elements, dtype=numpy.int64)
        self._interpolation = _validate_choice(interpolation, "interpolation", ("Closest", "Linear", "Cubic"))
        self._smooth = _validate_bool(smooth, "smooth")

        self._uv = None
        if mesh.uvmap is not None:
            self._uv = numpy.array(mesh.uvmap, dtype=numpy.float64).reshape(mesh.n_elements, 3, 2)
        self._albedo = None
        if texture is not None:
            if self._uv is None:
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
            self._albedo = numpy.clip((data - g_min) / (g_max - g_min), 0.0, 1.0)

        for array in (self._coordinates, self._faces, self._uv, self._albedo):
            if array is not None:
                array.setflags(write=False)

    def __repr__(self) -> str:
        texture = "no texture" if self._albedo is None else f"texture {self._albedo.shape[1]}x{self._albedo.shape[0]}"
        return f"BlenderMesh({self.n_vertices} vertices, {self.n_faces} triangles, {texture})"

    @property
    def coordinates(self) -> numpy.ndarray:
        r"""The (displaced) vertices :math:`(N_v, 3)` (read-only)."""
        return self._coordinates

    @property
    def faces(self) -> numpy.ndarray:
        r"""The triangles :math:`(N_e, 3)` (read-only)."""
        return self._faces

    @property
    def uv(self) -> Optional[numpy.ndarray]:
        r"""The UV coordinates :math:`(N_e, 3, 2)` of the vertices of each triangle, or None (read-only)."""
        return self._uv

    @property
    def albedo(self) -> Optional[numpy.ndarray]:
        r"""The reflectance of the texture :math:`(H_t, W_t)` in :math:`[0, 1]`, or None (read-only)."""
        return self._albedo

    @property
    def interpolation(self) -> str:
        r"""The interpolation of the texture by Blender."""
        return self._interpolation

    @property
    def smooth(self) -> bool:
        r"""Whether the shading interpolates the normals of the vertices."""
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
