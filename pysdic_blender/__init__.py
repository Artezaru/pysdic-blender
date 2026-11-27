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

# bpy is imported first, when pysdic_blender is loaded: it must be imported before open3d
# (lazily imported by pysdic for the ray casting), both bundle libtbb and the older one of
# open3d breaks bpy (undefined TBB symbol in libopenvdb) if it is loaded first.
try:
    import bpy  # noqa: F401
except ImportError as error:
    cause = f"{type(error).__name__}: {error}"
    if isinstance(error, ModuleNotFoundError) and error.name == "bpy":
        message = (
            "Missing 'bpy' (Blender as a Python module): install it with 'pip install bpy'. "
            "Each release of bpy supports exactly one version of Python (e.g. bpy 5.3 for Python 3.13): "
            f"use the version of Python of the release of Blender. Original error: {cause}"
        )
    else:
        message = (
            "The 'bpy' package is installed but cannot be imported (missing system library, conflict "
            "with a library already loaded, e.g. open3d imported before pysdic_blender, ...). "
            f"Original error: {cause}"
        )
    raise ImportError(message) from error

# Rendering of synthetic images for pysdic with Blender (Cycles), as a complement of
# pysdic.generate_synthetic_image (exact gray levels of the texture, without lighting).
#
# - BlenderCamera: a pysdic.Camera of the OpenCV pinhole model, without distortion (checked),
#   with an optional depth of field;
# - BlenderMesh: a 3D triangle_3 pysdic.Mesh, displaced, with its UV map and its texture;
# - BlenderMaterialBSDF: the parameters of the Principled BSDF shader, default materials;
# - BlenderSpotLight: a spot light (cone, emitting sphere of a given radius);
# - BlenderAreaLight: an area light (square, rectangle, disk or ellipse);
# - BlenderExperiment: a scene rendered from its cameras; render_blender_image: one image.
#
# The description objects (cameras, meshes, materials, lights) only use numpy: only
# BlenderExperiment calls bpy (imported above, at the loading of the package).

from .__version__ import __version__
from .blender_camera import BlenderCamera
from .blender_mesh import BlenderMesh
from .blender_material import (
    BlenderMaterialBSDF,
    get_copper_material,
    get_iron_material,
    get_matte_paint_material,
    get_mirror_material,
    get_steel_material,
    get_titanium_material,
)
from .blender_spotlight import BlenderSpotLight
from .blender_arealight import BlenderAreaLight
from .blender_experiment import BlenderExperiment, render_blender_image

__all__ = [
    "__version__",
    "BlenderCamera",
    "BlenderMesh",
    "BlenderMaterialBSDF",
    "get_copper_material",
    "get_iron_material",
    "get_matte_paint_material",
    "get_mirror_material",
    "get_steel_material",
    "get_titanium_material",
    "BlenderSpotLight",
    "BlenderAreaLight",
    "BlenderExperiment",
    "render_blender_image",
]