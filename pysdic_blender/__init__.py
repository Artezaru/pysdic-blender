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

# Rendering of synthetic images for pysdic with Blender (Cycles), as a complement of
# pysdic.generate_synthetic_image (exact gray levels of the texture, without lighting).
#
# - BlenderCamera: a pysdic.Camera of the OpenCV pinhole model, without distortion (checked);
# - BlenderMesh: a 3D triangle_3 pysdic.Mesh, displaced, with its UV map and its texture;
# - BlenderMaterialBSDF: the parameters of the Principled BSDF shader, default materials;
# - BlenderSpotLight: a spot light;
# - BlenderExperiment: a scene rendered from its cameras; render_blender_image: one image.
#
# The conversions do not import bpy: it is imported at the creation of the first experiment.

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
    "BlenderExperiment",
    "render_blender_image",
]
