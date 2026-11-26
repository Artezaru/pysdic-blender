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

import os
import tempfile
from numbers import Integral, Real
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Union

import numpy
from numpy.typing import ArrayLike

from pysdic import Camera, Image, Mesh
from ._validation import _as_array, _validate_choice, _validate_integer, _validate_real
from .blender_camera import BlenderCamera
from .blender_material import BlenderMaterialBSDF
from .blender_mesh import BlenderMesh
from .blender_spotlight import BlenderSpotLight


# =============================================================================
# Internal helpers
# =============================================================================
def _import_bpy() -> Any:
    # bpy is imported at the first use only: it is slow to import (it loads Blender)
    try:
        import bpy
    except ImportError as error:
        raise ImportError(
            "Missing 'bpy' (Blender as a Python module): install it with 'pip install bpy'. Each release of bpy "
            "supports exactly one version of Python (e.g. bpy 5.1 for Python 3.13): use the version of Python of "
            "the release of Blender."
        ) from error
    return bpy


def _validate_name(name: str, kind: str) -> str:
    # Name of a Blender object: a string of at most 50 characters (Blender limits the names to
    # 63 characters, and suffixes are added to the data blocks of the object)
    if not isinstance(name, str) or not name:
        raise TypeError(f"Input 'name' of the {kind} must be a non-empty string. Got {name!r}.")
    if len(name) > 50:
        raise ValueError(f"Input 'name' of the {kind} must have at most 50 characters. Got {len(name)}.")
    return name


def _enable_gpu(bpy: Any, backend: str) -> None:
    # Enable the GPU devices of a backend of Cycles (CUDA, OPTIX, HIP, METAL, ONEAPI)
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = backend
    preferences.get_devices()
    devices = [device for device in preferences.devices if device.type == backend]
    if not devices:
        raise RuntimeError(f"No GPU device of type {backend!r} found by Cycles: use device='CPU' or another backend.")
    for device in preferences.devices:
        device.use = device.type == backend


# =============================================================================
# Experiment
# =============================================================================
class BlenderExperiment(object):
    r"""
    A Blender scene of cameras, meshes and spot lights, rendered with Cycles.

    The objects of pysdic are converted when they are added (see :class:`BlenderCamera`,
    :class:`BlenderMesh`, :class:`BlenderMaterialBSDF`, :class:`BlenderSpotLight`). The scene is
    rendered from one camera at a time with :meth:`render`, which returns the image as an array.

    For a sequence of frames, update the scene between the renders, e.g. the vertices of a mesh
    displaced by the displacement of each frame (:meth:`update_mesh`).

    .. important::

        Requires ``bpy`` (Blender as a Python module), imported at the creation of the first
        experiment. Each release of ``bpy`` supports exactly one version of Python.

        Blender has one global state: creating an experiment **clears all the data of Blender**
        (factory settings, empty scene), and only the last created experiment can be used (the
        previous ones raise an error).

    **Colors.** The render is saved in OpenEXR (32-bit float, scene-linear values, without the
    tone curve of the view transform) and returned as a linear intensity: the mean of the RGB
    channels. Map it to gray levels and quantize it in pysdic (e.g. a gain, then
    :func:`pysdic.add_gaussian_noise` and :func:`pysdic.quantize_image`), as for
    :func:`pysdic.generate_synthetic_image`.

    Parameters
    ----------
    background_color : ArrayLike, optional (default: ``(1.0, 1.0, 1.0)``)
        The RGB color of the world (ambient light).

    background_strength : Real, optional (default: ``0.0``)
        The strength of the ambient light (0: only the spot lights).

    Examples
    --------
    .. code-block:: python

        from pysdic_blender import BlenderCamera, BlenderExperiment, BlenderMesh, BlenderSpotLight, get_matte_paint_material

        experiment = BlenderExperiment(background_strength=0.2)
        experiment.add_camera("left", BlenderCamera(left_camera))
        experiment.add_camera("right", BlenderCamera(right_camera))
        experiment.add_spotlight("spot", BlenderSpotLight.looking_at([0.0, 0.0, 600.0], [0.0, 0.0, 0.0], energy=5e5))
        experiment.add_mesh("specimen", BlenderMesh(mesh, texture=speckle), material=get_matte_paint_material())

        images = {}
        for frame, displacement in enumerate(displacements):
            experiment.update_mesh("specimen", BlenderMesh(mesh, displacement=displacement))
            images[frame] = [experiment.render(name, n_samples=256) for name in ("left", "right")]

    """

    _current: ClassVar[Optional[BlenderExperiment]] = None

    def __init__(
        self,
        *,
        background_color: ArrayLike = (1.0, 1.0, 1.0),
        background_strength: Real = 0.0,
    ) -> None:
        bpy = _import_bpy()
        bpy.ops.wm.read_factory_settings(use_empty=True)  # Empty scene, default preferences
        BlenderExperiment._current = self
        self._cameras: Dict[str, BlenderCamera] = {}
        self._meshes: Dict[str, BlenderMesh] = {}
        self._spotlights: Dict[str, BlenderSpotLight] = {}
        self.set_background(background_color, background_strength)

    # ===================================================================
    # Internal
    # ===================================================================
    def _bpy(self) -> Any:
        # The module bpy, after checking that this experiment owns the state of Blender
        if BlenderExperiment._current is not self:
            raise RuntimeError(
                "This BlenderExperiment is no longer active: a newer one has cleared the data of Blender."
            )
        return _import_bpy()

    def _check_new_name(self, bpy: Any, name: str, kind: str) -> str:
        name = _validate_name(name, kind)
        if name in bpy.data.objects:
            raise ValueError(f"An object named {name!r} already exists in the scene.")
        return name

    @staticmethod
    def _set_pose(obj: Any, location: numpy.ndarray, quaternion: numpy.ndarray) -> None:
        obj.location = tuple(float(v) for v in location)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = tuple(float(v) for v in quaternion)

    def _remove_object(self, name: str, registry: Dict[str, Any], kind: str) -> None:
        bpy = self._bpy()
        if name not in registry:
            raise KeyError(f"No {kind} named {name!r} in the experiment. Available: {sorted(registry)}.")
        obj = bpy.data.objects[name]
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if kind == "mesh":
            for material in list(data.materials):
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image is not None:
                        bpy.data.images.remove(node.image)
                bpy.data.materials.remove(material)
            bpy.data.meshes.remove(data)
        elif kind == "camera":
            bpy.data.cameras.remove(data)
        else:
            bpy.data.lights.remove(data)
        del registry[name]

    # ===================================================================
    # Scene
    # ===================================================================
    @property
    def camera_names(self) -> Tuple[str, ...]:
        r"""The names of the cameras."""
        return tuple(self._cameras)

    @property
    def mesh_names(self) -> Tuple[str, ...]:
        r"""The names of the meshes."""
        return tuple(self._meshes)

    @property
    def spotlight_names(self) -> Tuple[str, ...]:
        r"""The names of the spot lights."""
        return tuple(self._spotlights)

    def set_background(self, color: ArrayLike = (1.0, 1.0, 1.0), strength: Real = 0.0) -> None:
        r"""
        Set the world of the scene: a uniform ambient light of an RGB ``color`` and a
        ``strength`` (non-negative, 0 for no ambient light).
        """
        bpy = self._bpy()
        color = _as_array(color, "color", numpy.float64, shape=(3,), finite=True)
        strength = _validate_real(strength, "strength", non_negative=True, finite=True)
        scene = bpy.context.scene
        if scene.world is None:
            scene.world = bpy.data.worlds.new("[pysdic]_world")
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes["Background"]
        background.inputs["Color"].default_value = (*color.tolist(), 1.0)
        background.inputs["Strength"].default_value = strength

    # ===================================================================
    # Cameras
    # ===================================================================
    def add_camera(self, name: str, camera: BlenderCamera) -> None:
        r"""Add a camera (see :class:`BlenderCamera`) to the scene, under a unique ``name``."""
        bpy = self._bpy()
        name = self._check_new_name(bpy, name, "camera")
        if not isinstance(camera, BlenderCamera):
            raise TypeError(f"Input 'camera' must be a BlenderCamera. Got type {type(camera)}.")
        data = bpy.data.cameras.new(name)
        data.type = "PERSP"
        data.lens_unit = "MILLIMETERS"
        data.lens = camera.lens
        data.sensor_width = camera.sensor_width
        data.sensor_height = camera.sensor_height
        data.sensor_fit = camera.sensor_fit
        data.shift_x = camera.shift_x
        data.shift_y = camera.shift_y
        data.clip_start = camera.clip_start
        data.clip_end = camera.clip_end
        obj = bpy.data.objects.new(name, data)
        self._set_pose(obj, camera.location, camera.quaternion)
        bpy.context.scene.collection.objects.link(obj)
        self._cameras[name] = camera

    def get_camera(self, name: str) -> BlenderCamera:
        r"""The camera named ``name``."""
        if name not in self._cameras:
            raise KeyError(f"No camera named {name!r} in the experiment. Available: {sorted(self._cameras)}.")
        return self._cameras[name]

    def remove_camera(self, name: str) -> None:
        r"""Remove the camera named ``name`` from the scene."""
        self._remove_object(name, self._cameras, "camera")

    # ===================================================================
    # Meshes
    # ===================================================================
    def add_mesh(self, name: str, mesh: BlenderMesh, *, material: Optional[BlenderMaterialBSDF] = None) -> None:
        r"""
        Add a mesh (see :class:`BlenderMesh`) to the scene with a material (Principled BSDF, see
        :class:`BlenderMaterialBSDF`; by default the default material of Blender).

        The texture of the mesh, multiplied by the base color of the material, is the base color
        of the shader: a float image of Blender with the reflectances (linear values, no color
        transform).
        """
        bpy = self._bpy()
        name = self._check_new_name(bpy, name, "mesh")
        if not isinstance(mesh, BlenderMesh):
            raise TypeError(f"Input 'mesh' must be a BlenderMesh. Got type {type(mesh)}.")
        if material is None:
            material = BlenderMaterialBSDF()
        if not isinstance(material, BlenderMaterialBSDF):
            raise TypeError(f"Input 'material' must be a BlenderMaterialBSDF. Got type {type(material)}.")

        # Geometry: one polygon per triangle, its loops in the order of its vertices
        data = bpy.data.meshes.new(name)
        data.from_pydata(mesh.coordinates.tolist(), [], mesh.faces.tolist())
        data.polygons.foreach_set("use_smooth", [mesh.smooth] * mesh.n_faces)
        if mesh.uv is not None:
            uv_layer = data.uv_layers.new(name="[pysdic]_uv")
            uv_layer.data.foreach_set("uv", mesh.uv.astype(numpy.float32).ravel())
        data.update()

        # Material: Principled BSDF, base color from the texture (times the base color of the material)
        blender_material = bpy.data.materials.new(f"[pysdic]_{name}_mat")
        blender_material.use_nodes = True
        nodes = blender_material.node_tree.nodes
        principled = nodes.get("Principled BSDF")
        if principled is None:
            raise RuntimeError("The new material of Blender has no 'Principled BSDF' node.")
        for input_name, value in material.principled_inputs().items():
            principled.inputs[input_name].default_value = value
        if mesh.albedo is not None:
            base = numpy.asarray(material.base_color if material.base_color is not None else (1.0, 1.0, 1.0, 1.0))
            height, width = mesh.albedo.shape
            image = bpy.data.images.new(f"[pysdic]_{name}_tex", width=width, height=height, alpha=False, float_buffer=True)
            image.colorspace_settings.name = "Non-Color"  # Reflectances used as they are
            pixels = numpy.empty((height, width, 4), dtype=numpy.float32)
            pixels[..., :3] = mesh.albedo[::-1, :, numpy.newaxis] * base[:3]  # Blender: first row at v = 0
            pixels[..., 3] = 1.0
            image.pixels.foreach_set(pixels.ravel())
            image.update()
            texture = nodes.new("ShaderNodeTexImage")
            texture.name = f"[pysdic]_{name}_tex"
            texture.image = image
            texture.interpolation = mesh.interpolation
            texture.extension = "EXTEND"
            blender_material.node_tree.links.new(texture.outputs["Color"], principled.inputs["Base Color"])
        data.materials.append(blender_material)

        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        self._meshes[name] = mesh

    def update_mesh(self, name: str, mesh: BlenderMesh) -> None:
        r"""
        Replace the vertices of a mesh of the scene by those of ``mesh``, which must have the same
        triangles and UV coordinates (see :meth:`BlenderMesh.has_same_topology`), e.g. the mesh
        displaced by the displacement of another frame. The texture and the material are kept.
        """
        bpy = self._bpy()
        if name not in self._meshes:
            raise KeyError(f"No mesh named {name!r} in the experiment. Available: {sorted(self._meshes)}.")
        if not self._meshes[name].has_same_topology(mesh):
            raise ValueError(f"The new mesh must have the triangles and the UV coordinates of the mesh {name!r}.")
        data = bpy.data.objects[name].data
        data.vertices.foreach_set("co", mesh.coordinates.astype(numpy.float32).ravel())
        data.update()
        self._meshes[name] = mesh

    def get_mesh(self, name: str) -> BlenderMesh:
        r"""The mesh named ``name`` (its last update)."""
        if name not in self._meshes:
            raise KeyError(f"No mesh named {name!r} in the experiment. Available: {sorted(self._meshes)}.")
        return self._meshes[name]

    def remove_mesh(self, name: str) -> None:
        r"""Remove the mesh named ``name`` (with its material and texture) from the scene."""
        self._remove_object(name, self._meshes, "mesh")

    # ===================================================================
    # Spot lights
    # ===================================================================
    def add_spotlight(self, name: str, spotlight: BlenderSpotLight) -> None:
        r"""Add a spot light (see :class:`BlenderSpotLight`) to the scene, under a unique ``name``."""
        bpy = self._bpy()
        name = self._check_new_name(bpy, name, "spot light")
        if not isinstance(spotlight, BlenderSpotLight):
            raise TypeError(f"Input 'spotlight' must be a BlenderSpotLight. Got type {type(spotlight)}.")
        data = bpy.data.lights.new(name, type="SPOT")
        data.energy = spotlight.energy
        data.spot_size = spotlight.spot_size
        data.spot_blend = spotlight.spot_blend
        obj = bpy.data.objects.new(name, data)
        self._set_pose(obj, spotlight.position, spotlight.quaternion)
        bpy.context.scene.collection.objects.link(obj)
        self._spotlights[name] = spotlight

    def get_spotlight(self, name: str) -> BlenderSpotLight:
        r"""The spot light named ``name``."""
        if name not in self._spotlights:
            raise KeyError(f"No spot light named {name!r} in the experiment. Available: {sorted(self._spotlights)}.")
        return self._spotlights[name]

    def remove_spotlight(self, name: str) -> None:
        r"""Remove the spot light named ``name`` from the scene."""
        self._remove_object(name, self._spotlights, "spot light")

    # ===================================================================
    # Render
    # ===================================================================
    def render(
        self,
        camera: str,
        *,
        n_samples: Integral = 128,
        filter_width: Real = 1.5,
        seed: Integral = 0,
        device: str = "CPU",
        gpu_backend: str = "CUDA",
    ) -> numpy.ndarray:
        r"""
        Render the scene from a camera with Cycles.

        Parameters
        ----------
        camera : str
            The name of the camera.

        n_samples : Integral, optional (default: ``128``)
            The number of samples per pixel (the noise of Cycles decreases as
            :math:`1 / \sqrt{n}`). No denoising is applied.

        filter_width : Real, optional (default: ``1.5``)
            The width in pixels of the pixel filter of Cycles (Blackman-Harris): the blur of the
            render.

        seed : Integral, optional (default: ``0``)
            The seed of the sampling of Cycles (reproducible renders).

        device : str, optional (default: ``"CPU"``)
            ``"CPU"`` or ``"GPU"``.

        gpu_backend : str, optional (default: ``"CUDA"``)
            The backend of the GPU devices: ``"CUDA"``, ``"OPTIX"``, ``"HIP"``, ``"METAL"`` or
            ``"ONEAPI"`` (only with ``device="GPU"``).

        Returns
        -------
        numpy.ndarray
            The linear intensity :math:`(H, W)` (float64), mean of the RGB channels, with the
            rows of the image from top to bottom (pysdic convention).

        """
        bpy = self._bpy()
        blender_camera = self.get_camera(camera)
        n_samples = _validate_integer(n_samples, "n_samples", min_value=1)
        filter_width = _validate_real(filter_width, "filter_width", positive=True, finite=True)
        seed = _validate_integer(seed, "seed", min_value=0)
        device = _validate_choice(device, "device", ("CPU", "GPU"))
        gpu_backend = _validate_choice(gpu_backend, "gpu_backend", ("CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"))

        scene = bpy.context.scene
        scene.camera = bpy.data.objects[camera]
        render = scene.render
        render.engine = "CYCLES"
        render.resolution_x, render.resolution_y = blender_camera.width, blender_camera.height
        render.resolution_percentage = 100
        render.pixel_aspect_x, render.pixel_aspect_y = blender_camera.pixel_aspect
        scene.cycles.samples = n_samples
        scene.cycles.use_denoising = False
        scene.cycles.filter_width = filter_width
        scene.cycles.seed = seed
        scene.cycles.device = device
        if device == "GPU":
            _enable_gpu(bpy, gpu_backend)
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        render.image_settings.file_format = "OPEN_EXR"
        render.image_settings.color_mode = "RGB"
        render.image_settings.color_depth = "32"

        width, height = blender_camera.width, blender_camera.height
        with tempfile.TemporaryDirectory(prefix="pysdic_blender_") as folder:
            render.filepath = os.path.join(folder, "render.exr")
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(render.filepath, check_existing=False)
            try:
                if tuple(image.size) != (width, height):
                    raise RuntimeError(f"The render has the size {tuple(image.size)}, expected {(width, height)}.")
                pixels = numpy.empty(width * height * image.channels, dtype=numpy.float32)
                image.pixels.foreach_get(pixels)
                channels = image.channels
            finally:
                bpy.data.images.remove(image)
        pixels = pixels.reshape(height, width, channels)[::-1]  # Blender: first row at the bottom
        return pixels[..., :3].mean(axis=2).astype(numpy.float64)


# =============================================================================
# One image
# =============================================================================
def render_blender_image(
    camera: Camera,
    mesh: Mesh,
    *,
    displacement: Optional[ArrayLike] = None,
    texture: Optional[Union[Image, ArrayLike]] = None,
    texture_range: Tuple[float, float] = (0.0, 255.0),
    material: Optional[BlenderMaterialBSDF] = None,
    spotlights: Sequence[BlenderSpotLight] = (),
    background_color: ArrayLike = (1.0, 1.0, 1.0),
    background_strength: Real = 1.0,
    **render_options: Any,
) -> numpy.ndarray:
    r"""
    Render one image of a mesh seen by a camera with Blender: the counterpart of
    :func:`pysdic.generate_synthetic_image` with the lighting and the materials of Cycles
    (shading, shadows, specular reflections).

    Creates a :class:`BlenderExperiment` (which clears the data of Blender) with the camera (see
    :class:`BlenderCamera`), the mesh (see :class:`BlenderMesh`) and the lights, and renders it.
    For several images of the same scene, use a :class:`BlenderExperiment` directly (the scene
    is built once).

    .. important::

        Requires ``bpy``. The camera must follow the OpenCV pinhole model without distortion,
        and the mesh must be a 3D ``triangle_3`` mesh.

    Parameters
    ----------
    camera : :class:`pysdic.Camera`
        The camera.

    mesh : :class:`pysdic.Mesh`
        The 3D ``triangle_3`` mesh, with a UV map for a texture.

    displacement, texture, texture_range :
        See :class:`BlenderMesh`.

    material : Optional[:class:`BlenderMaterialBSDF`], optional
        The material of the mesh. By default, the default material of Blender.

    spotlights : Sequence[:class:`BlenderSpotLight`], optional
        The spot lights.

    background_color, background_strength :
        The ambient light (see :meth:`BlenderExperiment.set_background`), by default a white
        ambient light of strength 1 (a uniform lighting without spot light).

    **render_options :
        The options of :meth:`BlenderExperiment.render` (``n_samples``, ``filter_width``, ...).

    Returns
    -------
    numpy.ndarray
        The linear intensity :math:`(H, W)` (float64).

    """
    experiment = BlenderExperiment(background_color=background_color, background_strength=background_strength)
    experiment.add_camera("camera", BlenderCamera(camera))
    experiment.add_mesh(
        "mesh", BlenderMesh(mesh, displacement=displacement, texture=texture, texture_range=texture_range), material=material
    )
    for index, spotlight in enumerate(spotlights):
        experiment.add_spotlight(f"spotlight_{index}", spotlight)
    return experiment.render("camera", **render_options)
