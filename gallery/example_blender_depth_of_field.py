r"""
.. _example_blender_depth_of_field:

Depth of field of an inclined camera
====================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

In stereo DIC, the cameras are inclined on the specimen: the depth varies over the field and
only one line of the specimen is exactly in focus. This example renders a speckled plate seen at
45° with a pinhole camera and with a lens at f/8 and f/2, and compares the blur to its
prediction :meth:`pysdic_blender.BlenderCamera.defocus_blur`.

.. important::

    Requires ``pysdic-blender`` (``pip install pysdic-blender``), which installs ``bpy``.

"""

# %%
# The scene
# ---------
#
# A 25 mm lens on a sensor of 10 µm pixels (:math:`f_x = 2500` pixels), 500 mm from the center
# of the plate, inclined at 45° around the :math:`x` axis of the plate.

import numpy
from scipy.spatial.transform import Rotation
from pysdic import Camera, PerspectiveExtrinsic, PinholeIntrinsic, compute_wave_speckle, create_3d_triangle_3_heightmap

angle = numpy.radians(45.0)
position = 500.0 * numpy.array([0.0, -numpy.sin(angle), numpy.cos(angle)])
z_axis = -position / numpy.linalg.norm(position)  # Optical axis towards the center of the plate
y_axis = numpy.cross(z_axis, [1.0, 0.0, 0.0])  # Rows downwards
rotation = numpy.stack((numpy.cross(y_axis, z_axis), y_axis, z_axis))
camera = Camera(
    360, 480,
    extrinsic=PerspectiveExtrinsic.from_rt(
        rvec=Rotation.from_matrix(rotation).as_rotvec().tolist(), tvec=(-rotation @ position).tolist()
    ),
    intrinsic=PinholeIntrinsic.from_matrix([[2500.0, 0.0, 239.5], [0.0, 2500.0, 179.5], [0.0, 0.0, 1.0]]),
)

plate = create_3d_triangle_3_heightmap(x_bounds=(-60.0, 60.0), y_bounds=(-60.0, 60.0), n_x=3, n_y=3)
x, y = numpy.linspace(-60.0, 60.0, 1500), numpy.linspace(60.0, -60.0, 1500)  # First row: v = 1
speckle = compute_wave_speckle(numpy.stack(numpy.meshgrid(x, y), axis=-1), wavelength=0.8)

# %%
# Cameras with a depth of field
# -----------------------------
#
# The aperture of a lens of focal length :math:`f` at the f-number :math:`N` has the diameter
# :math:`D = f / N`, in the unit of the scene (mm). The focus is on the center of the plate:
# its depth :math:`z_c` is the third component of :math:`R \vec{X} + \vec{t}` at the origin.

from pysdic_blender import BlenderCamera

pinhole = BlenderCamera(camera, pixel_size=0.01)  # lens = 25 mm
focus = float(pinhole.translation[2])  # Depth of the origin of the world
f = pinhole.lens  # mm
blender_cameras = {
    "Pinhole": pinhole,
    "f/8": BlenderCamera(camera, pixel_size=0.01, aperture_diameter=f / 8, focus_distance=focus),
    "f/2": BlenderCamera(camera, pixel_size=0.01, aperture_diameter=f / 2, focus_distance=focus),
}
for name, blender_camera in blender_cameras.items():
    print(name, blender_camera)

# %%
# Rendering
# ---------
#
# All the cameras are in the same scene, lit by an LED panel above the camera.

from pysdic_blender import BlenderAreaLight, BlenderExperiment, BlenderMesh, get_matte_paint_material

experiment = BlenderExperiment(background_strength=0.05)
experiment.add_mesh("plate", BlenderMesh(plate, texture=speckle), material=get_matte_paint_material())
experiment.add_arealight(
    "panel", BlenderAreaLight.looking_at(position * 1.2 + [0.0, 0.0, 150.0], [0.0, 0.0, 0.0], size=300.0, energy=1e6)
)
renders = {}
for name, blender_camera in blender_cameras.items():
    experiment.add_camera(name, blender_camera)
    renders[name] = experiment.render(name, n_samples=256)

# %%
# The top of the images is the far side of the plate, the bottom the near side: the blur grows
# on both sides of the line in focus (the middle row). The crops show the top-left corner.

import matplotlib.pyplot as plt

gain = 1.0 / numpy.percentile(renders["Pinhole"], 99.5)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={"height_ratios": [3, 2]})
for column, (name, render) in enumerate(renders.items()):
    axes[0, column].imshow(render * gain, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, column].add_patch(plt.Rectangle((0, 0), 80, 50, fill=False, edgecolor="tab:red"))
    axes[0, column].set_title(name)
    axes[1, column].imshow(render[:50, :80] * gain, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, column].set_title(f"{name}: far corner")
for ax in axes.ravel():
    ax.axis("off")
fig.tight_layout()
plt.show()

# %%
# Predicted blur
# --------------
#
# :meth:`pysdic_blender.BlenderCamera.defocus_blur` gives the diameter of the blur
# :math:`f_x D |1/z_c - 1/z_f|` in pixels. Along the central column of the image, the depth of
# the plate goes from the far side (top) to the near side (bottom).

normalized_y = (numpy.arange(camera.sensor_height) - 179.5) / 2500.0
directions = numpy.stack((numpy.zeros_like(normalized_y), normalized_y, numpy.ones_like(normalized_y)), axis=1)
directions = directions @ pinhole.rotation  # Rays of the central column in the world
center = pinhole.center
depth = (-center[2] / directions[:, 2])  # Depth z_c of the plate z = 0 along each ray (direction with z_c = 1)

fig, ax = plt.subplots(figsize=(7, 4))
for name in ("f/8", "f/2"):
    ax.plot(numpy.arange(camera.sensor_height), blender_cameras[name].defocus_blur(depth), label=name)
ax.set_xlabel("Row of the image")
ax.set_ylabel("Diameter of the blur (pixels)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
