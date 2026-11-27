r"""
.. _example_blender_stereo_sequence:

Stereo images of a deforming specimen
=====================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

A stereo DIC test needs the images of both cameras at each frame of the deformation. This
example builds the scene once (two cameras, two lights, a speckled plate), then updates the
vertices of the plate with the displacement of each frame and renders both cameras. The setup is
saved to JSON files to regenerate the images later.

.. important::

    Requires ``pysdic-blender`` (``pip install pysdic-blender``), which installs ``bpy``.

"""

# %%
# Cameras and specimen
# --------------------
#
# Two cameras at ±15° around the :math:`y` axis, 500 mm from a plate of 100 x 80 mm (in mm).

import numpy
from scipy.spatial.transform import Rotation
from pysdic import Camera, PerspectiveExtrinsic, PinholeIntrinsic, compute_wave_speckle, create_3d_triangle_3_heightmap


def stereo_camera(angle):
    # Camera 500 mm from the origin, rotated by `angle` around the y axis, looking at the origin
    rotation = Rotation.from_rotvec([0.0, angle, 0.0]).as_matrix() @ numpy.diag([1.0, -1.0, -1.0])
    rotation = rotation.T  # World-to-camera
    position = 500.0 * numpy.array([numpy.sin(angle), 0.0, numpy.cos(angle)])
    return Camera(
        360, 480,
        extrinsic=PerspectiveExtrinsic.from_rt(
            rvec=Rotation.from_matrix(rotation).as_rotvec().tolist(), tvec=(-rotation @ position).tolist()
        ),
        intrinsic=PinholeIntrinsic.from_matrix([[2200.0, 0.0, 239.5], [0.0, 2200.0, 179.5], [0.0, 0.0, 1.0]]),
    )


cameras = {"left": stereo_camera(numpy.radians(-15.0)), "right": stereo_camera(numpy.radians(15.0))}
plate = create_3d_triangle_3_heightmap(x_bounds=(-50.0, 50.0), y_bounds=(-40.0, 40.0), n_x=101, n_y=81)
x, y = numpy.linspace(-50.0, 50.0, 1000), numpy.linspace(40.0, -40.0, 800)  # First row: v = 1
speckle = compute_wave_speckle(numpy.stack(numpy.meshgrid(x, y), axis=-1), wavelength=0.8)

# %%
# Displacements
# -------------
#
# At the frame :math:`k`, a stretching of :math:`1.5 k` % along :math:`x` and an out-of-plane
# bulge of :math:`2 k` mm in the center of the plate.

coordinates = numpy.asarray(plate.coordinates)
displacements = {}
for frame in range(4):
    displacement = numpy.zeros_like(coordinates)
    displacement[:, 0] = 0.015 * frame * coordinates[:, 0]
    displacement[:, 2] = 2.0 * frame * numpy.exp(-(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2) / (2 * 20.0 ** 2))
    displacements[frame] = displacement

# %%
# The scene
# ---------
#
# Each camera has a spot light next to it (radius of 20 mm). The texture and the material are
# given once, with the mesh of the frame 0.

from pysdic_blender import BlenderCamera, BlenderExperiment, BlenderMesh, BlenderSpotLight, get_matte_paint_material

experiment = BlenderExperiment(background_strength=0.03)
for name, camera in cameras.items():
    blender_camera = BlenderCamera(camera)
    experiment.add_camera(name, blender_camera)
    experiment.add_spotlight(
        f"spot_{name}",
        BlenderSpotLight.looking_at(blender_camera.center + [0.0, 80.0, 0.0], [0.0, 0.0, 0.0], energy=1e6, radius=20.0),
    )
experiment.add_mesh("plate", BlenderMesh(plate, displacement=displacements[0], texture=speckle), material=get_matte_paint_material())

# %%
# Rendering the frames
# --------------------
#
# :meth:`pysdic_blender.BlenderExperiment.update_mesh` only moves the vertices (same triangles
# and UV map): the texture follows the material points. One gain is used for all the images,
# so that the gray levels of the frames are comparable.

images = {}
for frame, displacement in displacements.items():
    experiment.update_mesh("plate", BlenderMesh(plate, displacement=displacement))
    for name in cameras:
        images[frame, name] = experiment.render(name, n_samples=128, seed=frame)
gain = 220.0 / numpy.percentile(images[0, "left"], 99.5)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, len(displacements), figsize=(4 * len(displacements), 6))
for (frame, name), image in images.items():
    ax = axes[list(cameras).index(name), frame]
    ax.imshow(numpy.clip(gain * image, 0.0, 255.0), cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Frame {frame}, {name}")
    ax.axis("off")
fig.tight_layout()
plt.show()

# %%
# Saving the setup
# ----------------
#
# The cameras, the lights and the meshes are exported to JSON files and read back without the
# pysdic objects (the cameras then have ``camera = None``). A mesh file contains the
# reflectances of its texture: the mesh of a frame is exported with the texture (the meshes given
# to ``update_mesh`` have none).

import os
import tempfile

folder = tempfile.mkdtemp(prefix="pysdic_blender_setup_")
for name in cameras:
    experiment.get_camera(name).to_json(os.path.join(folder, f"camera_{name}.json"), description=f"Camera {name}")
    experiment.get_spotlight(f"spot_{name}").to_json(os.path.join(folder, f"spot_{name}.json"))
BlenderMesh(plate, displacement=displacements[3], texture=speckle).to_json(
    os.path.join(folder, "plate_frame_3.json"), description="Plate at the frame 3"
)

camera_left = BlenderCamera.from_json(os.path.join(folder, "camera_left.json"))
print(camera_left, camera_left.camera)
print(BlenderMesh.from_json(os.path.join(folder, "plate_frame_3.json")))
print(sorted(os.listdir(folder)))
