# pysdic-blender

Rendering of synthetic images for [pysdic](https://github.com/Artezaru/pysdic) with Blender (Cycles):
shading, shadows and the reflections of materials, as a complement of `pysdic.generate_synthetic_image`
(exact gray levels of the texture, without lighting).

## Why a separate package

Blender as a Python module (`bpy`) supports **exactly one version of Python** per release of Blender
(Blender 5.1: Python 3.13). Keeping it out of pysdic lets pysdic support several versions of Python,
while pysdic-blender follows the version of Python of Blender.

## Installation

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install pysdic-blender   # installs pysdic and bpy
```

## Usage

```python
from pysdic_blender import (BlenderCamera, BlenderExperiment, BlenderMesh, BlenderSpotLight,
                            get_matte_paint_material)

experiment = BlenderExperiment(background_strength=0.1)
experiment.add_camera("left", BlenderCamera(left_camera))      # pysdic.Camera, OpenCV pinhole model
experiment.add_spotlight("spot", BlenderSpotLight.looking_at([0, -200, 600], [0, 0, 0], energy=5e5))
experiment.add_mesh("specimen", BlenderMesh(mesh, texture=speckle), material=get_matte_paint_material())

for frame, displacement in enumerate(displacements):
    experiment.update_mesh("specimen", BlenderMesh(mesh, displacement=displacement))
    intensity = experiment.render("left", n_samples=256)   # (H, W) linear intensity
```

Requirements on the pysdic objects:

- the camera: `PerspectiveExtrinsic`, `PinholeIntrinsic` without skew, no distortion (render
  without the distortion, then apply it with `pysdic.distort_image`);
- the mesh: a 3D `triangle_3` mesh (with a UV map for a texture).

The conversions (`BlenderCamera`, `BlenderMesh`, ...) do not import `bpy`: it is imported at the
creation of the first `BlenderExperiment`.
