Usage
=====

.. contents:: Table of Contents
   :local:
   :depth: 1

The package ``pysdic-blender`` provides a collection of examples demonstrating how to render synthetic images of a
``pysdic`` scene with Blender. Each example can be downloaded as a Python script or a Jupyter notebook.

.. note::

   The examples need ``bpy``, available only with Python 3.13 (see :doc:`Installation <installation>`).


A first rendering
-----------------

A rendering is built in four steps: a :class:`pysdic.Camera` and a ``triangle_3`` :class:`pysdic.Mesh` are created
with ``pysdic``, converted into Blender objects (:class:`pysdic_blender.BlenderCamera`,
:class:`pysdic_blender.BlenderMesh`), added to a :class:`pysdic_blender.BlenderExperiment` with a material and a light,
then rendered.

.. code-block:: python

   import numpy
   from pysdic import Camera, PerspectiveExtrinsic, PinholeIntrinsic, compute_wave_speckle, create_3d_triangle_3_heightmap
   from pysdic_blender import BlenderCamera, BlenderExperiment, BlenderMesh, BlenderSpotLight, get_matte_paint_material

   camera = Camera(
       180, 240,
       extrinsic=PerspectiveExtrinsic.from_rt(rvec=[numpy.pi, 0.0, 0.0], tvec=[0.0, 0.0, 5.0]),
       intrinsic=PinholeIntrinsic.from_matrix([[300.0, 0.0, 119.5], [0.0, 300.0, 89.5], [0.0, 0.0, 1.0]]),
   )
   plate = create_3d_triangle_3_heightmap(x_bounds=(-2.5, 2.5), y_bounds=(-2.0, 2.0), n_x=11, n_y=9)
   x, y = numpy.linspace(-2.5, 2.5, 500), numpy.linspace(2.0, -2.0, 400)
   speckle = compute_wave_speckle(numpy.stack(numpy.meshgrid(x, y), axis=-1), wavelength=0.15)

   experiment = BlenderExperiment(background_strength=0.1)
   experiment.add_camera("camera", BlenderCamera(camera))
   experiment.add_mesh("plate", BlenderMesh(plate, texture=speckle), material=get_matte_paint_material())
   experiment.add_spotlight("spot", BlenderSpotLight.looking_at([3.0, 0.0, 6.0], [0.0, 0.0, 0.0], energy=2000.0))
   image = experiment.render("camera", n_samples=64)  # (180, 240), scene-linear


All the examples
----------------

.. include:: /_gallery/index.rst
   :start-after: _gallery:
