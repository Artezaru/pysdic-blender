API Reference
=============

.. currentmodule:: pysdic_blender

.. contents:: Table of Contents
   :local:
   :depth: 1

This section contains a detailed description of the objects included in ``pysdic-blender``.
The reference describes how the methods work and which parameters can be used.


Rendering a scene
-----------------

A Blender scene is built from the objects of ``pysdic``:

.. code-block:: text

   pysdic                         pysdic-blender
   ------                         --------------
   Camera ──────────────────────► BlenderCamera ──────┐
   Mesh (triangle_3) + texture ─► BlenderMesh ────────┤
                                  BlenderMaterialBSDF ┼─► BlenderExperiment ──► render(camera) ──► image
                                  BlenderSpotLight ───┤
                                  BlenderAreaLight ───┘

The pose of the camera is converted from the OpenCV convention (:math:`z` forward, :math:`y` down) to the Blender
convention (:math:`z` backward, :math:`y` up): with the world-to-camera rotation :math:`R` and translation
:math:`\vec{t}` of the extrinsic, the center of the camera is :math:`\vec{C} = -R^T \vec{t}` and its orientation in
Blender is :math:`R^T \, \text{diag}(1, -1, -1)`.

The render is a linear intensity (float64, no quantization): a weighted sum of the RGB channels of Cycles for a
monochrome sensor (``channel_weights``), or the RGB image with ``color=True``. The gray levels, the noise and the
quantization of the sensor are added in ``pysdic``.

.. note::

   Blender has a single scene per process: only one :class:`BlenderExperiment` is active at a time (a new experiment
   resets the scene).


Experiment
----------

- :doc:`BlenderExperiment <api/BlenderExperiment>`: The Blender scene (cameras, meshes, lights, background) and its rendering.
- :doc:`render_blender_image <api/render_blender_image>`: Render one image in a single call.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./api/BlenderExperiment.rst
   ./api/render_blender_image.rst


Camera and mesh
---------------

- :doc:`BlenderCamera <api/BlenderCamera>`: A Blender camera from a :class:`pysdic.Camera` (OpenCV pinhole model only), with an optional depth of field.
- :doc:`BlenderMesh <api/BlenderMesh>`: A Blender mesh from a ``triangle_3`` :class:`pysdic.Mesh` with its UV map and its texture.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./api/BlenderCamera.rst
   ./api/BlenderMesh.rst


Materials
---------

- :doc:`BlenderMaterialBSDF <api/BlenderMaterialBSDF>`: The material of a mesh (Principled BSDF).
- :doc:`get_matte_paint_material <api/get_matte_paint_material>`: A matte paint material.
- :doc:`get_steel_material <api/get_steel_material>`: A steel material.
- :doc:`get_iron_material <api/get_iron_material>`: An iron material.
- :doc:`get_copper_material <api/get_copper_material>`: A copper material.
- :doc:`get_titanium_material <api/get_titanium_material>`: A titanium material.
- :doc:`get_mirror_material <api/get_mirror_material>`: A mirror material.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./api/BlenderMaterialBSDF.rst
   ./api/get_matte_paint_material.rst
   ./api/get_steel_material.rst
   ./api/get_iron_material.rst
   ./api/get_copper_material.rst
   ./api/get_titanium_material.rst
   ./api/get_mirror_material.rst


Lights
------

- :doc:`BlenderSpotLight <api/BlenderSpotLight>`: A spot light (cone of light from an emitting sphere), created from its position and direction or with ``looking_at``.
- :doc:`BlenderAreaLight <api/BlenderAreaLight>`: An area light (square, rectangle, disk or ellipse emitting from one side), e.g. an LED panel.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./api/BlenderSpotLight.rst
   ./api/BlenderAreaLight.rst