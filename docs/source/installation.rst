Installation
============

.. contents:: Table of Contents
   :local:
   :depth: 1

``pysdic-blender`` requires **Python 3.13**. To install it, you can use the following command:

.. code-block:: bash

    pip install pysdic-blender

Or install it directly from the GitHub repository using:

.. code-block:: bash

    pip install git+https://github.com/Artezaru/pysdic-blender.git


Why Python 3.13?
----------------

``bpy`` (Blender as a Python module) is published for only one version of Python per Blender release:
``bpy`` 5.1 requires Python 3.13. ``pysdic`` supports Python 3.11 and later, so the rendering is kept in a separate
package to avoid imposing Python 3.13 on ``pysdic``.

Use a dedicated environment for the rendering, e.g.:

.. code-block:: bash

    python3.13 -m venv blender-env
    source blender-env/bin/activate
    pip install pysdic-blender


Required dependencies
---------------------

The following dependencies are installed with ``pysdic-blender``:

- **pysdic** (https://Artezaru.github.io/pysdic/): the cameras, the meshes and the images of the scene.
- **bpy** (https://pypi.org/project/bpy/): Blender as a Python module (Cycles renderer).
- **NumPy** (https://numpy.org/): the arrays of the meshes, the textures and the rendered images.


Requirements on the scene
-------------------------

Blender only knows the pinhole camera model and splits non-planar polygons its own way. A scene is accepted only if:

- the **camera** follows the OpenCV pinhole model: a :class:`pysdic.PerspectiveExtrinsic`, a
  :class:`pysdic.PinholeIntrinsic` without skew and a :class:`pysdic.NoDistortion` (an error is raised otherwise);
- the **mesh** is a ``triangle_3`` mesh with a UV map (convert a quadrangle mesh with
  :func:`pysdic.quadrangle_4_to_triangle_3`).
