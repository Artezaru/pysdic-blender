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
``bpy`` 5.3 requires Python 3.13. ``pysdic`` supports Python 3.11 and later, so the rendering is kept in a separate
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

``bpy`` is imported when ``pysdic_blender`` is loaded: without it, ``import pysdic_blender`` raises an
``ImportError`` that gives the original error.


Import ``pysdic_blender`` before using open3d
---------------------------------------------

``bpy`` and ``open3d`` (the optional dependency of ``pysdic`` for the ray casting) both bundle the library TBB, and
a process can load only one of them. The version of ``open3d`` is older: if it is loaded first, ``bpy`` cannot be
imported (``undefined symbol`` in ``libopenvdb``).

Import ``pysdic_blender`` at the top of the scripts, before any function of ``pysdic`` that uses ``open3d``
(:meth:`pysdic.Mesh.cast_rays`, :func:`pysdic.generate_synthetic_image` with a 3D mesh, ...):

.. code-block:: python

    import pysdic_blender  # First: loads bpy and its version of TBB
    import pysdic

    image = pysdic.generate_synthetic_image(camera, mesh, texture)  # Imports open3d: OK

.. note::

   ``pysdic`` imports ``open3d`` only at the first call of a feature that needs it: ``import pysdic`` before
   ``import pysdic_blender`` is fine, as long as no ray casting is done in between.


Rendering on a server without display
-------------------------------------

On a Linux server or a CI runner without display (e.g. GitHub Actions), ``bpy`` and ``open3d`` need some system
libraries that are not always installed. If the import fails with ``cannot open shared object file``, install them,
e.g. on Ubuntu:

.. code-block:: bash

    sudo apt-get install -y libegl1 libgl1 libgomp1 \
      libx11-6 libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libsm6 libice6


Requirements on the scene
-------------------------

Blender only knows the pinhole camera model and splits non-planar polygons its own way. A scene is accepted only if:

- the **camera** follows the OpenCV pinhole model: a :class:`pysdic.PerspectiveExtrinsic`, a
  :class:`pysdic.PinholeIntrinsic` without skew and a :class:`pysdic.NoDistortion` (an error is raised otherwise);
- the **mesh** is a ``triangle_3`` mesh with a UV map (convert a quadrangle mesh with
  :func:`pysdic.quadrangle_4_to_triangle_3`).