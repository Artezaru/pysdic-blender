Welcome to pysdic-blender's documentation!
=========================================================================================

Description of the package
--------------------------

Rendering of synthetic images with Blender (lighting, shading, materials) from the objects of
`pysdic <https://Artezaru.github.io/pysdic/>`_ (cameras and meshes).

:func:`pysdic.generate_synthetic_image` renders the exact gray levels of a textured mesh, without lighting.
``pysdic-blender`` drives Blender (Cycles renderer) to render the same scene with lighting, shading and the
reflections of a material.

.. important::

   ``pysdic-blender`` is a separate package because ``bpy`` (Blender as a Python module) exists for only one version
   of Python per Blender release: ``pysdic-blender`` requires **Python 3.13** (``bpy`` 5.1), while ``pysdic``
   supports Python 3.11 and later.


Contents
--------

.. grid:: 3

    .. grid-item-card::
      :text-align: center

      Installation
      ^^^

      This section describes how to install the package into a Python environment,
      with ``bpy`` and its version of Python.

      +++

      .. button-ref:: installation
         :expand:
         :color: secondary
         :click-parent:

         To the installation guide

    .. grid-item-card::
      :text-align: center

      API Reference
      ^^^

      The reference guide contains a detailed description of the objects included in ``pysdic-blender``:
      the cameras, the meshes, the materials, the lights and the experiment.

      +++

      .. button-ref:: api
         :expand:
         :color: secondary
         :click-parent:

         To the API reference

    .. grid-item-card::
      :text-align: center

      Examples Gallery
      ^^^

      This section contains examples demonstrating how to render synthetic images of a ``pysdic`` scene
      with Blender.

      +++

      .. button-ref:: usage
         :expand:
         :color: secondary
         :click-parent:

         To the examples gallery

.. toctree::
   :caption: Contents:
   :hidden:

   installation
   api
   usage


Author
------

The package ``pysdic-blender`` was created by the following authors:

- Artezaru <artezaru.github@proton.me>

You can access the package and the documentation with the following URL:

- **Git Plateform**: https://github.com/Artezaru/pysdic-blender.git
- **Online Documentation**: https://Artezaru.github.io/pysdic-blender


License
-------

.. code-block:: text 

   pysdic-blender - Rendering of synthetic images with Blender for pysdic
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
