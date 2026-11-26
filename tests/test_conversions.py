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

# Tests of the conversions (without bpy) and a smoke test of the rendering (with bpy).

import dataclasses

import numpy
import pytest
from scipy.spatial.transform import Rotation

import pysdic
from pysdic_blender import BlenderCamera, BlenderMaterialBSDF, BlenderMesh, BlenderSpotLight, get_steel_material


def make_camera(distortion=None) -> pysdic.Camera:
    return pysdic.Camera(
        120, 160,
        extrinsic=pysdic.PerspectiveExtrinsic.from_rt(rvec=[2.9, 0.3, -0.2], tvec=[0.4, -0.3, 6.0]),
        intrinsic=pysdic.PinholeIntrinsic.from_matrix([[300.0, 0.0, 81.0], [0.0, 320.0, 58.5], [0.0, 0.0, 1.0]]),
        distortion=distortion,
    )


def test_pose_opencv_to_blender():
    # Convention 4 (X_c = R X_w + t) -> convention 0 (X_w = R_b X_b + C): X_b = (x_c, -y_c, -z_c)
    blender_camera = BlenderCamera(make_camera())
    R, t = blender_camera.rotation, blender_camera.translation
    w, x, y, z = blender_camera.quaternion
    R_b = Rotation.from_quat([x, y, z, w]).as_matrix()
    assert numpy.allclose(R_b, R.T @ numpy.diag([1.0, -1.0, -1.0]))
    assert numpy.allclose(blender_camera.location, -R.T @ t)
    points = numpy.random.default_rng(0).normal(size=(10, 3))
    camera_points = points @ R.T + t
    blender_points = (points - blender_camera.location) @ R_b
    assert numpy.allclose(blender_points, camera_points * [1.0, -1.0, -1.0])


def test_pose_matches_projection():
    # The extracted pose and intrinsic reproduce the projection of the pysdic camera
    camera = make_camera()
    blender_camera = BlenderCamera(camera)
    points = numpy.array([[0.1, 0.2, 0.0], [-0.5, 0.3, 0.2], [0.4, -0.4, -0.1]])
    camera_points = points @ blender_camera.rotation.T + blender_camera.translation
    image_points = (camera_points / camera_points[:, 2:]) @ blender_camera.intrinsic_matrix.T
    assert numpy.allclose(image_points[:, :2], camera.project(points).image_points)


def test_camera_with_distortion_is_refused():
    with pytest.raises(TypeError):
        BlenderCamera(make_camera(distortion=pysdic.BrownDistortion(numpy.zeros(5))))


def test_mesh_must_be_triangle_3():
    mesh = pysdic.create_3d_quadrangle_4_heightmap(x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0), n_x=3, n_y=3)
    with pytest.raises(ValueError):
        BlenderMesh(mesh)
    mesh = pysdic.create_3d_triangle_3_heightmap(x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0), n_x=3, n_y=3)
    blender_mesh = BlenderMesh(mesh, texture=numpy.full((4, 4), 255.0))
    assert blender_mesh.uv.shape == (mesh.n_elements, 3, 2)
    assert numpy.all(blender_mesh.albedo == 1.0)


def test_material_round_trip():
    material = dataclasses.replace(get_steel_material(), roughness=0.4)
    assert BlenderMaterialBSDF.from_dict(material.to_dict("test")) == material


def test_spotlight_direction():
    light = BlenderSpotLight.looking_at([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    w, x, y, z = light.quaternion
    direction = Rotation.from_quat([x, y, z, w]).apply([0.0, 0.0, -1.0])  # Blender lights emit along -z
    assert numpy.allclose(direction, -numpy.array([1.0, 2.0, 3.0]) / numpy.sqrt(14.0))


def test_render_smoke():
    pytest.importorskip("bpy")
    from pysdic_blender import render_blender_image

    mesh = pysdic.create_3d_triangle_3_heightmap(x_bounds=(-3.0, 3.0), y_bounds=(-3.0, 3.0), n_x=5, n_y=5)
    camera = pysdic.Camera(
        60, 80,
        extrinsic=pysdic.PerspectiveExtrinsic.from_rt(rvec=[numpy.pi, 0.0, 0.0], tvec=[0.0, 0.0, 5.0]),
        intrinsic=pysdic.PinholeIntrinsic.from_matrix([[100.0, 0.0, 39.5], [0.0, 100.0, 29.5], [0.0, 0.0, 1.0]]),
    )
    image = render_blender_image(camera, mesh, n_samples=4)
    assert image.shape == (60, 80)
    assert numpy.all(numpy.isfinite(image)) and image.max() > 0.0
