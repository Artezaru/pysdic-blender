"""
Consistency of the Blender cameras with the pysdic cameras.

A plate textured with Gaussian spots is rendered by Blender (pure diffuse material under a
uniform ambient light, so the intensity is proportional to the reflectance). The centroid of
each spot in the render must be the centroid predicted by the projection of pysdic, for poses
and intrinsics covering all the conversions of BlenderCamera: f_x != f_y in both directions,
off-center principal point, portrait image, horizontal and vertical sensor fits, roll.

The predicted centroid integrates the reflectance of the spot projected by pysdic, weighted by
the Jacobian of the projection: the projection of the center of a spot is not the center of
its image under perspective (bias ~ f r^2 / z^2).

The example ``example_blender_projection_check.py`` of the gallery shows the same check.
"""

import numpy
import pytest

pytest.importorskip("pysdic_blender")  # Requires bpy

from scipy.spatial.transform import Rotation

from pysdic import Camera, PerspectiveExtrinsic, PinholeIntrinsic, create_3d_triangle_3_heightmap
from pysdic_blender import BlenderCamera, BlenderExperiment, BlenderMaterialBSDF, BlenderMesh


SIGMA = 0.75  # Standard deviation of the spots on the plate
CENTERS = numpy.array([[x, y, 0.0] for y in (-30.0, -10.0, 10.0, 30.0) for x in (-40.0, -20.0, 0.0, 20.0, 40.0)])
HALF_SIZE = 150.0  # The plate covers the whole field of view
N_TEXELS = 1600
WINDOW = 9  # Half size of the window of a centroid, in pixels
N_SAMPLES = 512
TOLERANCE = 0.02  # Maximum error of a centroid, in pixels

# (name, height, width, fx, fy, cx, cy, position, target, roll)
CONFIGURATIONS = [
    ("frontal", 240, 320, 1000.0, 1000.0, 159.5, 119.5, (0.0, 0.0, 500.0), (0.0, 0.0, 0.0), 0.0),
    ("fy > fx, off-center, roll", 240, 320, 1000.0, 1150.0, 171.3, 108.2, (60.0, -40.0, 480.0), (5.0, -3.0, 0.0), 0.3),
    ("portrait, inclined", 320, 240, 1100.0, 1100.0, 112.0, 170.5, (0.0, -320.0, 380.0), (0.0, 0.0, 0.0), 0.0),
    ("fx > fy, vertical fit, roll", 260, 300, 1200.0, 1000.0, 140.7, 135.9, (-300.0, 150.0, 400.0), (0.0, 0.0, 0.0), -0.6),
]


# =============================================================================
# Helpers
# =============================================================================
def look_at_camera(height, width, fx, fy, cx, cy, position, target, roll=0.0):
    # OpenCV camera at `position` looking at `target`, the world y axis upwards in the image
    position, target = numpy.asarray(position, dtype=float), numpy.asarray(target, dtype=float)
    z = (target - position) / numpy.linalg.norm(target - position)
    up = numpy.array([0.0, 1.0, 0.0])
    y = -(up - (up @ z) * z)
    y /= numpy.linalg.norm(y)
    x = numpy.cross(y, z)
    c, s = numpy.cos(roll), numpy.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    rotation = numpy.stack((x, y, z))
    extrinsic = PerspectiveExtrinsic.from_rt(
        rvec=Rotation.from_matrix(rotation).as_rotvec().tolist(), tvec=(-rotation @ position).tolist()
    )
    intrinsic = PinholeIntrinsic.from_matrix([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return Camera(height, width, extrinsic=extrinsic, intrinsic=intrinsic)


def project(camera, points):
    # Pixel coordinates (x, y) of world points with the pysdic camera (no distortion)
    normalized = numpy.asarray(camera.extrinsic.apply(points).x)
    return numpy.asarray(camera.intrinsic.apply(normalized).x)


def uv_affine_map(mesh):
    # (M, b) with uv = M (x, y) + b on the plane z = 0, fitted on the UV map of the mesh
    xyz = numpy.asarray(mesh.coordinates)[numpy.asarray(mesh.elements)].reshape(-1, 3)
    uv = numpy.asarray(mesh.uvmap).reshape(-1, 2)
    assert numpy.abs(xyz[:, 2]).max() < 1e-12, "The plate must be in the plane z = 0."
    design = numpy.column_stack((xyz[:, :2], numpy.ones(len(xyz))))
    coefficients = numpy.linalg.lstsq(design, uv, rcond=None)[0]
    assert numpy.abs(design @ coefficients - uv).max() < 1e-9, "The UV map of the plate must be affine."
    return coefficients[:2].T, coefficients[2]


def gaussian_spots(xy, centers, sigma):
    # Reflectance of the spots at the points xy (..., 2), truncated at 4 sigma
    albedo = numpy.zeros(xy.shape[:-1])
    for center in centers:
        d2 = numpy.sum((xy - center[:2]) ** 2, axis=-1)
        albedo += numpy.where(d2 < (4 * sigma) ** 2, numpy.exp(-d2 / (2 * sigma ** 2)), 0.0)
    return albedo


def spots_texture(mesh, centers, sigma, n_texels):
    # Reflectance texture (row 0: v = 1, texel centers at (j + 0.5) / n) of the spots
    M, b = uv_affine_map(mesh)
    u = (numpy.arange(n_texels) + 0.5) / n_texels
    U, V = numpy.meshgrid(u, 1.0 - u)
    xy = (numpy.stack((U, V), axis=-1) - b) @ numpy.linalg.inv(M).T
    return gaussian_spots(xy, centers, sigma)


def expected_centroid(camera, center, sigma, n=129):
    # Centroid of the image of a spot: reflectance projected by pysdic, weighted by the Jacobian
    s = numpy.linspace(-4 * sigma, 4 * sigma, n)
    S, T = numpy.meshgrid(s, s)
    points = numpy.stack((center[0] + S, center[1] + T, numpy.zeros_like(S)), axis=-1)
    pixels = project(camera, points.reshape(-1, 3)).reshape(n, n, 2)
    h = s[1] - s[0]
    d_s, d_t = numpy.gradient(pixels, h, axis=1), numpy.gradient(pixels, h, axis=0)
    jacobian = numpy.abs(d_s[..., 0] * d_t[..., 1] - d_s[..., 1] * d_t[..., 0])
    weights = gaussian_spots(points[..., :2], center[None], sigma) * jacobian
    return numpy.sum(weights[..., None] * pixels, axis=(0, 1)) / weights.sum()


def measured_centroid(image, guess, window):
    # Centroid (x, y) of the intensity in a window around the guess, or None out of the image
    col, row = int(round(guess[0])), int(round(guess[1]))
    if row - window < 0 or col - window < 0 or row + window >= image.shape[0] or col + window >= image.shape[1]:
        return None
    patch = image[row - window:row + window + 1, col - window:col + window + 1]
    rows, cols = numpy.mgrid[row - window:row + window + 1, col - window:col + window + 1]
    return numpy.array([numpy.sum(patch * cols), numpy.sum(patch * rows)]) / patch.sum()


# =============================================================================
# Test
# =============================================================================
@pytest.fixture(scope="module")
def plate():
    mesh = create_3d_triangle_3_heightmap(x_bounds=(-HALF_SIZE, HALF_SIZE), y_bounds=(-HALF_SIZE, HALF_SIZE), n_x=3, n_y=3)
    albedo = spots_texture(mesh, CENTERS, SIGMA, N_TEXELS)
    return BlenderMesh(mesh, texture=albedo, texture_range=(0.0, 1.0), interpolation="Linear")


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=[c[0] for c in CONFIGURATIONS])
def test_blender_camera_matches_pysdic_projection(plate, configuration):
    name, *parameters = configuration
    camera = look_at_camera(*parameters)

    # Pure diffuse material under a uniform white ambient light: intensity proportional to the reflectance
    experiment = BlenderExperiment(background_color=(1.0, 1.0, 1.0), background_strength=1.0)
    experiment.add_camera("camera", BlenderCamera(camera))
    material = BlenderMaterialBSDF(base_color=(1.0, 1.0, 1.0, 1.0), metallic=0.0, roughness=1.0, specular_IOR_level=0.0)
    experiment.add_mesh("plate", plate, material=material)
    image = experiment.render("camera", n_samples=N_SAMPLES, seed=0)
    assert image.shape == (camera.sensor_height, camera.sensor_width)

    errors = []
    for center in CENTERS:
        expected = expected_centroid(camera, center, SIGMA)
        measured = measured_centroid(image, expected, WINDOW)
        if measured is not None:
            errors.append(measured - expected)
    errors = numpy.array(errors)
    assert len(errors) >= 10, f"{name}: only {len(errors)} spots in the image."
    norms = numpy.linalg.norm(errors, axis=1)
    assert norms.max() < TOLERANCE, (
        f"{name}: maximum error {norms.max():.4f} px (mean error {errors.mean(axis=0)} px, {len(errors)} spots)."
    )
