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

from __future__ import annotations

from numbers import Real
from typing import Any, Dict, Tuple

import numpy

from pysdic import Camera, NoDistortion, PerspectiveExtrinsic, PinholeIntrinsic
from ._validation import _validate_real


# =============================================================================
# Internal helpers
# =============================================================================
def _rotation_to_quaternion(matrix: numpy.ndarray) -> numpy.ndarray:
    # Unit quaternion (w, x, y, z) of a rotation matrix (convention of Blender)
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return numpy.array([w, x, y, z])


def _extract_pose(extrinsic: PerspectiveExtrinsic) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Rotation R (3, 3) and translation t (3,) of X_c = R X_w + t, read from the rays of the
    # extrinsic (independent of the storage of its parameters): the ray of the normalized
    # point (x_n, y_n) has the direction R^T (x_n, y_n, 1) and starts at the center C = -R^T t
    rays = extrinsic.compute_rays(numpy.array([[0.0, 0.0], [1.0, 0.0]]))
    origins, directions = rays.origins.reshape(-1, 3), rays.directions.reshape(-1, 3)
    directions = directions / numpy.linalg.norm(directions, axis=1, keepdims=True)
    z_axis = directions[0]
    x_axis = directions[1] * numpy.sqrt(2.0) - z_axis  # R^T (1, 0, 1) / sqrt(2)
    x_axis /= numpy.linalg.norm(x_axis)
    y_axis = numpy.cross(z_axis, x_axis)
    rotation = numpy.stack((x_axis, y_axis, z_axis))  # Rows: axes of the camera in the world
    return rotation, -rotation @ origins[0]


def _check_pose(extrinsic: PerspectiveExtrinsic, rotation: numpy.ndarray, translation: numpy.ndarray) -> None:
    # The extrinsic must be X_c = R X_w + t followed by the perspective division
    points = numpy.array([[0.0, 0.0, 1.0], [0.3, -0.2, 2.0], [-0.4, 0.5, 3.0], [0.1, 0.2, 0.5]])
    world = (points - translation) @ rotation  # World points in front of the camera
    expected = points[:, :2] / points[:, 2:]
    obtained = numpy.asarray(extrinsic.apply(world).x)
    if not numpy.allclose(obtained, expected, rtol=1e-9, atol=1e-9):
        raise ValueError(
            "The extrinsic of the camera does not follow the OpenCV model X_c = R X_w + t with a perspective "
            f"division (maximum error {numpy.abs(obtained - expected).max():.2e} on test points)."
        )


def _extract_intrinsic(intrinsic: PinholeIntrinsic) -> Tuple[float, float, float, float]:
    # fx, fy, cx, cy of a pinhole intrinsic without skew, checked on test points
    fx, fy, cx, cy = (float(getattr(intrinsic, name)) for name in ("fx", "fy", "cx", "cy"))
    points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.3, -0.7]])
    expected = numpy.stack((fx * points[:, 0] + cx, fy * points[:, 1] + cy), axis=1)
    obtained = numpy.asarray(intrinsic.apply(points).x)
    if not numpy.allclose(obtained, expected, rtol=1e-9, atol=1e-9):
        raise ValueError(
            "The intrinsic of the camera is not a pinhole model x = fx x_n + cx, y = fy y_n + cy without skew "
            f"(maximum error {numpy.abs(obtained - expected).max():.2e} pixels on test points): the skew is not "
            "supported by Blender."
        )
    return fx, fy, cx, cy


# =============================================================================
# Camera
# =============================================================================
class BlenderCamera(object):
    r"""
    The parameters of a Blender camera reproducing a :class:`pysdic.Camera` of the OpenCV
    pinhole model, without distortion.

    The camera must have:

    - a :class:`pysdic.PerspectiveExtrinsic`: :math:`\vec{X}_c = R \vec{X}_w + \vec{t}` followed
      by the perspective division (OpenCV frame: :math:`x` to the right, :math:`y` downwards,
      :math:`z` along the optical axis);
    - a :class:`pysdic.PinholeIntrinsic` without skew: :math:`x = f_x x_n + c_x`,
      :math:`y = f_y y_n + c_y`;
    - no distortion (Blender does not model it).

    Both models are checked on test points, so that the Blender camera projects exactly as the
    pysdic camera, with the same convention of the pixels (centers at integer coordinates).

    .. note::

        For a camera with a distortion, render the images with the same camera without
        distortion, then apply the distortion with :func:`pysdic.distort_image`.

    **Blender parameters.** Blender describes a pinhole camera by a focal length ``lens`` and a
    sensor size in millimeters (only their ratio matters), a sensor fit, a shift of the
    principal point and a pixel aspect ratio. With :math:`a_x = \max(1, f_y / f_x)`,
    :math:`a_y = \max(1, f_x / f_y)` and the resolution :math:`(W, H)`:

    - sensor fit: ``"HORIZONTAL"`` if :math:`a_x W \geq a_y H`, else ``"VERTICAL"``;
    - view factor :math:`v = W` (horizontal) or :math:`v = (a_y / a_x) H` (vertical);
    - ``lens`` :math:`= f_x s / v`, with the sensor size :math:`s = p W` (horizontal) or
      :math:`p H` (vertical) for a pixel size :math:`p`;
    - ``shift_x`` :math:`= -(c_x - (W - 1)/2) / v`, ``shift_y`` :math:`= (c_y - (H - 1)/2)(a_y / a_x) / v`.

    **Pose.** pysdic and OpenCV give the world-to-camera transformation, Blender the pose of the
    camera object in the world (camera-to-world). In the conventions of ``py3dframe`` (frame
    :math:`E` = world, frame :math:`F` = camera), the pose goes from the convention 4 to the
    convention 0, with the change of the axes of the camera from OpenCV (:math:`x` to the right,
    :math:`y` downwards, looking along :math:`+z`) to OpenGL (:math:`y` upwards, looking along
    :math:`-z`):

    .. math::

        \text{OpenCV (convention 4)}: \quad \vec{X}_c = R \vec{X}_w + \vec{t}

        \text{Blender (convention 0)}: \quad \vec{X}_w = R_b \vec{X}_b + \vec{C},
        \qquad R_b = R^T \, \text{diag}(1, -1, -1), \qquad \vec{C} = -R^T \vec{t}

    so that the coordinates of a point in the frame of the Blender camera are
    :math:`\vec{X}_b = (x_c, -y_c, -z_c)`. The ``location`` of the object is :math:`\vec{C}`
    (:attr:`center`) and its ``rotation_quaternion`` is the quaternion of :math:`R_b`
    (:attr:`quaternion`): the inverse of the pose needs **both** the center
    :math:`-R^T \vec{t}` and the transposed rotation :math:`R^T`.

    The object is immutable: it reads the camera once at its creation.

    Parameters
    ----------
    camera : :class:`pysdic.Camera`
        The camera to reproduce.

    pixel_size : Real, optional (default: ``0.01``)
        The size of a pixel in millimeters, giving the sensor size and ``lens`` (only their
        ratio matters for the image). Blender requires ``lens >= 1`` mm: increase it for a
        short focal length.

    clip_start, clip_end : Real, optional (default: ``1e-3`` and ``1e6``)
        The distances of the near and far clipping planes, in the unit of the scene.

    Raises
    ------
    TypeError
        If ``camera`` is not a :class:`pysdic.Camera`, or its transformations are not of the
        types above.

    ValueError
        If a transformation does not follow its model on the test points, or ``lens < 1`` mm.

    Examples
    --------
    .. code-block:: python

        from pysdic import Camera, PerspectiveExtrinsic, PinholeIntrinsic
        from pysdic_blender import BlenderCamera

        camera = Camera(
            1000, 1200,
            extrinsic=PerspectiveExtrinsic.from_rt(rvec=[3.14159, 0.0, 0.0], tvec=[0.0, 0.0, 500.0]),
            intrinsic=PinholeIntrinsic.from_matrix([[2000.0, 0.0, 599.5], [0.0, 2000.0, 499.5], [0.0, 0.0, 1.0]]),
        )
        blender_camera = BlenderCamera(camera)
        print(blender_camera.lens, blender_camera.shift_x, blender_camera.sensor_fit)

    """

    __slots__ = (
        "_camera", "_rotation", "_translation", "_fx", "_fy", "_cx", "_cy", "_pixel_size", "_clip_start", "_clip_end",
    )

    def __init__(
        self,
        camera: Camera,
        *,
        pixel_size: Real = 0.01,
        clip_start: Real = 1e-3,
        clip_end: Real = 1e6,
    ) -> None:
        if not isinstance(camera, Camera):
            raise TypeError(f"Input 'camera' must be a Camera. Got type {type(camera)}.")
        if not isinstance(camera.extrinsic, PerspectiveExtrinsic):
            raise TypeError(
                f"The extrinsic of the camera must be a PerspectiveExtrinsic (OpenCV model). Got "
                f"{type(camera.extrinsic).__name__}."
            )
        if not isinstance(camera.intrinsic, PinholeIntrinsic):
            raise TypeError(
                f"The intrinsic of the camera must be a PinholeIntrinsic (OpenCV model). Got "
                f"{type(camera.intrinsic).__name__}."
            )
        if camera.distortion is not None and not isinstance(camera.distortion, NoDistortion):
            raise TypeError(
                f"The camera must have no distortion (Blender does not model it). Got "
                f"{type(camera.distortion).__name__}: render with the camera without distortion, then apply it with "
                "pysdic.distort_image."
            )
        self._rotation, self._translation = _extract_pose(camera.extrinsic)
        _check_pose(camera.extrinsic, self._rotation, self._translation)
        self._fx, self._fy, self._cx, self._cy = _extract_intrinsic(camera.intrinsic)
        self._camera = camera
        self._pixel_size = _validate_real(pixel_size, "pixel_size", positive=True, finite=True)
        self._clip_start = _validate_real(clip_start, "clip_start", positive=True, finite=True)
        self._clip_end = _validate_real(clip_end, "clip_end", positive=True, finite=True)
        if not self._clip_end > self._clip_start:
            raise ValueError(f"Input 'clip_end' must be greater than 'clip_start'. Got {clip_start} and {clip_end}.")
        for rotation_part in (self._rotation, self._translation):
            rotation_part.setflags(write=False)
        if self.lens < 1.0:
            raise ValueError(
                f"The focal length of the Blender camera is {self.lens:.3f} mm, Blender requires at least 1 mm: "
                f"increase 'pixel_size' (got {self._pixel_size} mm)."
            )

    def __repr__(self) -> str:
        return (
            f"BlenderCamera({self.width}x{self.height} pixels, lens={self.lens:.3f} mm, "
            f"sensor_fit={self.sensor_fit!r}, shift=({self.shift_x:.4f}, {self.shift_y:.4f}))"
        )

    # ===================================================================
    # pysdic camera
    # ===================================================================
    @property
    def camera(self) -> Camera:
        r"""The reproduced :class:`pysdic.Camera`."""
        return self._camera

    @property
    def width(self) -> int:
        r"""The resolution :math:`W` along :math:`x` (pixels)."""
        return int(self._camera.sensor_width)

    @property
    def height(self) -> int:
        r"""The resolution :math:`H` along :math:`y` (pixels)."""
        return int(self._camera.sensor_height)

    @property
    def rotation(self) -> numpy.ndarray:
        r"""The rotation :math:`R` (3, 3) of :math:`\vec{X}_c = R \vec{X}_w + \vec{t}` (read-only)."""
        return self._rotation

    @property
    def translation(self) -> numpy.ndarray:
        r"""The translation :math:`\vec{t}` (3,) of :math:`\vec{X}_c = R \vec{X}_w + \vec{t}` (read-only)."""
        return self._translation

    @property
    def center(self) -> numpy.ndarray:
        r"""The center of the camera :math:`\vec{C} = -R^T \vec{t}` (3,) in the world."""
        return -self._rotation.T @ self._translation

    @property
    def intrinsic_matrix(self) -> numpy.ndarray:
        r"""The intrinsic matrix :math:`K` (3, 3)."""
        return numpy.array([[self._fx, 0.0, self._cx], [0.0, self._fy, self._cy], [0.0, 0.0, 1.0]])

    # ===================================================================
    # Blender parameters
    # ===================================================================
    @property
    def pixel_size(self) -> float:
        r"""The size of a pixel in millimeters."""
        return self._pixel_size

    @property
    def clip_start(self) -> float:
        r"""The distance of the near clipping plane."""
        return self._clip_start

    @property
    def clip_end(self) -> float:
        r"""The distance of the far clipping plane."""
        return self._clip_end

    @property
    def pixel_aspect(self) -> Tuple[float, float]:
        r"""The pixel aspect ratio :math:`(a_x, a_y)` of the render (1 for square pixels)."""
        return max(1.0, self._fy / self._fx), max(1.0, self._fx / self._fy)

    @property
    def sensor_fit(self) -> str:
        r"""The sensor fit, ``"HORIZONTAL"`` or ``"VERTICAL"``."""
        a_x, a_y = self.pixel_aspect
        return "HORIZONTAL" if a_x * self.width >= a_y * self.height else "VERTICAL"

    @property
    def sensor_width(self) -> float:
        r"""The width of the sensor :math:`p W` in millimeters."""
        return self._pixel_size * self.width

    @property
    def sensor_height(self) -> float:
        r"""The height of the sensor :math:`p H` in millimeters."""
        return self._pixel_size * self.height

    @property
    def _view_factor(self) -> float:
        a_x, a_y = self.pixel_aspect
        return float(self.width) if self.sensor_fit == "HORIZONTAL" else (a_y / a_x) * self.height

    @property
    def lens(self) -> float:
        r"""The focal length ``lens`` of the Blender camera in millimeters."""
        sensor = self.sensor_width if self.sensor_fit == "HORIZONTAL" else self.sensor_height
        return self._fx * sensor / self._view_factor

    @property
    def shift_x(self) -> float:
        r"""The shift of the principal point along :math:`x` (fraction of the view factor)."""
        return -(self._cx - (self.width - 1) / 2) / self._view_factor

    @property
    def shift_y(self) -> float:
        r"""The shift of the principal point along :math:`y` (fraction of the view factor)."""
        a_x, a_y = self.pixel_aspect
        return (self._cy - (self.height - 1) / 2) * (a_y / a_x) / self._view_factor

    @property
    def location(self) -> numpy.ndarray:
        r"""The location of the Blender camera object: its center :math:`\vec{C} = -R^T \vec{t}` (convention 0)."""
        return self.center

    @property
    def quaternion(self) -> numpy.ndarray:
        r"""
        The rotation ``(w, x, y, z)`` of the Blender camera object: the quaternion of
        :math:`R_b = R^T \, \text{diag}(1, -1, -1)` (convention 0, OpenGL axes).
        """
        # Columns of R_b: the rows of R (transposed rotation) with the signs of y and z changed
        axes = numpy.stack((self._rotation[0], -self._rotation[1], -self._rotation[2]), axis=1)
        return _rotation_to_quaternion(axes)

    def to_dict(self) -> Dict[str, Any]:
        r"""
        The parameters of the Blender camera (as given to Blender), e.g. to save the setup of a
        rendering.

        Returns
        -------
        Dict[str, Any]
            ``resolution``, ``pixel_aspect``, ``lens``, ``sensor_width``, ``sensor_height``,
            ``sensor_fit``, ``shift_x``, ``shift_y``, ``clip_start``, ``clip_end``, ``location``
            and ``quaternion``.

        """
        return {
            "type": "BlenderCamera",
            "resolution": [self.width, self.height],
            "pixel_aspect": list(self.pixel_aspect),
            "lens": self.lens,
            "sensor_width": self.sensor_width,
            "sensor_height": self.sensor_height,
            "sensor_fit": self.sensor_fit,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
            "clip_start": self._clip_start,
            "clip_end": self._clip_end,
            "location": self.location.tolist(),
            "quaternion": self.quaternion.tolist(),
        }
