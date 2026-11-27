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

import json
from numbers import Integral, Real
from typing import Any, Dict, Optional, Tuple

import numpy
from numpy.typing import ArrayLike

from pysdic import Camera, NoDistortion, PerspectiveExtrinsic, PinholeIntrinsic
from ._validation import _as_array, _validate_integer, _validate_real


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


def _check_rotation(rotation: numpy.ndarray) -> None:
    # A proper rotation matrix: orthonormal with a determinant +1
    error = numpy.abs(rotation @ rotation.T - numpy.eye(3)).max()
    if error > 1e-9:
        raise ValueError(f"Input 'rotation' must be an orthonormal matrix. Got |R R^T - I| = {error:.2e}.")
    if numpy.linalg.det(rotation) <= 0.0:
        raise ValueError("Input 'rotation' must be a proper rotation (determinant +1). Got a reflection.")


def _parse_intrinsic_matrix(matrix: ArrayLike) -> Tuple[float, float, float, float]:
    # fx, fy, cx, cy of an intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    K = _as_array(matrix, "intrinsic_matrix", numpy.float64, shape=(3, 3), finite=True)
    expected_zeros = (K[0, 1], K[1, 0], K[2, 0], K[2, 1])
    if any(value != 0.0 for value in expected_zeros) or K[2, 2] != 1.0:
        raise ValueError(
            "Input 'intrinsic_matrix' must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (no skew, not supported by "
            f"Blender). Got {K.tolist()}."
        )
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


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

    **Depth of field.** By default the camera is a pinhole (everything sharp). With an
    ``aperture_diameter`` :math:`D` and a ``focus_distance`` :math:`z_f`, Cycles renders a thin
    lens: the rays start on a disk of diameter :math:`D` centered on :attr:`center`, normal to
    the optical axis, and converge on the plane :math:`z_c = z_f`. The points of this plane are
    sharp; a point at the depth :math:`z_c` is spread on a disk of diameter (in pixels along
    :math:`x`)

    .. math::

        c = f_x \, D \, \left| \frac{1}{z_c} - \frac{1}{z_f} \right|

    centered on its pinhole projection: the projection model above stays exact for the center
    of the blur (no change of magnification with the focus). For a real lens of focal length
    :math:`f` and f-number :math:`N`, :math:`D = f / N` (e.g. 50 mm at f/8: :math:`D = 6.25`
    mm), in the unit of the scene. Blender computes the aperture from ``lens`` and its
    ``aperture_fstop`` assuming a scene in meters: :attr:`aperture_fstop` is computed so that the
    aperture is :math:`D` whatever the unit of the scene and :attr:`pixel_size`.

    **Export.** :meth:`to_dict` / :meth:`to_json` save the parameters of the camera
    (:math:`R`, :math:`\vec{t}`, :math:`K`, the resolution, the pixel size, the clipping
    planes); :meth:`from_dict` / :meth:`from_json` rebuild the Blender camera **without** the
    :class:`pysdic.Camera` (:attr:`camera` is then None).

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

    aperture_diameter : Optional[Real], optional
        The diameter :math:`D` of the aperture (entrance pupil) in the unit of the scene, for a
        depth of field. By default None: pinhole camera, no defocus blur. Requires
        ``focus_distance``.

    focus_distance : Optional[Real], optional
        The depth :math:`z_f` of the plane in focus (distance along the optical axis, in the
        unit of the scene), e.g. ``(R @ target + t)[2]`` for a point ``target`` of the world.
        Requires ``aperture_diameter``.

    Raises
    ------
    TypeError
        If ``camera`` is not a :class:`pysdic.Camera`, or its transformations are not of the
        types above.

    ValueError
        If a transformation does not follow its model on the test points, ``lens < 1`` mm, or
        only one of ``aperture_diameter`` and ``focus_distance`` is given.

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

        blender_camera.to_json("camera.json")
        same_camera = BlenderCamera.from_json("camera.json")  # same_camera.camera is None

    """

    __slots__ = (
        "_camera", "_width", "_height", "_rotation", "_translation", "_fx", "_fy", "_cx", "_cy",
        "_pixel_size", "_clip_start", "_clip_end", "_aperture_diameter", "_focus_distance",
    )

    def __init__(
        self,
        camera: Camera,
        *,
        pixel_size: Real = 0.01,
        clip_start: Real = 1e-3,
        clip_end: Real = 1e6,
        aperture_diameter: Optional[Real] = None,
        focus_distance: Optional[Real] = None,
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
        rotation, translation = _extract_pose(camera.extrinsic)
        _check_pose(camera.extrinsic, rotation, translation)
        fx, fy, cx, cy = _extract_intrinsic(camera.intrinsic)
        self._setup(
            camera=camera,
            width=int(camera.sensor_width),
            height=int(camera.sensor_height),
            rotation=rotation,
            translation=translation,
            intrinsic=(fx, fy, cx, cy),
            pixel_size=pixel_size,
            clip_start=clip_start,
            clip_end=clip_end,
            aperture_diameter=aperture_diameter,
            focus_distance=focus_distance,
        )

    def _setup(
        self,
        *,
        camera: Optional[Camera],
        width: Integral,
        height: Integral,
        rotation: ArrayLike,
        translation: ArrayLike,
        intrinsic: Tuple[float, float, float, float],
        pixel_size: Real,
        clip_start: Real,
        clip_end: Real,
        aperture_diameter: Optional[Real],
        focus_distance: Optional[Real],
    ) -> None:
        # Validate and store the parameters (shared by __init__ and from_dict)
        self._camera = camera
        self._width = _validate_integer(width, "width", min_value=1)
        self._height = _validate_integer(height, "height", min_value=1)
        self._rotation = _as_array(rotation, "rotation", numpy.float64, shape=(3, 3), finite=True).copy()
        _check_rotation(self._rotation)
        self._translation = _as_array(translation, "translation", numpy.float64, shape=(3,), finite=True).copy()
        fx, fy, cx, cy = intrinsic
        self._fx = _validate_real(fx, "fx", positive=True, finite=True)
        self._fy = _validate_real(fy, "fy", positive=True, finite=True)
        self._cx = _validate_real(cx, "cx", finite=True)
        self._cy = _validate_real(cy, "cy", finite=True)
        self._pixel_size = _validate_real(pixel_size, "pixel_size", positive=True, finite=True)
        self._clip_start = _validate_real(clip_start, "clip_start", positive=True, finite=True)
        self._clip_end = _validate_real(clip_end, "clip_end", positive=True, finite=True)
        if not self._clip_end > self._clip_start:
            raise ValueError(f"Input 'clip_end' must be greater than 'clip_start'. Got {clip_start} and {clip_end}.")
        if (aperture_diameter is None) != (focus_distance is None):
            raise ValueError(
                "Inputs 'aperture_diameter' and 'focus_distance' must be given together (depth of field) or both be "
                f"None (pinhole camera). Got {aperture_diameter} and {focus_distance}."
            )
        self._aperture_diameter = None
        self._focus_distance = None
        if aperture_diameter is not None:
            self._aperture_diameter = _validate_real(aperture_diameter, "aperture_diameter", positive=True, finite=True)
            self._focus_distance = _validate_real(focus_distance, "focus_distance", positive=True, finite=True)
        self._rotation.setflags(write=False)
        self._translation.setflags(write=False)
        if self.lens < 1.0:
            raise ValueError(
                f"The focal length of the Blender camera is {self.lens:.3f} mm, Blender requires at least 1 mm: "
                f"increase 'pixel_size' (got {self._pixel_size} mm)."
            )

    def __repr__(self) -> str:
        dof = "" if not self.use_dof else f", aperture={self._aperture_diameter}, focus={self._focus_distance}"
        return (
            f"BlenderCamera({self.width}x{self.height} pixels, lens={self.lens:.3f} mm, "
            f"sensor_fit={self.sensor_fit!r}, shift=({self.shift_x:.4f}, {self.shift_y:.4f}){dof})"
        )

    # ===================================================================
    # pysdic camera
    # ===================================================================
    @property
    def camera(self) -> Optional[Camera]:
        r"""
        The reproduced :class:`pysdic.Camera`, or None if the Blender camera was rebuilt with
        :meth:`from_dict` / :meth:`from_json` (only the parameters below are saved).
        """
        return self._camera

    @property
    def width(self) -> int:
        r"""
        The resolution :math:`W` of the image along :math:`x` (number of columns), in **pixels**.

        It is :attr:`pysdic.Camera.sensor_width` of the pysdic camera, given to Blender as
        ``render.resolution_x``. Not to be confused with :attr:`sensor_width` (millimeters).
        """
        return self._width

    @property
    def height(self) -> int:
        r"""
        The resolution :math:`H` of the image along :math:`y` (number of rows), in **pixels**.

        It is :attr:`pysdic.Camera.sensor_height` of the pysdic camera, given to Blender as
        ``render.resolution_y``. Not to be confused with :attr:`sensor_height` (millimeters).
        """
        return self._height

    @property
    def rotation(self) -> numpy.ndarray:
        r"""
        The rotation :math:`R` (3, 3) of the world-to-camera transformation
        :math:`\vec{X}_c = R \vec{X}_w + \vec{t}` (OpenCV convention, read-only).

        Its rows are the axes :math:`x_c` (to the right), :math:`y_c` (downwards) and
        :math:`z_c` (optical axis) of the camera expressed in the world. It is read from the rays
        of the extrinsic (independent of the storage of its parameters) and checked on test
        points.
        """
        return self._rotation

    @property
    def translation(self) -> numpy.ndarray:
        r"""
        The translation :math:`\vec{t}` (3,) of the world-to-camera transformation
        :math:`\vec{X}_c = R \vec{X}_w + \vec{t}` (OpenCV convention, read-only): the world
        origin in the frame of the camera. The position of the camera is :attr:`center`.
        """
        return self._translation

    @property
    def center(self) -> numpy.ndarray:
        r"""
        The optical center :math:`\vec{C} = -R^T \vec{t}` (3,) of the camera in the world, in
        the unit of the scene.
        """
        return -self._rotation.T @ self._translation

    @property
    def intrinsic_matrix(self) -> numpy.ndarray:
        r"""
        The intrinsic matrix :math:`K` (3, 3) of the pinhole model, in pixels:

        .. math::

            K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}

        with the pixel centers at integer coordinates (the center of the first pixel is
        :math:`(0, 0)`, the center of the image is :math:`((W - 1)/2, (H - 1)/2)`).
        """
        return numpy.array([[self._fx, 0.0, self._cx], [0.0, self._fy, self._cy], [0.0, 0.0, 1.0]])

    # ===================================================================
    # Blender parameters
    # ===================================================================
    @property
    def pixel_size(self) -> float:
        r"""
        The size :math:`p` of a pixel in millimeters, used to express :attr:`sensor_width`,
        :attr:`sensor_height` and :attr:`lens` in millimeters. It does not change the image
        (only the ratio ``lens / sensor`` matters), but must give ``lens >= 1`` mm.
        """
        return self._pixel_size

    @property
    def clip_start(self) -> float:
        r"""The distance of the near clipping plane, in the unit of the scene (nothing closer is rendered)."""
        return self._clip_start

    @property
    def clip_end(self) -> float:
        r"""The distance of the far clipping plane, in the unit of the scene (nothing farther is rendered)."""
        return self._clip_end

    @property
    def pixel_aspect(self) -> Tuple[float, float]:
        r"""
        The pixel aspect ratio :math:`(a_x, a_y) = (\max(1, f_y / f_x), \max(1, f_x / f_y))`
        given to Blender as ``render.pixel_aspect_x`` / ``pixel_aspect_y``.

        :math:`(1, 1)` for square pixels (:math:`f_x = f_y`). Blender requires values
        :math:`\geq 1`: only the largest ratio differs from 1.
        """
        return max(1.0, self._fy / self._fx), max(1.0, self._fx / self._fy)

    @property
    def sensor_fit(self) -> str:
        r"""
        The sensor fit of the Blender camera: the dimension of the image along which the sensor
        size (and ``lens``) is measured.

        ``"HORIZONTAL"`` if :math:`a_x W \geq a_y H` (the displayed image is wider than tall),
        else ``"VERTICAL"``. It sets which of :attr:`sensor_width` and :attr:`sensor_height` is
        used by :attr:`lens`, and the view factor of :attr:`shift_x` and :attr:`shift_y`.
        """
        a_x, a_y = self.pixel_aspect
        return "HORIZONTAL" if a_x * self.width >= a_y * self.height else "VERTICAL"

    @property
    def sensor_width(self) -> float:
        r"""
        The width of the Blender sensor :math:`p W` in **millimeters** (``camera.sensor_width``
        of Blender).

        .. warning::

            Not to be confused with :attr:`pysdic.Camera.sensor_width`, the width of the image in
            **pixels** (here :attr:`width`).
        """
        return self._pixel_size * self.width

    @property
    def sensor_height(self) -> float:
        r"""
        The height of the Blender sensor :math:`p H` in **millimeters** (``camera.sensor_height``
        of Blender).

        .. warning::

            Not to be confused with :attr:`pysdic.Camera.sensor_height`, the height of the image
            in **pixels** (here :attr:`height`).
        """
        return self._pixel_size * self.height

    @property
    def _view_factor(self) -> float:
        a_x, a_y = self.pixel_aspect
        return float(self.width) if self.sensor_fit == "HORIZONTAL" else (a_y / a_x) * self.height

    @property
    def lens(self) -> float:
        r"""
        The focal length ``lens`` of the Blender camera in millimeters.

        .. math::

            \text{lens} = \frac{f_x \, s}{v}

        with the sensor size :math:`s` (:attr:`sensor_width` if :attr:`sensor_fit` is
        ``"HORIZONTAL"``, else :attr:`sensor_height`) and the view factor :math:`v` (:math:`W`
        or :math:`(a_y / a_x) H`). As :math:`s = p W` (or :math:`p H`), ``lens`` is proportional
        to :attr:`pixel_size` while the image only depends on :math:`f_x` in pixels: the pixel
        size only has to give ``lens >= 1`` mm, as required by Blender.
        """
        sensor = self.sensor_width if self.sensor_fit == "HORIZONTAL" else self.sensor_height
        return self._fx * sensor / self._view_factor

    @property
    def shift_x(self) -> float:
        r"""
        The shift ``shift_x`` of the principal point of the Blender camera, as a fraction of the
        view factor :math:`v`:

        .. math::

            \text{shift}_x = -\frac{c_x - (W - 1)/2}{v}

        Blender places the principal point at the center of the image for a zero shift, i.e. at
        :math:`(W - 1)/2` in the pysdic convention of the pixels (centers at integer
        coordinates). A positive shift moves the image window to the right, so the principal
        point moves to the left: :math:`c_x < (W - 1)/2` gives :math:`\text{shift}_x > 0`.
        """
        return -(self._cx - (self.width - 1) / 2) / self._view_factor

    @property
    def shift_y(self) -> float:
        r"""
        The shift ``shift_y`` of the principal point of the Blender camera, as a fraction of the
        view factor :math:`v`:

        .. math::

            \text{shift}_y = \frac{(c_y - (H - 1)/2) \, a_y / a_x}{v}

        The sign is opposite to :attr:`shift_x` because the :math:`y` axis of Blender goes
        upwards while the rows of the image go downwards; the factor :math:`a_y / a_x` converts
        the rows into the displayed (square) units of Blender.
        """
        a_x, a_y = self.pixel_aspect
        return (self._cy - (self.height - 1) / 2) * (a_y / a_x) / self._view_factor

    @property
    def location(self) -> numpy.ndarray:
        r"""
        The ``location`` (3,) of the Blender camera object: its center
        :math:`\vec{C} = -R^T \vec{t}` (:attr:`center`, convention 0 camera-to-world).
        """
        return self.center

    @property
    def quaternion(self) -> numpy.ndarray:
        r"""
        The ``rotation_quaternion`` ``(w, x, y, z)`` (unit, Blender order) of the Blender camera
        object: the quaternion of :math:`R_b = R^T \, \text{diag}(1, -1, -1)`.

        :math:`R_b` is the camera-to-world rotation (convention 0) with the OpenGL axes of
        Blender: its columns are the axes :math:`x_c`, :math:`-y_c` and :math:`-z_c` of the
        OpenCV camera in the world (the Blender camera looks along its local :math:`-z`).
        """
        # Columns of R_b: the rows of R (transposed rotation) with the signs of y and z changed
        axes = numpy.stack((self._rotation[0], -self._rotation[1], -self._rotation[2]), axis=1)
        return _rotation_to_quaternion(axes)

    # ===================================================================
    # Depth of field
    # ===================================================================
    @property
    def use_dof(self) -> bool:
        r"""Whether the camera has a depth of field (thin lens), i.e. an :attr:`aperture_diameter`."""
        return self._aperture_diameter is not None

    @property
    def aperture_diameter(self) -> Optional[float]:
        r"""
        The diameter :math:`D` of the aperture in the unit of the scene, or None for a pinhole
        camera. For a lens of focal length :math:`f` and f-number :math:`N`, :math:`D = f / N`.
        """
        return self._aperture_diameter

    @property
    def focus_distance(self) -> Optional[float]:
        r"""
        The depth :math:`z_f` of the plane in focus (along the optical axis :math:`z_c`) in the
        unit of the scene, or None for a pinhole camera.
        """
        return self._focus_distance

    @property
    def aperture_fstop(self) -> Optional[float]:
        r"""
        The ``dof.aperture_fstop`` given to Blender, or None for a pinhole camera.

        Blender takes an aperture of diameter ``lens * 1e-3 / aperture_fstop`` in the unit of
        the scene (``lens`` in millimeters, the scene assumed in meters), so

        .. math::

            \text{aperture\_fstop} = \frac{\text{lens} \times 10^{-3}}{D}

        It is **not** the f-number of the real lens (it depends on :attr:`pixel_size` and on the
        unit of the scene): only :attr:`aperture_diameter` has a physical meaning here.
        """
        if self._aperture_diameter is None:
            return None
        return self.lens * 1e-3 / self._aperture_diameter

    def defocus_blur(self, depth: ArrayLike) -> numpy.ndarray:
        r"""
        The diameter in pixels (along :math:`x`) of the defocus blur of points at the depths
        ``depth`` (:math:`z_c > 0`, any shape): :math:`c = f_x D |1/z_c - 1/z_f|`, 0 for a
        pinhole camera. Useful to check the blur over a specimen seen with an inclination.
        """
        depth = numpy.asarray(depth, dtype=numpy.float64)
        if not numpy.all(numpy.isfinite(depth)) or numpy.any(depth <= 0.0):
            raise ValueError("Input 'depth' must contain finite positive depths.")
        if self._aperture_diameter is None:
            return numpy.zeros_like(depth)
        return self._fx * self._aperture_diameter * numpy.abs(1.0 / depth - 1.0 / self._focus_distance)

    # ===================================================================
    # Export
    # ===================================================================
    def blender_parameters(self) -> Dict[str, Any]:
        r"""
        The parameters as given to Blender: ``resolution``, ``pixel_aspect``, ``lens``,
        ``sensor_width``, ``sensor_height``, ``sensor_fit``, ``shift_x``, ``shift_y``,
        ``clip_start``, ``clip_end``, ``location``, ``quaternion``, ``use_dof`` and, with a
        depth of field, ``aperture_fstop`` and ``focus_distance``.
        """
        parameters = {
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
            "use_dof": self.use_dof,
        }
        if self.use_dof:
            parameters["aperture_fstop"] = self.aperture_fstop
            parameters["focus_distance"] = self._focus_distance
        return parameters

    def to_dict(self, description: Optional[str] = None) -> Dict[str, Any]:
        r"""
        Export the camera to a dictionary.

        The keys ``resolution`` :math:`(W, H)`, ``rotation`` :math:`R`, ``translation``
        :math:`\vec{t}`, ``intrinsic_matrix`` :math:`K`, ``pixel_size``, ``clip_start``,
        ``clip_end``, ``aperture_diameter`` and ``focus_distance`` (None for a pinhole camera)
        define the camera; ``blender`` holds the derived parameters given to
        Blender (see :meth:`blender_parameters`), for information.

        The :class:`pysdic.Camera` itself is not saved.
        """
        data: Dict[str, Any] = {"type": "BlenderCamera"}
        if description is not None:
            if not isinstance(description, str):
                raise TypeError(f"Input 'description' must be a string. Got type {type(description)}.")
            data["description"] = description
        data.update({
            "resolution": [self.width, self.height],
            "rotation": self._rotation.tolist(),
            "translation": self._translation.tolist(),
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "pixel_size": self._pixel_size,
            "clip_start": self._clip_start,
            "clip_end": self._clip_end,
            "aperture_diameter": self._aperture_diameter,
            "focus_distance": self._focus_distance,
            "blender": self.blender_parameters(),
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlenderCamera:
        r"""
        Create a Blender camera from a dictionary of :meth:`to_dict`, **without** the
        :class:`pysdic.Camera` (:attr:`camera` is None).

        ``resolution``, ``rotation``, ``translation`` and ``intrinsic_matrix`` are required;
        ``pixel_size``, ``clip_start``, ``clip_end``, ``aperture_diameter`` and
        ``focus_distance`` take the default values of the constructor if absent. ``type`` and ``description`` are ignored. The derived parameters
        ``blender`` are not read but checked: if present, they must match the parameters
        recomputed from the camera (a ``ValueError`` otherwise, e.g. after a manual edit of
        ``lens``). An unknown key raises a ``KeyError``.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        required = ("resolution", "rotation", "translation", "intrinsic_matrix")
        allowed = set(required) | {
            "type", "description", "pixel_size", "clip_start", "clip_end", "aperture_diameter", "focus_distance",
            "blender",
        }
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing fields of a BlenderCamera: {missing}.")
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise KeyError(f"Unknown fields of a BlenderCamera: {unknown}.")
        resolution = data["resolution"]
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError(f"Field 'resolution' must be [width, height]. Got {resolution!r}.")

        camera = cls.__new__(cls)
        camera._setup(
            camera=None,
            width=resolution[0],
            height=resolution[1],
            rotation=data["rotation"],
            translation=data["translation"],
            intrinsic=_parse_intrinsic_matrix(data["intrinsic_matrix"]),
            pixel_size=data.get("pixel_size", 0.01),
            clip_start=data.get("clip_start", 1e-3),
            clip_end=data.get("clip_end", 1e6),
            aperture_diameter=data.get("aperture_diameter"),
            focus_distance=data.get("focus_distance"),
        )

        # The derived parameters must match the camera (no silent ignore of an edited value)
        saved = data.get("blender")
        if saved is not None:
            if not isinstance(saved, dict):
                raise TypeError(f"Field 'blender' must be a dictionary. Got type {type(saved)}.")
            computed = camera.blender_parameters()
            unknown = sorted(set(saved) - set(computed))
            if unknown:
                raise KeyError(f"Unknown fields in 'blender' of a BlenderCamera: {unknown}.")
            for key, value in saved.items():
                expected = computed[key]
                if key in ("sensor_fit", "use_dof"):
                    same = value == expected
                elif key == "quaternion":  # q and -q are the same rotation
                    q = numpy.asarray(value, dtype=numpy.float64)
                    same = q.shape == (4,) and (
                        numpy.allclose(q, expected, rtol=1e-9, atol=1e-12)
                        or numpy.allclose(-q, expected, rtol=1e-9, atol=1e-12)
                    )
                else:
                    v = numpy.asarray(value, dtype=numpy.float64)
                    same = v.shape == numpy.shape(expected) and numpy.allclose(v, expected, rtol=1e-9, atol=1e-12)
                if not same:
                    raise ValueError(
                        f"The saved Blender parameter {key!r} ({value!r}) does not match the camera ({expected!r}): "
                        "the parameters of 'blender' are derived from the camera and cannot be edited, change "
                        "'rotation', 'translation', 'intrinsic_matrix', 'pixel_size' or the clipping planes instead."
                    )
        return camera

    def to_json(self, path: str, description: Optional[str] = None) -> None:
        r"""Export the camera to a JSON file (see :meth:`to_dict`)."""
        with open(path, "w") as file:
            json.dump(self.to_dict(description), file, indent=4)

    @classmethod
    def from_json(cls, path: str) -> BlenderCamera:
        r"""Create a Blender camera from a JSON file (see :meth:`from_dict`), without the :class:`pysdic.Camera`."""
        with open(path, "r") as file:
            return cls.from_dict(json.load(file))