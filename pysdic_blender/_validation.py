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

# Validation of the inputs (internal): pysdic-blender does not use the private helpers of pysdic.

from numbers import Integral, Real
from typing import Any, Optional, Sequence, Tuple

import numpy


def _as_array(
    value: Any, name: str, dtype: Any, *, shape: Optional[Tuple[Optional[int], ...]] = None, finite: bool = False
) -> numpy.ndarray:
    # Array of a dtype, with an optional shape (None: any size on an axis) and finite values
    try:
        array = numpy.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Input '{name}' must be convertible to an array of {numpy.dtype(dtype)}.") from error
    if shape is not None:
        if array.ndim != len(shape) or any(s is not None and s != n for s, n in zip(shape, array.shape)):
            expected = tuple("N" if s is None else s for s in shape)
            raise ValueError(f"Input '{name}' must have the shape {expected}. Got {array.shape}.")
    if finite and numpy.issubdtype(array.dtype, numpy.floating) and not numpy.all(numpy.isfinite(array)):
        raise ValueError(f"Input '{name}' must have finite values.")
    return array


def _validate_real(
    value: Any, name: str, *, positive: bool = False, non_negative: bool = False, finite: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Input '{name}' must be a real number. Got type {type(value)}.")
    value = float(value)
    if finite and not numpy.isfinite(value):
        raise ValueError(f"Input '{name}' must be finite. Got {value}.")
    if positive and not value > 0.0:
        raise ValueError(f"Input '{name}' must be strictly positive. Got {value}.")
    if non_negative and not value >= 0.0:
        raise ValueError(f"Input '{name}' must be non-negative. Got {value}.")
    return value


def _validate_integer(value: Any, name: str, *, min_value: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Input '{name}' must be an integer. Got type {type(value)}.")
    value = int(value)
    if min_value is not None and value < min_value:
        raise ValueError(f"Input '{name}' must be at least {min_value}. Got {value}.")
    return value


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, numpy.bool_)):
        raise TypeError(f"Input '{name}' must be a boolean. Got type {type(value)}.")
    return bool(value)


def _validate_choice(value: Any, name: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise ValueError(f"Input '{name}' must be one of {tuple(choices)}. Got {value!r}.")
    return value
