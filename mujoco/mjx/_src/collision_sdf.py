# Copyright 2025 The Physics-Next Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import warp as wp
from .types import GeomType


@wp.struct
class GradientState:
  dist: float
  x: wp.vec3


@wp.func
def sphere(p: wp.vec3, size: wp.vec3) -> float:
  return wp.length(p) - size[0]


@wp.func
def ellipsoid(p: wp.vec3, size: wp.vec3) -> float:
  scaled_p = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])
  k0 = wp.length(scaled_p)
  k1 = wp.length(
    wp.vec3(p[0] / (size[0] ** 2.0), p[1] / (size[1] ** 2.0), p[2] / (size[2] ** 2.0))
  )
  if k1 != 0.0:
    denom = k1
  else:
    denom = 1e-12
  return k0 * (k0 - 1.0) / denom


@wp.func
def grad_sphere(p: wp.vec3) -> wp.vec3:
  c = wp.length(p)
  if c > 1e-9:
    return p / c
  else:
    wp.vec3(0.0)


@wp.func
def grad_ellipsoid(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  a = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])

  b = wp.vec3(a[0] / size[0], a[1] / size[1], a[2] / size[2])
  k0 = wp.length(a)
  k1 = wp.length(b)
  invK0 = 1.0 / k0
  invK1 = 1.0 / k1

  gk0 = b * invK0
  gk1 = wp.vec3(
    b[0] * invK1 / (size[0] * size[0]),
    b[1] * invK1 / (size[1] * size[1]),
    b[2] * invK1 / (size[2] * size[2]),
  )
  df_dk0 = (2.0 * k0 - 1.0) * invK1
  df_dk1 = k0 * (k0 - 1.0) * invK1 * invK1

  raw_grad = gk0 * df_dk0 - gk1 * df_dk1
  return raw_grad / wp.length(raw_grad)


def compute_grad(type1: int, type2: int):
  @wp.func
  def _compute_grad(
    p1: wp.vec3, p2: wp.vec3, s1: wp.vec3, s2: wp.vec3, rel_mat: wp.mat33
  ) -> wp.vec3:
    if wp.static(type1 == GeomType.SPHERE.value) and wp.static(
      type2 == GeomType.ELLIPSOID.value
    ):
      A = sphere(p1, s1)
      B = ellipsoid(p2, s2)

      grad1 = grad_sphere(p1)
      grad2 = grad_ellipsoid(p2, s2)

      grad1_transformed = rel_mat * grad1
      gradient = grad2 + grad1_transformed
      max_val = wp.max(A, B)
      if A > B:
        max_grad = grad1_transformed
      else:
        max_grad = grad2
      sign = wp.sign(max_val)
      gradient += max_grad * sign
      return gradient
    else:
      return wp.vec3(0.0, 0.0, 0.0)

  return _compute_grad


def clearance(type1: int, type2: int):
  @wp.func
  def _clearance(p1: wp.vec3, p2: wp.vec3, s1: wp.vec3, s2: wp.vec3) -> float:
    if wp.static(type1 == GeomType.SPHERE.value) and wp.static(
      type2 == GeomType.ELLIPSOID.value
    ):
      s = sphere(p1, s1)
      e = ellipsoid(p2, s2)
      return s + e + wp.abs(wp.max(s, e))
    else:
      return 0.0

  return _clearance


@wp.struct
class OptimizationParams:
  rel_mat: wp.mat33
  rel_pos: wp.vec3
  size1: wp.vec3
  size2: wp.vec3


def gradient_step(type1: int, type2: int):
  @wp.func
  def _gradient_step(
    state: GradientState,
    params: OptimizationParams,
  ) -> GradientState:
    amin = 1e-4
    amax = 2.0
    nlinesearch = 10

    x1 = params.rel_mat * state.x + params.rel_pos
    grad = wp.static(compute_grad(type1, type2))(
      x1, state.x, params.size1, params.size2, params.rel_mat
    )
    best_value = float(1e18)
    best_candidate = wp.vec3(0.0, 0.0, 0.0)

    for i in range(nlinesearch):
      t = float(i) / float(nlinesearch - 1)
      alpha = wp.exp(wp.log(amin) * (1.0 - t) + wp.log(amax) * t)
      candidate = state.x - grad * alpha
      x1 = params.rel_mat * candidate + params.rel_pos
      value = wp.static(clearance(type1, type2))(
        x1, candidate, params.size1, params.size2
      )

      if value < best_value:
        best_value = value
        best_candidate = candidate

    return GradientState(best_value, best_candidate)

  return _gradient_step


def gradient_descent(type1: int, type2: int):
  @wp.func
  def _gradient_descent(
    x: wp.vec3,
    niter: int,
    params: OptimizationParams,
  ):
    state = GradientState(1e10, x)

    for _ in range(niter):
      state = wp.static(gradient_step(type1, type2))(state, params)

    return state.dist, state.x

  return _gradient_descent
