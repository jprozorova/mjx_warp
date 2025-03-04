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

import mujoco
import warp as wp
from .types import Model
from .types import Data
from .types import array2df


def is_sparse(m: mujoco.MjModel):
  if m.opt.jacobian == mujoco.mjtJacobian.mjJAC_AUTO:
    return m.nv >= 60
  return m.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE


@wp.kernel
def process_level(
  body_tree: wp.array(ndim=1, dtype=int),
  body_parentid: wp.array(ndim=1, dtype=int),
  dof_bodyid: wp.array(ndim=1, dtype=int),
  mask: wp.array2d(dtype=wp.bool),
  beg: int,
):
  dofid, tid_y = wp.tid()
  j = beg + tid_y
  el = body_tree[j]
  parent_id = body_parentid[el]
  parent_val = mask[dofid, parent_id]
  mask[dofid, el] = parent_val or (dof_bodyid[dofid] == el)


@wp.kernel
def compute_qfrc(
  d: Data,
  m: Model,
  mask: wp.array2d(dtype=wp.bool),
  qfrc_total: array2df,
):
  worldid, dofid = wp.tid()
  accumul = float(0.0)
  cdof_vec = d.cdof[worldid, dofid]
  rotational_cdof = wp.vec3(cdof_vec[0], cdof_vec[1], cdof_vec[2])

  jac = wp.spatial_vector(
    cdof_vec[3], cdof_vec[4], cdof_vec[5], cdof_vec[0], cdof_vec[1], cdof_vec[2]
  )

  for bodyid in range(m.nbody):
    if mask[dofid, bodyid]:
      offset = d.xipos[worldid, bodyid] - d.subtree_com[worldid, m.body_rootid[bodyid]]
      cross_term = wp.cross(rotational_cdof, offset)
      accumul += wp.dot(jac, d.xfrc_applied[worldid, bodyid]) + wp.dot(
        cross_term,
        wp.vec3(
          d.xfrc_applied[worldid, bodyid][0],
          d.xfrc_applied[worldid, bodyid][1],
          d.xfrc_applied[worldid, bodyid][2],
        ),
      )

  qfrc_total[worldid, dofid] = accumul


def xfrc_accumulate(m: Model, d: Data) -> array2df:
  body_treeadr_np = m.body_treeadr.numpy()
  mask = wp.zeros((m.nv, m.nbody), dtype=wp.bool)

  for i in range(len(body_treeadr_np)):
    beg = body_treeadr_np[i]
    end = m.nbody if i == len(body_treeadr_np) - 1 else body_treeadr_np[i + 1]

    if end > beg:
      wp.launch(
        kernel=process_level,
        dim=[m.nv, (end - beg)],
        inputs=[m.body_tree, m.body_parentid, m.dof_bodyid, mask, beg],
      )

  qfrc_total = wp.zeros((d.nworld, m.nv), dtype=float)

  wp.launch(kernel=compute_qfrc, dim=(d.nworld, m.nv), inputs=[d, m, mask, qfrc_total])

  return qfrc_total
