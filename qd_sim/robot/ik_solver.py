"""
Damped-least-squares differential IK for the 6-DOF arm, driving a chosen
site (normally "gripper/pinch") toward a target Cartesian pose.

This is deliberately a velocity-level (resolved-rate) solver rather than a
full-pose IK solve to convergence in one shot: it's called every control
tick with the current pose error, which also makes it a natural fit for
tracking a moving target - the screw controller later adds a coupled
rotate+translate twist on top of this same per-tick velocity command
instead of needing a different solve method.
"""

import mujoco
import numpy as np

from qd_sim.robot.kinematics import arm_dof_indices, pose_error_twist, site_pose


class DampedLeastSquaresIK:
    def __init__(self, model, site_name, damping=0.05, max_joint_speed=4.0):
        self.model = model
        self.site_id = model.site(site_name).id
        self.dof_idx = arm_dof_indices(model)
        self.damping = damping
        # rad/s, clamps a single step. The QD is carried by a kinematic
        # weld rather than friction (see qd_sim/tasks/kinematic_weld.py),
        # so there's no risk of shaking it loose at higher speed - this cap
        # mainly exists to keep the arm's reactive tracking (vision-guided
        # descent, screwing) from moving in unrealistically large jumps.
        self.max_joint_speed = max_joint_speed

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def compute_joint_velocities(self, data, target_pos, target_quat, twist_extra=None):
        """Returns a 6-vector of arm joint velocities (rad/s) that drives
        the site toward (target_pos, target_quat).

        twist_extra: optional additional 6D world-frame twist to add on top
        of the pose-correction twist - used by the screw controller to
        superimpose the helical advance on top of ordinary position
        tracking, instead of needing a separate control path.
        """
        cur_pos, cur_quat = site_pose(data, self.site_id)
        twist = pose_error_twist(cur_pos, cur_quat, target_pos, target_quat)
        if twist_extra is not None:
            twist = twist + np.asarray(twist_extra)

        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.site_id)
        J_full = np.vstack([self._jacp, self._jacr])  # (6, nv)
        J = J_full[:, self.dof_idx]  # (6, 6) - arm columns only

        # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 twist
        lam2 = self.damping ** 2
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(6), twist)

        speed = np.max(np.abs(dq))
        if speed > self.max_joint_speed:
            dq *= self.max_joint_speed / speed
        return dq
