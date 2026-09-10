"""
Position-tracking controller for the 6-DOF arm: holds a target Cartesian
pose for the gripper's pinch site, and each tick converts the current pose
error into a joint velocity (via IK) that's integrated into a joint-position
command sent to the arm's actuators.

The UR5e's actuators are position servos (ctrl = desired joint angle, see
ur5e.xml's <general biastype="affine">), so "control" here means maintaining
our own running qpos target and writing it to ctrl every tick - not sending
raw torques or velocities directly.
"""

import numpy as np

from qd_sim.robot.ik_solver import DampedLeastSquaresIK
from qd_sim.robot.kinematics import ARM_JOINT_NAMES, arm_qpos_indices, site_pose
from qd_sim.utils import transforms as tf


def _smootherstep(t):
    """Perlin's quintic ease: zero velocity AND zero acceleration at both
    t=0 and t=1 (smoothstep's cubic only zeroes velocity) - used so a
    planned trajectory starts and ends at rest with no jerk at the handoff."""
    return t * t * t * (t * (t * 6 - 15) + 10)


# Default cruise pace for set_smooth_target_paced() - one place every caller
# (record_full_session.py, PickSequence, TransportSequence) shares, so every
# move is paced consistently rather than each picking its own.
DEFAULT_CRUISE_SPEED_MPS = 0.30
DEFAULT_CRUISE_ANGULAR_SPEED_DEG_S = 260.0
DEFAULT_MIN_MOVE_STEPS = 60


def _catmull_rom(p0, p1, p2, p3, t):
    """Position at parameter t in [0,1] between control points p1 and p2,
    given neighbors p0/p3 - the standard uniform Catmull-Rom formula.
    Continuous in both position and tangent at every control point, which
    is what makes the path curve through a waypoint instead of cornering."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


class ArmController:
    def __init__(self, model, site_name="gripper/pinch", max_joint_accel=14.0):
        self.model = model
        self.site_id = model.site(site_name).id
        self.qpos_idx = arm_qpos_indices(model)
        self.actuator_idx = [model.actuator(n.replace("_joint", "")).id for n in ARM_JOINT_NAMES]
        self.ik = DampedLeastSquaresIK(model, site_name)
        # rad/s^2, caps how fast commanded joint velocity can change per
        # tick - the IK's own max_joint_speed clamps speed but not how
        # quickly it's reached, so a fresh set_target_pose() would otherwise
        # snap dq straight to max speed on the first tick. This ramps it
        # instead.
        self.max_joint_accel = max_joint_accel

        self.target_pos = None
        self.target_quat = None
        self._target_qpos = None  # our running joint-position command
        self._prev_dq = None
        self._traj = None  # active smooth Cartesian trajectory, if any - see set_smooth_path()
        self._last_pos = None       # site position as of the previous step() call
        self._cur_velocity = np.zeros(3)  # finite-difference estimate of actual Cartesian
                                            # velocity, updated every step() - used as a fresh
                                            # trajectory's start tangent (set_smooth_path())
        # Which of the 6 arm joints (by ARM_JOINT_NAMES index) are locked -
        # their commanded velocity is forced to zero every step() regardless
        # of what the IK solves for. Used to hard-guarantee wrist_3 can't
        # move during screwing (qd_sim/control/screw_controller.py), on top
        # of already feeding the IK a non-fighting target.
        self.locked_dof = np.zeros(len(self.actuator_idx), dtype=bool)

    def lock_dof(self, index):
        self.locked_dof[index] = True

    def unlock_dof(self, index):
        self.locked_dof[index] = False

    def reset(self, data):
        """Snap the internal target to the arm's current actual joint
        position - call once after any qpos reset (e.g. a keyframe load) so
        the first control tick doesn't jump."""
        self._target_qpos = data.qpos[self.qpos_idx].copy()
        cur_pos, cur_quat = site_pose(data, self.site_id)
        self.target_pos, self.target_quat = cur_pos, cur_quat
        self._prev_dq = np.zeros(6)
        self._traj = None
        self._last_pos = cur_pos
        self._cur_velocity = np.zeros(3)

    def set_target_pose(self, pos, quat):
        """Set a single fixed target that the arm chases reactively every
        tick via velocity IK - the path taken is whatever the IK's per-tick
        solution produces, not a planned curve. Cancels any in-progress
        smooth trajectory."""
        self.target_pos = np.asarray(pos, dtype=float)
        self.target_quat = np.asarray(quat, dtype=float)
        self._traj = None

    def set_smooth_target(self, data, pos, quat, n_steps):
        """Plan a smooth Cartesian trajectory from the arm's current pose to
        (pos, quat) - the single-destination case of set_smooth_path()
        below; see that docstring for the full reasoning."""
        self.set_smooth_path(data, [(pos, quat)], n_steps)

    def set_smooth_target_paced(self, data, pos, quat, dt, cruise_speed_mps=DEFAULT_CRUISE_SPEED_MPS,
                                 cruise_angular_speed_deg_s=DEFAULT_CRUISE_ANGULAR_SPEED_DEG_S,
                                 min_steps=DEFAULT_MIN_MOVE_STEPS):
        """set_smooth_target(), but picking n_steps automatically from the
        distance/angle to cover and a cruise pace, instead of the caller
        computing it. One pacing formula, one set of default cruise
        constants, so every caller moves at a comparable pace."""
        pos, quat = np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
        cur_pos, cur_quat = site_pose(data, self.site_id)
        dist_m, angle_deg = tf.pose_error(cur_pos, cur_quat, pos, quat)
        duration_s = max(dist_m / cruise_speed_mps, angle_deg / cruise_angular_speed_deg_s)
        n_steps = max(min_steps, int(duration_s / dt))
        self.set_smooth_target(data, pos, quat, n_steps)

    def set_smooth_path(self, data, waypoints, n_steps):
        """Plan a smooth Cartesian trajectory from the arm's current pose
        through an ordered sequence of via-points to a final destination
        (waypoints = [(pos, quat), ...], last entry is the destination),
        and start tracking it - each step() call advances one sample along
        this pre-planned path and uses that as target_pos/target_quat,
        instead of jumping straight to a fixed point and leaving the
        reactive IK to chase it from far away.

        One continuous smootherstep time profile governs progress along the
        whole path, not one profile per leg, so intermediate via-points
        (e.g. fly_safe()'s rise-then-translate-then-descend) are passed
        through at cruising speed with no stop in between, while still
        being visited in order. Only the final destination gets the
        profile's zero-velocity ending.

        Position follows a Catmull-Rom spline through the waypoints rather
        than a piecewise-straight-line path, so it curves smoothly through
        each via-point instead of cornering (see _catmull_rom()). The
        spline's start tangent is derived from _cur_velocity - the arm's
        actual measured Cartesian velocity - so a new trajectory continues
        in whatever direction the arm is already moving, rather than
        implying a sudden direction change at the handoff. Orientation
        still slerps linearly per leg (transforms.slerp()) since
        orientation changes here are small/rare enough not to need the
        spline treatment.

        This also avoids the per-axis wobble a plain set_target_pose() move
        can show: since the reference itself follows an explicit path and
        never backtracks, there's nothing non-monotonic in the IK's
        multi-DOF resolution for the arm's own tracking error to inherit.

        n_steps is in ticks, not seconds - the caller picks it from
        whatever cruise speed/dt it's using (see set_smooth_target_paced()).
        is_converged() won't report done while a trajectory is active,
        regardless of momentary position-matching early in the ramp."""
        # A re-plan (a path is already active) starts from the current
        # reference point, not the arm's measured pose - the reference
        # legitimately leads the arm by however much the controller is
        # lagging, and restarting from the measured pose would throw that
        # lead away on every re-plan.
        if self._traj is not None:
            cur_pos, cur_quat = np.asarray(self.target_pos, dtype=float), np.asarray(self.target_quat, dtype=float)
        else:
            cur_pos, cur_quat = site_pose(data, self.site_id)
        positions = [cur_pos] + [np.asarray(p, dtype=float) for p, _ in waypoints]
        quats = [cur_quat] + [np.asarray(q, dtype=float) for _, q in waypoints]
        seg_lengths = [float(np.linalg.norm(positions[i + 1] - positions[i]))
                       for i in range(len(positions) - 1)]
        total = sum(seg_lengths) or 1e-9  # all-zero-length edge case (pure reorientation)
        # A phantom point behind the start, placed so the spline's initial
        # tangent matches the arm's actual current velocity direction, at
        # the first leg's own scale rather than an arbitrary distance.
        # Falls back to duplicating the start point when the arm is at rest.
        speed = float(np.linalg.norm(self._cur_velocity))
        if speed > 1e-6:
            phantom_start = cur_pos - (self._cur_velocity / speed) * seg_lengths[0]
        else:
            phantom_start = cur_pos
        self._traj = {
            "positions": positions, "quats": quats, "seg_lengths": seg_lengths, "total": total,
            "phantom_start": phantom_start, "n": max(1, int(n_steps)), "i": 0,
        }

    def _sample_path(self, s):
        """Position/orientation at fractional progress s in [0,1] along the
        active trajectory's Catmull-Rom spline."""
        traj = self._traj
        traveled = s * traj["total"]
        acc = 0.0
        positions, quats, seg_lengths = traj["positions"], traj["quats"], traj["seg_lengths"]
        n_segs = len(seg_lengths)
        for i, seg_len in enumerate(seg_lengths):
            is_last = i == n_segs - 1
            if traveled <= acc + seg_len or is_last:
                frac = 0.0 if seg_len < 1e-9 else np.clip((traveled - acc) / seg_len, 0.0, 1.0)
                p0 = positions[i - 1] if i > 0 else traj["phantom_start"]
                p1, p2 = positions[i], positions[i + 1]
                p3 = positions[i + 2] if i + 2 < len(positions) else positions[-1]
                pos = _catmull_rom(p0, p1, p2, p3, frac)
                quat = tf.slerp(quats[i], quats[i + 1], frac)
                return pos, quat
            acc += seg_len
        return positions[-1], quats[-1]

    def hold_here(self, data):
        """Freeze the target at the arm's current actual pose and clear any
        residual commanded joint velocity - unlike set_target_pose() alone,
        which leaves _prev_dq untouched so a target change ramps in from
        whatever velocity was already commanded (deliberate, for genuine
        moves to a new distant target, but wrong for "stop exactly here":
        zero pose error means near-zero desired velocity, but the accel-
        limited chase still takes several ticks to bring a nonzero prev_dq
        to zero, so the target keeps coasting past the freeze point before
        correcting back. This fixed a visible dip-then-recover right as
        screwing starts, when transport's final descend could still have
        residual downward velocity at the instant its convergence check
        passed."""
        cur_pos, cur_quat = site_pose(data, self.site_id)
        self.target_pos, self.target_quat = cur_pos, cur_quat
        self._prev_dq = np.zeros(6)
        self._traj = None
        self._cur_velocity = np.zeros(3)

    def step(self, data, dt, twist_extra=None, extra_quat_correction=None):
        """Advance the joint-position command by one control tick and write
        it to data.ctrl. Must be called after reset().

        extra_quat_correction: an optional 3x3 world-frame rotation matrix,
        composed onto target_quat after trajectory sampling but before the
        IK call. Exists because the screw_adapter joint (between the wrist
        and gripper) can rotate the same site this controller tracks, out
        from under it, without this controller's own trajectory/IK knowing.
        During actual screwing ScrewController.tick() handles this directly
        by shifting target_quat itself; this generalizes the same idea to a
        caller whose target isn't fixed (a moving flight trajectory)."""
        if self._target_qpos is None:
            raise RuntimeError("ArmController.reset(data) must be called before step()")

        # Finite-difference estimate of actual Cartesian velocity, used by
        # set_smooth_path() as a new trajectory's start tangent.
        cur_pos, _ = site_pose(data, self.site_id)
        if self._last_pos is not None:
            self._cur_velocity = (cur_pos - self._last_pos) / dt
        self._last_pos = cur_pos

        if self._traj is not None:
            traj = self._traj
            s = _smootherstep(min(traj["i"] / traj["n"], 1.0))
            self.target_pos, self.target_quat = self._sample_path(s)
            traj["i"] += 1
            if traj["i"] > traj["n"]:
                self._traj = None  # trajectory consumed - target_pos/quat now sit exactly at the destination

        if extra_quat_correction is not None:
            self.target_quat = tf.mat_to_quat(extra_quat_correction @ tf.quat_to_mat(self.target_quat))

        dq_desired = self.ik.compute_joint_velocities(data, self.target_pos, self.target_quat, twist_extra)
        dq_desired = np.where(self.locked_dof, 0.0, dq_desired)
        max_delta = self.max_joint_accel * dt
        dq = self._prev_dq + np.clip(dq_desired - self._prev_dq, -max_delta, max_delta)
        self._prev_dq = dq

        self._target_qpos = self._target_qpos + dq * dt
        data.ctrl[self.actuator_idx] = self._target_qpos

    def pose_error(self, data):
        """(position error in m, angle error in deg) between the site's
        current actual pose and the target - for convergence checks."""
        cur_pos, cur_quat = site_pose(data, self.site_id)
        return tf.pose_error(cur_pos, cur_quat, self.target_pos, self.target_quat)

    def is_converged(self, data, pos_tol_m=0.003, angle_tol_deg=2.0):
        """True once the arm has actually arrived at target_pos/target_quat.
        While a smooth trajectory is active, this always returns False
        regardless of momentary position-matching - target_pos is a moving
        intermediate sample during a trajectory, and early in the ramp it
        sits close to the arm's starting pose, which would otherwise report
        a false "converged" before the arm has gone anywhere. Because this
        also waits for the trajectory's own decelerate-to-rest ending, a
        caller using it as a completion signal always gets a full stop -
        correct when a stop is actually wanted, wrong when the caller wants
        to hand off while still moving (see is_near_final())."""
        if self._traj is not None:
            return False
        pos_err, angle_err = self.pose_error(data)
        return pos_err < pos_tol_m and angle_err < angle_tol_deg

    def final_pose_error(self, data):
        """Like pose_error(), but against the trajectory's ultimate
        destination even mid-trajectory, not the current moving
        intermediate sample - for callers that want "are we basically
        there" without waiting for the decelerate-to-rest ending. Falls
        back to plain target_pos/target_quat when no trajectory is active."""
        cur_pos, cur_quat = site_pose(data, self.site_id)
        if self._traj is not None:
            final_pos, final_quat = self._traj["positions"][-1], self._traj["quats"][-1]
        else:
            final_pos, final_quat = self.target_pos, self.target_quat
        return tf.pose_error(cur_pos, cur_quat, final_pos, final_quat)

    def is_near_final(self, data, pos_tol_m=0.03, angle_tol_deg=8.0):
        """True once the arm is close to its trajectory's ultimate
        destination, whether or not the trajectory has finished decelerating
        to rest - so a caller flying to a coarse hover (or any other cruise-
        through handoff) can hand off to whatever tracks/moves next while
        still moving, instead of waiting for a full stop it doesn't need.
        Safe to hand off mid-trajectory: switching targets never resets
        _prev_dq (only hold_here() does that, deliberately, for the cases
        that do need a real stop), so whatever velocity the arm already has
        carries smoothly into the next target."""
        pos_err, angle_err = self.final_pose_error(data)
        return pos_err < pos_tol_m and angle_err < angle_tol_deg
