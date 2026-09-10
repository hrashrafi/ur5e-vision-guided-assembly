"""
Kinematic thread-advance screwing motion: rotates a dedicated rotary tool
adapter (see build_ur5e_robotiq.py's "screw_adapter" joint) - and, via the
kinematic weld, the QD mounted through it - while the arm descends at the
exact helical relationship v = pitch/(2*pi) * omega, for the QD's measured
engagement depth (12.0mm). This is feed-forward: the helix itself is the
control law, not something discovered by a force loop. There's no force/
torque-based fault detection because the manifold's collision is a
deliberately open slot and the QD is a kinematically-welded mocap body,
which can't generate or resist physical force anyway. "Seated" means the
real measured depth was reached, not a fixed commanded step count.

Thread pitch: 7/8"-14 UNF (ISO 11926-3 / SAE J1926 ORB) per the connector's
datasheet - 25.4mm/14 = 1.8143mm/turn, ~6.61 turns over 12.0mm.

Why a dedicated joint instead of driving the arm's own wrist_3 directly:
wrist_3_joint has a hard +-2*pi limit (real UR5e hardware), nowhere near 12
turns of travel, and commanding rotation about the tool axis while pointing
straight down sits right at this arm's wrist singularity anyway (wrist_1/
wrist_3 co-axial) - even a small roll correction there measured as tens of
degrees of error that never converged. screw_adapter sidesteps both: no
range limit, and the arm's own 6 joints never need to reorient for
screwing. The kinematic weld just reads the pinch site's actual world pose
each tick, so it picks up this joint's rotation for free.

The pinch site is still downstream of screw_adapter, though, so its actual
orientation keeps rotating as screw_adapter turns even though the arm's own
joints aren't. A fixed target_quat would make the IK fight that rotation as
a growing error - measured as wrist_3 spinning an extra 341deg over one
screw phase purely from fighting screw_adapter. Computing target_quat
analytically each tick to track the intended rotation (see _advance()
below) avoids that and leaves much less residual tilt on the QD.

Rotation speed is ramped rather than commanded at full speed immediately -
screw_adapter is a real actuator with damping/armature, and snapping to
full angular velocity produces a visible reaction-torque jolt at the start
and end of screwing. Descent speed is derived from the same ramped angular
speed so the helix's pitch relationship holds exactly even mid-ramp.
"""

import numpy as np

from qd_sim.robot.kinematics import site_pose
from qd_sim.utils import transforms as tf


class ScrewController:
    def __init__(self, arm_ctrl, model, engagement_depth_m, expected_turns,
                 angular_speed_rad_s=2 * np.pi, angular_accel_rad_s2=4 * np.pi, reverse=False,
                 unwind_speed_rad_s=None, unwind_accel_rad_s2=None):
        self.arm = arm_ctrl
        self.pitch_m = engagement_depth_m / expected_turns
        self.omega_mag = abs(angular_speed_rad_s)
        self.angular_accel = angular_accel_rad_s2
        # Unwind (see unwind_tick() below) is a purely cosmetic derotation
        # after the QD is already released, so it isn't bound to the real
        # screwing speed - both default to the screw values but a caller
        # can pass faster ones. In practice the acceleration matters more
        # than the speed ceiling: unwind's own remaining angle is usually
        # under a full turn, so it's a pure accel-limited ramp that rarely
        # reaches the speed ceiling at all.
        self.unwind_omega_mag = abs(unwind_speed_rad_s) if unwind_speed_rad_s is not None else self.omega_mag
        self.unwind_accel = abs(unwind_accel_rad_s2) if unwind_accel_rad_s2 is not None else self.angular_accel
        self.sign = -1 if reverse else 1  # which way the tool visually spins
        self.screw_qpos_adr = model.joint("screw_adapter_joint").qposadr[0]
        self.screw_actuator_id = model.actuator("screw_adapter").id
        self._current_omega_mag = 0.0

    def start(self, data):
        # hold_here(), not set_target_pose(): also clears residual commanded
        # joint velocity, not just the target position - see its docstring
        # for the dip-then-recover this avoids when handing off from a
        # still-descending transport phase.
        self.arm.hold_here(data)
        _, quat = site_pose(data, self.arm.site_id)
        self._initial_target_quat = quat.copy()
        self._screw_target = data.qpos[self.screw_qpos_adr]
        self._start_angle = self._screw_target

    def finish(self):
        self._current_omega_mag = 0.0

    def begin_stop(self, data):
        """Call exactly once, right after tick()'s real-depth condition
        fires and before the decelerate_tick() loop - resyncs
        arm.target_pos to the arm's actual current position rather than the
        commanded Z target, which normally runs a small steady-state lag
        behind under sustained descent. Without this, decelerate_tick()
        would spend its ramp closing that stale gap and overshoot the depth
        that was already correct."""
        pos, _ = site_pose(data, self.arm.site_id)
        self.arm.target_pos = pos

    def _advance(self, data, dt, target_omega_mag, direction_sign, track_arm=True, advance_z=True, accel=None):
        max_delta = (accel if accel is not None else self.angular_accel) * dt
        self._current_omega_mag += np.clip(target_omega_mag - self._current_omega_mag, -max_delta, max_delta)
        self._screw_target += direction_sign * self._current_omega_mag * dt
        data.ctrl[self.screw_actuator_id] = self._screw_target
        if track_arm:
            if advance_z:
                dz = self.pitch_m / (2 * np.pi) * self._current_omega_mag * dt
                self.arm.target_pos = self.arm.target_pos + np.array([0.0, 0.0, -dz])
            # Keep the arm's own orientation target in sync with reality
            # instead of fixed - see the module docstring. Computed
            # analytically from the intended rotation rather than read back
            # from the actual site pose, which would let small tracking sag
            # get silently accepted as correct every tick with nothing
            # pulling it back.
            total_dtheta = -(self._screw_target - self._start_angle)
            c, s = np.cos(total_dtheta), np.sin(total_dtheta)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self.arm.target_quat = tf.mat_to_quat(Rz @ tf.quat_to_mat(self._initial_target_quat))
            self.arm.step(data, dt)

    def tick(self, data, dt):
        """Advance the helix by one tick and drive the arm toward it - call
        every physics step until a real measured depth (e.g.
        thread_face_z() <= target_z) says to stop, then switch to
        decelerate_tick() instead of stopping cold. This is open-loop
        position tracking on the commanded angle, not closed-loop on depth
        itself, so checking the real result is what matters."""
        self._advance(data, dt, self.omega_mag, self.sign)

    def decelerate_tick(self, data, dt):
        """Ramp rotation down to a stop instead of freezing the target
        instantly - see the module docstring for the reaction-torque jolt
        that avoids. Doesn't keep advancing the commanded depth
        (advance_z=False): by the time this is called the real-depth check
        has already fired, so continuing to feed a deeper target would
        overshoot. Orientation is still tracked so the arm doesn't fight
        the tool's decelerating spin."""
        self._advance(data, dt, 0.0, self.sign, advance_z=False)

    def is_stopped(self, tol=0.01):
        return self._current_omega_mag < tol

    def start_unwind(self, data):
        """Prepare to spin the tool back - not all the way to its angle at
        start(), just to the nearest angle that looks the same (a whole
        number of turns short of it). The tool is rotationally symmetric
        and has already released the QD by this point, so unwinding the
        full historical amount would just be slower than necessary."""
        self._current_omega_mag = 0.0
        current = data.qpos[self.screw_qpos_adr]
        total_forward = (current - self._start_angle) * self.sign  # >= 0, in "turns forward" units
        remainder = total_forward % (2 * np.pi)
        self._unwind_target_qpos = current - self.sign * remainder
        # Reference point for unwind_compensation_rotation() below: how much
        # screw_adapter has rotated since UNWIND started, not since the
        # original screw began.
        self._unwind_reference_angle = current
        self._screw_target = current

    def unwind_tick(self, data, dt):
        """Rotate the tool joint back toward the angle computed in
        start_unwind() - call after releasing the QD (weld.release()) and
        before flying elsewhere, until unwind_done(data). Doesn't touch the
        arm; without this, the pinch site would keep whatever roll screwing
        left behind and the next pick's IK would have to correct it,
        hitting the same wrist singularity all over again.

        Speed ramps down automatically as it nears the target (a standard
        stopping-distance switch) for the same reaction-jolt reason as
        tick()/decelerate_tick(). Recomputing a "safe speed" fresh from the
        remaining distance every tick (tried first) turned out numerically
        unstable for small distances - the target speed and remaining
        distance shrink together and the ramp never reaches a real cruise
        phase."""
        remaining = abs(self._unwind_target_qpos - data.qpos[self.screw_qpos_adr])
        stopping_distance = self._current_omega_mag ** 2 / (2 * self.unwind_accel) if self.unwind_accel > 0 else 0.0
        target_omega = 0.0 if remaining <= stopping_distance else self.unwind_omega_mag
        self._advance(data, dt, target_omega, -self.sign, track_arm=False, accel=self.unwind_accel)

    def unwind_done(self, data, tol_rad=0.01):
        return abs(data.qpos[self.screw_qpos_adr] - self._unwind_target_qpos) < tol_rad

    def unwind_compensation_rotation(self):
        """3x3 world-frame rotation matrix for how much screw_adapter has
        rotated since start_unwind() - for a caller driving the arm's other
        6 joints toward a separate target at the SAME TIME unwind_tick()
        runs, compose this onto that target's orientation each tick (see
        ArmController.step()'s extra_quat_correction) so the IK doesn't
        fight screw_adapter's own contribution to the gripper's pose. Same
        idea as _advance()'s analytical compensation, generalized to a
        moving target.

        Running unwind concurrently with flying and no compensation self-
        collides badly (12 contact pairs vs. the usual 0 in a full run) -
        this is the missing piece."""
        total_dtheta = -(self._screw_target - self._unwind_reference_angle)
        c, s = np.cos(total_dtheta), np.sin(total_dtheta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
