"""
Pick sequence: approach above a QD's grasp point (the hex flange), descend,
close the gripper, lift. States: APPROACH -> DESCEND -> GRASP -> LIFT -> DONE.

The flange has hex flats, and the gripper has to align its closing axis with
a pair of opposite flats instead of just pointing straight down at whatever
yaw. Get this wrong and the QD tips ~30deg the moment lifting starts - two
flat pads closing on a hex at a corner-catching angle just isn't a stable
grip.

GRIPPER_DOWN_QUAT bakes in the "point straight down" 180deg-about-X flip
plus a yaw tuned against the hex's actual corner angles. The yaw came from a
small empirical sweep rather than deriving it analytically end to end - the
mesh-frame-to-world-frame sign conventions were fiddly enough that testing a
few values was faster and more reliable.

Current numbers: 0.51deg tilt (down from ~30deg) at yaw=40deg, grasping near
the flange's lower edge.
"""

import numpy as np

from qd_sim.control.state_machine import StateMachine
from qd_sim.robot.arm_controller import DEFAULT_CRUISE_SPEED_MPS as CRUISE_SPEED_MPS
from qd_sim.utils import transforms as tf


def _gripper_down_quat(yaw_deg):
    """Point the gripper straight down (180deg about X: local +Z, toward
    the fingers, maps to world -Z), then yaw about world Z by yaw_deg to
    align the closing axis with a pair of hex flats."""
    R_flip = np.diag([1.0, -1.0, -1.0])
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return tf.mat_to_quat(R_yaw @ R_flip)


GRASP_YAW_DEG = 40.0  # empirically tuned against the flange's hex flats
GRIPPER_DOWN_QUAT = _gripper_down_quat(GRASP_YAW_DEG)

APPROACH_CLEARANCE_M = 0.08  # height above the grasp point to approach from

# Loose tolerance for the APPROACH->DESCEND handoff, so the arm doesn't
# fully decelerate to a stop above the grasp point before starting the
# descent (a visible stutter otherwise). DESCEND's own convergence into the
# actual grasp stays tight.
APPROACH_LOOSE_POS_TOL_M = 0.03
APPROACH_LOOSE_ANGLE_TOL_DEG = 8.0

# Same idea for LIFT's completion check - DESCEND->GRASP is the one moment
# that actually needs to be stationary; grinding to a full stop before
# flying to the manifold was just wasted time.
LIFT_LOOSE_POS_TOL_M = 0.03
LIFT_LOOSE_ANGLE_TOL_DEG = 8.0
GRASP_SETTLE_TICKS = 440     # ticks to hold the close command before lifting
GRASP_RAMP_TICKS = 400       # of those, ticks spent ramping ctrl open->closed
                              # (0.8s at 500Hz) instead of snapping straight
                              # to full close, which can jolt the QD out of
                              # alignment right as it's grasped.

# The "pinch" site isn't where the pads actually make contact - queried
# from the pad collision geoms, their combined center sits ~14.3mm short of
# the pinch site. Targeting pinch==grasp_point directly put the flange too
# deep into the gripper for a good, centered grip; this offset corrects for
# it by pulling the commanded pinch position back by the same amount.
PINCH_TO_PAD_CENTER_OFFSET_M = 0.0143


class PickSequence:
    def __init__(self, arm_ctrl, gripper_ctrl, grasp_point_world):
        self.arm = arm_ctrl
        self.gripper = gripper_ctrl
        self.grasp_point = np.asarray(grasp_point_world, dtype=float)
        # What we actually command the pinch site to (see
        # PINCH_TO_PAD_CENTER_OFFSET_M) so the pads' real contact center,
        # not the pinch site, lands on the flange.
        self._pinch_target_at_grasp = self.grasp_point + np.array([0, 0, -PINCH_TO_PAD_CENTER_OFFSET_M])
        self.approach_point = self._pinch_target_at_grasp + np.array([0, 0, APPROACH_CLEARANCE_M])
        self._grasp_ticks = 0
        self._dt = None  # set every tick() call

        self.fsm = StateMachine(
            handlers={
                "APPROACH": self._h_approach,
                "DESCEND": self._h_descend,
                "GRASP": self._h_grasp,
                "LIFT": self._h_lift,
            },
            start_state="APPROACH",
        )

    def start(self, data, dt):
        # Deliberately not calling self.arm.reset(data) here - that zeroes
        # ArmController's tracked velocity, which would defeat the smooth
        # trajectory's start tangent below. Callers do their own one-time
        # arm.reset() before the first pick instead.
        self._dt = dt
        self.gripper.reset(data)
        self.gripper.open(data)
        self.arm.set_smooth_target_paced(data, self.approach_point, GRIPPER_DOWN_QUAT, dt,
                                          cruise_speed_mps=CRUISE_SPEED_MPS)
        self.fsm.reset()

    def _h_approach(self, data):
        # is_near_final(), not is_converged(): APPROACH's entry above is a
        # smooth trajectory, and is_converged() won't fire while any
        # trajectory is active - using it here would mean waiting for the
        # planned decelerate-to-rest to finish before even checking.
        if self.arm.is_near_final(data, pos_tol_m=APPROACH_LOOSE_POS_TOL_M,
                                   angle_tol_deg=APPROACH_LOOSE_ANGLE_TOL_DEG):
            self.arm.set_smooth_target_paced(data, self._pinch_target_at_grasp, GRIPPER_DOWN_QUAT,
                                              self._dt, cruise_speed_mps=CRUISE_SPEED_MPS)
            return "DESCEND"
        return "APPROACH"

    def _h_descend(self, data):
        if self.arm.is_converged(data, pos_tol_m=0.002):
            self._grasp_ticks = 0
            return "GRASP"
        return "DESCEND"

    def _h_grasp(self, data):
        self._grasp_ticks += 1
        # Ramp the close command open->closed over GRASP_RAMP_TICKS instead
        # of commanding full close in one step, then hold fully closed for
        # the rest of the settle window.
        ramp_frac = min(self._grasp_ticks / GRASP_RAMP_TICKS, 1.0)
        ctrl = self.gripper.OPEN_CTRL + ramp_frac * (self.gripper.CLOSED_CTRL - self.gripper.OPEN_CTRL)
        self.gripper.close(data, ctrl_value=ctrl)
        if self._grasp_ticks >= GRASP_SETTLE_TICKS:
            self.arm.set_smooth_target_paced(data, self.approach_point, GRIPPER_DOWN_QUAT,
                                              self._dt, cruise_speed_mps=CRUISE_SPEED_MPS)
            return "LIFT"
        return "GRASP"

    def _h_lift(self, data):
        # is_near_final(), not is_converged() - same reasoning as
        # _h_approach: LIFT's entry above is also a smooth trajectory.
        if self.arm.is_near_final(data, pos_tol_m=LIFT_LOOSE_POS_TOL_M,
                                   angle_tol_deg=LIFT_LOOSE_ANGLE_TOL_DEG):
            return "DONE"
        return "LIFT"

    def tick(self, data, dt):
        """Advance the arm/gripper controllers by one control tick and run
        one FSM transition check. Call every physics step."""
        self._dt = dt
        self.arm.step(data, dt)
        self.fsm.tick(data)
        return self.fsm.state

    @property
    def state(self):
        return self.fsm.state

    def is_done(self):
        return self.fsm.is_done()
