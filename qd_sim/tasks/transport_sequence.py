"""
Transport sequence: carry a grasped QD from wherever the pick sequence left
it to a pre-insertion pose above the target port, ready for the
insert-and-screw controller to take over.
States: TO_PORT_ABOVE -> DESCEND_TO_PRE_INSERT -> DONE.

The QD is carried by a kinematic weld (qd_sim/tasks/kinematic_weld.py), not
friction/contact - the arm just needs to move, the weld keeps the QD's pose
exactly right relative to the gripper regardless of where that is.

pinch_to_thread_face_offset_m has to be measured by the caller (pinch_z
minus thread_face_z, taken right after the grasp) and passed in, rather than
assumed fixed - the pinch site isn't guaranteed to be concentric with the
flange, and hardcoding that relationship broke silently as soon as the
grasp point moved.
"""

import numpy as np

from qd_sim.control.state_machine import StateMachine
from qd_sim.robot.arm_controller import DEFAULT_CRUISE_SPEED_MPS as CRUISE_SPEED_MPS
from qd_sim.tasks.pick_sequence import GRIPPER_DOWN_QUAT

TRANSIT_CLEARANCE_M = 0.10   # height above the port to fly over at during transit
PRE_INSERT_CLEARANCE_M = 0.020  # height above full-seat pose to hand off to
                                  # the screw controller. The screw controller
                                  # advances gradually per tick rather than
                                  # jumping straight to depth, so there's no
                                  # collision risk bringing this hand-off
                                  # point in close - it still leaves the
                                  # thread tip a few mm clear of the port at
                                  # the start.

# Loose tolerance for the TO_PORT_ABOVE->DESCEND_TO_PRE_INSERT handoff -
# mirrors pick_sequence.py's APPROACH_LOOSE_POS_TOL_M on the insert side.
# Full tight convergence (is_converged()'s default 3mm/2deg) above the port
# before starting the descent caused a visible stop-then-go; this loosens
# just the handoff, DESCEND_TO_PRE_INSERT's own convergence stays tight.
TRANSIT_LOOSE_POS_TOL_M = 0.03
TRANSIT_LOOSE_ANGLE_TOL_DEG = 8.0


class TransportSequence:
    def __init__(self, arm_ctrl, port_target_world, pinch_to_thread_face_offset_m):
        self.arm = arm_ctrl
        port = np.asarray(port_target_world, dtype=float)
        # port_target_world[2] is already the full-seat thread_face_z target,
        # so no separate "seated pose" reference point is needed here.
        pinch_offset = np.array([0, 0, pinch_to_thread_face_offset_m])
        self.transit_point = port + np.array([0, 0, TRANSIT_CLEARANCE_M]) + pinch_offset
        self.pre_insert_point = port + np.array([0, 0, PRE_INSERT_CLEARANCE_M]) + pinch_offset
        self._dt = None  # set every tick() call

        self.fsm = StateMachine(
            handlers={
                "TO_PORT_ABOVE": self._h_to_port_above,
                "DESCEND_TO_PRE_INSERT": self._h_descend,
            },
            start_state="TO_PORT_ABOVE",
        )

    def start(self, data, dt):
        # Go straight to the transit point (well above the port) first, so
        # the grasped QD doesn't get dragged sideways through anything at
        # grasp height on the way there. set_smooth_target_paced() blends
        # in from the arm's current velocity, same as PickSequence.start().
        self._dt = dt
        self.arm.set_smooth_target_paced(data, self.transit_point, GRIPPER_DOWN_QUAT, dt,
                                          cruise_speed_mps=CRUISE_SPEED_MPS)
        self.fsm.reset()

    def _h_to_port_above(self, data):
        # is_near_final(), not is_converged() - TO_PORT_ABOVE's own entry
        # (above) is now a smooth trajectory, and is_converged() refuses
        # to fire while any trajectory is active (see its own docstring) -
        # same reasoning as pick_sequence.py's _h_approach.
        if self.arm.is_near_final(data, pos_tol_m=TRANSIT_LOOSE_POS_TOL_M,
                                   angle_tol_deg=TRANSIT_LOOSE_ANGLE_TOL_DEG):
            self.arm.set_smooth_target_paced(data, self.pre_insert_point, GRIPPER_DOWN_QUAT,
                                              self._dt, cruise_speed_mps=CRUISE_SPEED_MPS)
            return "DESCEND_TO_PRE_INSERT"
        return "TO_PORT_ABOVE"

    def _h_descend(self, data):
        if self.arm.is_converged(data, pos_tol_m=0.002):
            return "DONE"
        return "DESCEND_TO_PRE_INSERT"

    def tick(self, data, dt):
        self._dt = dt
        self.arm.step(data, dt)
        self.fsm.tick(data)
        return self.fsm.state

    @property
    def state(self):
        return self.fsm.state

    def is_done(self):
        return self.fsm.is_done()
