"""
Thin wrapper around the Robotiq 2F-85's single actuator. ctrl range is
0-255 (0=open, 255=fully closed - see 2f85.xml's actuator comment), not a
joint angle directly, since the driver joint is tendon-coupled across both
fingers via a single "split" tendon.
"""

import numpy as np


class GripperController:
    OPEN_CTRL = 0.0
    CLOSED_CTRL = 255.0

    def __init__(self, model, actuator_name="gripper/fingers_actuator"):
        self.model = model
        self.actuator_id = model.actuator(actuator_name).id
        self._target = None

    def reset(self, data):
        self._target = data.ctrl[self.actuator_id]

    def open(self, data):
        self._target = self.OPEN_CTRL
        data.ctrl[self.actuator_id] = self._target

    def close(self, data, ctrl_value=None):
        """ctrl_value lets a caller ask for a partial close (e.g. a grip
        force well short of full-closed, for a stable-but-gentle hold);
        defaults to fully closed."""
        self._target = self.CLOSED_CTRL if ctrl_value is None else ctrl_value
        data.ctrl[self.actuator_id] = self._target

    def is_closed(self, data, tol=5.0):
        return abs(data.ctrl[self.actuator_id] - self.CLOSED_CTRL) < tol

    def is_open(self, data, tol=5.0):
        return abs(data.ctrl[self.actuator_id] - self.OPEN_CTRL) < tol

    def hold_here(self, data):
        """Freeze the actuator's target at wherever it is currently,
        physically, holding - not the open/close ctrl value it was last
        commanded toward. close() always commands CLOSED_CTRL (255)
        regardless of contact: it's a soft position-servo grip, so while an
        object resists, the real steady-state position sits well short of
        255, held back by the contact force. The instant that resisting
        contact disappears - e.g. right when a kinematic weld takes over a
        grasped object's pose and its collision gets turned off - the same
        still-255 ctrl has nothing left to resist it and drives the fingers
        straight through the now-collision-free mesh. This must be called
        that same tick to prevent that.

        Solves the actuator's own affine force equation (MuJoCo's standard
        <general biastype="affine"> form) for the ctrl that makes force ~0
        at the current actuator length, rather than hardcoding this
        actuator's specific gain/bias numbers - stays correct if they're
        ever retuned. Velocity's contribution is dropped since by the time
        this is called the grasp hold has already settled for hundreds of
        ticks."""
        gainprm = self.model.actuator_gainprm[self.actuator_id]
        biasprm = self.model.actuator_biasprm[self.actuator_id]
        length = data.actuator_length[self.actuator_id]
        ctrl = -(biasprm[0] + biasprm[1] * length) / gainprm[0]
        ctrl = float(np.clip(ctrl, self.OPEN_CTRL, self.CLOSED_CTRL))
        self._target = ctrl
        data.ctrl[self.actuator_id] = ctrl
        return ctrl
