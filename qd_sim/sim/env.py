"""
Thin wrapper around loading the full scene and stepping physics - shared
by task sequences, tests, and (later) the dashboard/recording and headless
eval scripts, so there's one place that knows how to load/reset the model.
"""

from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENE = ROOT / "scene" / "scene_real.xml"


class SimEnv:
    def __init__(self, scene_path=DEFAULT_SCENE, keyframe="ready"):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.keyframe_name = keyframe
        self.reset()

    def reset(self):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe_name)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def body_pos(self, name):
        return self.data.body(name).xpos.copy()

    def body_contacts(self, name_a, name_b):
        """Number of active contacts between two named bodies right now."""
        count = 0
        for c in self.data.contact[: self.data.ncon]:
            b1 = self.model.body(self.model.geom_bodyid[c.geom1]).name
            b2 = self.model.body(self.model.geom_bodyid[c.geom2]).name
            if {b1, b2} == {name_a, name_b}:
                count += 1
        return count
