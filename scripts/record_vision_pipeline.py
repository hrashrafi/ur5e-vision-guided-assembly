"""Records a separate video from record_full_session.py's own output: the
wrist_cam's view, continuously, for the whole session - it plays live
throughout, the same way the main session video does for third_person_cam,
rather than just showing static snapshots at detection instants. At each
real vision-detection moment, the live feed briefly freezes on that frame
with the detection result drawn onto it (a hex outline, a hole circle, an
occupied port in a third color, and each one's computed world X/Y) before
continuing.

Reuses the real constants/detectors from record_full_session.py (imported,
not re-derived) and runs the same real pick/transport/insert motion, so
later port-detection events see genuine occupied-port states rather than a
staged one.
"""

import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

# Two paths need adding: this file's own directory (scripts/), so the
# sibling-script import below works, and its parent (the project root), so
# qd_sim/ is importable too.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from record_full_session import (  # noqa: E402
    ENGAGEMENT_DEPTH_M, EXPECTED_TURNS, GRASP_OFFSET, MANIFOLD_HOVER_XY,
    MANIFOLD_HOVER_Z, PORT_MATCH_RADIUS_M, PORT_TRACK_FLOOR_Z, PORT_TRACK_STEPS,
    PORT_Z, PORTS_HARDCODED, QD_FLANGE_TOP_Z, QD_GRASP_Z, QD_MATCH_RADIUS_M,
    QD_NEST_HARDCODED, QD_TRACK_FLOOR_Z, QD_TRACK_STEPS,
    SAFE_Z, SCREW_ANGULAR_ACCEL_RAD_S2, SCREW_ANGULAR_SPEED_RAD_S, SESSION,
    THREAD_FACE_LOCAL_OFFSET, TRAY_HOVER_XY, TRAY_HOVER_Z, VISION_HEIGHT,
    VISION_WIDTH, WRIST_CAM_OPTICAL_OFFSET_XY_M, disable_qd_collision, quat_close,
)
from qd_sim.control.screw_controller import ScrewController
from qd_sim.robot.arm_controller import ArmController
from qd_sim.robot.gripper_controller import GripperController
from qd_sim.robot.kinematics import site_pose
from qd_sim.sim.env import SimEnv
from qd_sim.tasks.kinematic_weld import KinematicWeld
from qd_sim.tasks.pick_sequence import (GRIPPER_DOWN_QUAT,
                                         APPROACH_CLEARANCE_M, PickSequence)
from qd_sim.tasks.transport_sequence import TransportSequence
from qd_sim.vision.annotate import (COAST_TEXT_COLOR, DIM_COLOR, HEX_COLOR,
                                     HOLE_COLOR, OCCUPIED_COLOR, annotate)
from qd_sim.vision.camera_intrinsics import intrinsics_from_mj_camera
from qd_sim.vision.ray_cast import pixel_to_world_on_plane
from qd_sim.vision.shape_detector import (detect_hexagons, detect_holes,
                                           is_hole_occupied)

FPS = 30
PHYSICS_HZ = 500
STEPS_PER_FRAME = PHYSICS_HZ // FPS
# Output is displayed larger than the native 960x720 vision render so the
# overlays/labels stay legible - the vision pipeline itself is unaffected
# (all detection still runs at the real 960x720; this is a display-only
# upscale of the same pixels/annotations).
OUT_SCALE = 1.5
OUT_WIDTH, OUT_HEIGHT = int(VISION_WIDTH * OUT_SCALE), int(VISION_HEIGHT * OUT_SCALE)
STEP_HOLD_S = 1.4    # how long the live feed pauses on each live-tracking step
FINAL_HOLD_S = 2.5   # extra pause on the last tracking step, before install() resumes the live feed
# annotate()/draw_labels() and the HEX/HOLE/OCCUPIED/DIM/COAST_TEXT color
# constants used to live here directly - moved to qd_sim/vision/annotate.py
# once qd_sim/viz/live_dashboard.py needed the exact same drawing logic
# for the --live interactive window.


class VisionPipelineRecorder:
    def __init__(self, out_path):
        self.env = SimEnv()
        self.arm = ArmController(self.env.model, "gripper/pinch")
        self.gripper = GripperController(self.env.model)
        self.weld = KinematicWeld(self.env.model)
        self.arm.reset(self.env.data)
        self.gripper.reset(self.env.data)
        self.vision_renderer = mujoco.Renderer(self.env.model, height=VISION_HEIGHT, width=VISION_WIDTH)
        self.writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (OUT_WIDTH, OUT_HEIGHT))
        self.cam_id = self.env.model.camera("wrist_cam").id
        self.K, _ = intrinsics_from_mj_camera(self.env.model, "wrist_cam", VISION_WIDTH, VISION_HEIGHT)
        self.frame_count = 0

    # ---- live recording: every sub-loop below calls this at
    # STEPS_PER_FRAME cadence, so the output plays continuously for the
    # WHOLE session (mirrors record_full_session.py's own record_frame()
    # pattern, just rendering wrist_cam instead of third_person_cam) ----
    def render_wrist(self):
        self.vision_renderer.update_scene(self.env.data, camera="wrist_cam")
        return cv2.cvtColor(self.vision_renderer.render(), cv2.COLOR_RGB2BGR)

    def record_frame(self):
        disp = cv2.resize(self.render_wrist(), (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
        self.writer.write(disp)
        self.frame_count += 1

    def hold_frame(self, frame_bgr, seconds):
        """Freeze the live feed on an already-rendered (typically
        annotated) frame for a beat - used only at real detection moments."""
        disp = cv2.resize(frame_bgr, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
        for _ in range(int(seconds * FPS)):
            self.writer.write(disp)
        self.frame_count += int(seconds * FPS)

    # ---- plumbing mirrored from record_full_session.py, extended to
    # record continuously instead of just a handful of isolated stills ----
    def run_until(self, step_fn, is_done_fn, max_steps):
        for i in range(max_steps):
            step_fn()
            self.env.step(1)
            self.weld.update(self.env.data)
            if i % STEPS_PER_FRAME == 0:
                self.record_frame()
            if is_done_fn():
                return i
        return max_steps

    def move_to(self, pos, quat, max_steps=8000):
        self.arm.set_target_pose(np.asarray(pos), np.asarray(quat))
        return self.run_until(lambda: self.arm.step(self.env.data, self.env.dt),
                               lambda: self.arm.is_converged(self.env.data), max_steps)

    def fly_safe(self, target_xy, target_z, quat=GRIPPER_DOWN_QUAT, safe_z=SAFE_Z):
        cur_pos, cur_quat = site_pose(self.env.data, self.arm.site_id)
        if not quat_close(cur_quat, quat):
            self.move_to(cur_pos, quat)
        cur_pos, _ = site_pose(self.env.data, self.arm.site_id)
        self.move_to([cur_pos[0], cur_pos[1], safe_z], quat)
        self.move_to([target_xy[0], target_xy[1], safe_z], quat)
        self.move_to([target_xy[0], target_xy[1], target_z], quat)

    def thread_face_z(self, qd_name):
        qd_pos = self.env.data.body(qd_name).xpos.copy()
        qd_R = self.env.data.body(qd_name).xmat.reshape(3, 3).copy()
        return (qd_pos + qd_R @ THREAD_FACE_LOCAL_OFFSET)[2]

    def install(self, qd_name, grasp_point, port):
        approach_xy = grasp_point[:2]
        approach_z = grasp_point[2] + APPROACH_CLEARANCE_M
        self.fly_safe(approach_xy, approach_z)
        pick = PickSequence(self.arm, self.gripper, grasp_point)
        pick.start(self.env.data, self.env.dt)
        prev_state = [pick.state]

        def pick_step():
            pick.tick(self.env.data, self.env.dt)
            if prev_state[0] == "GRASP" and pick.state == "LIFT":
                self.weld.engage(self.env.data, qd_name, self.arm.site_id)
                disable_qd_collision(self.env.model, qd_name)
                self.gripper.hold_here(self.env.data)
            prev_state[0] = pick.state
        self.run_until(pick_step, pick.is_done, 15000)

        self.fly_safe(port[:2], MANIFOLD_HOVER_Z)
        pinch_pos, _ = site_pose(self.env.data, self.arm.site_id)
        offset = pinch_pos[2] - self.thread_face_z(qd_name)
        transport = TransportSequence(self.arm, port, offset)
        transport.start(self.env.data, self.env.dt)
        self.run_until(lambda: transport.tick(self.env.data, self.env.dt), transport.is_done, 20000)

        screw = ScrewController(self.arm, self.env.model, ENGAGEMENT_DEPTH_M, EXPECTED_TURNS,
                                 SCREW_ANGULAR_SPEED_RAD_S, angular_accel_rad_s2=SCREW_ANGULAR_ACCEL_RAD_S2, reverse=False)
        screw.start(self.env.data)
        self.run_until(lambda: screw.tick(self.env.data, self.env.dt),
                        lambda: self.thread_face_z(qd_name) <= port[2], 40000)
        screw.begin_stop(self.env.data)
        self.run_until(lambda: screw.decelerate_tick(self.env.data, self.env.dt), screw.is_stopped, 4000)
        for i in range(150):
            self.env.step(1); self.weld.update(self.env.data)
            if i % STEPS_PER_FRAME == 0:
                self.record_frame()
        start_ctrl = self.env.data.ctrl[self.gripper.actuator_id]
        for tick in range(180):
            frac = (tick + 1) / 180
            self.gripper.close(self.env.data, ctrl_value=start_ctrl + frac * (self.gripper.OPEN_CTRL - start_ctrl))
            self.env.step(1); self.weld.update(self.env.data)
            if tick % STEPS_PER_FRAME == 0:
                self.record_frame()
        self.weld.release()
        screw.finish()
        disable_qd_collision(self.env.model, qd_name)
        cur_pos, cur_quat = site_pose(self.env.data, self.arm.site_id)
        self.move_to([cur_pos[0], cur_pos[1], SAFE_Z], cur_quat)
        screw.start_unwind(self.env.data)
        self.run_until(lambda: screw.unwind_tick(self.env.data, self.env.dt), lambda: screw.unwind_done(self.env.data), 8000)

    # ---- closed-loop visual tracking, visualized: mirrors
    # record_full_session.py's live_track_descend() (same constants
    # imported from there so the two scripts can't drift apart), but
    # pauses the live feed briefly at every tracking step instead of just
    # once at the end, so the live-adjustment process is actually visible:
    # every detected shape is drawn, the tracked one in its bright color
    # with a "TRACKING" label, others dimmed gray, and a banner when
    # nothing was found this step (coasting on the last known position). ----
    def detect_hex_shapes(self, img):
        """Every currently-visible hex: (px, py, polygon, world_xy)."""
        out = []
        for (px, py), polygon in detect_hexagons(img, return_polygons=True):
            world_xy = pixel_to_world_on_plane(px, py, self.K, self.env.data.cam_xpos[self.cam_id],
                                                self.env.data.cam_xmat[self.cam_id], QD_FLANGE_TOP_Z)[:2]
            out.append((px, py, polygon, world_xy))
        return out, []  # no non-trackable "extra" shapes for the QD case

    def detect_hole_shapes(self, img):
        """Every currently-visible EMPTY hole (tracking candidates):
        (px, py, (px,py,r), world_xy). Occupied ports are real shapes too
        but never tracking candidates - returned separately, just for
        display (dimmed differently, in OCCUPIED_COLOR, not DIM_COLOR, so
        they still read as "occupied" rather than "some other empty hole
        we're ignoring")."""
        empty, occupied = [], []
        for px, py, r in detect_holes(img, return_radius=True):
            if is_hole_occupied(img, px, py):
                occupied.append(("circle", (px, py, r), OCCUPIED_COLOR))
            else:
                world_xy = pixel_to_world_on_plane(px, py, self.K, self.env.data.cam_xpos[self.cam_id],
                                                    self.env.data.cam_xmat[self.cam_id], PORT_Z)[:2]
                empty.append((px, py, (px, py, r), world_xy))
        return empty, occupied

    def live_track_and_record(self, hover_xy, hover_z, track_floor_z, seed_xy,
                               detect_shapes_fn, match_radius_m, n_steps, label,
                               shape_kind, active_color):
        """Mirrors record_full_session.py's live_track_descend() (same
        math, same per-step camera-offset compensation), pausing the live
        feed on each step's annotated frame instead of running silently."""
        self.fly_safe(hover_xy, hover_z)
        current_xy = np.array(seed_xy, dtype=float)
        z_schedule = list(np.linspace(hover_z, track_floor_z, n_steps + 1))
        for step_i, z in enumerate(z_schedule):
            if step_i > 0:
                cam_target = current_xy - np.array(WRIST_CAM_OPTICAL_OFFSET_XY_M)
                self.move_to([cam_target[0], cam_target[1], z], GRIPPER_DOWN_QUAT)
            raw = self.render_wrist()
            candidates, extra_shapes = detect_shapes_fn(raw)

            shapes, det_labels = list(extra_shapes), []
            tracked = False
            for px, py, shape_repr, world_xy in candidates:
                dist = np.linalg.norm(np.asarray(world_xy) - current_xy)
                if dist < match_radius_m and not tracked:
                    shapes.append((shape_kind, shape_repr, active_color))
                    det_labels.append((px, py, [f"TRACKING {label}", f"x={world_xy[0]*1000:+.1f}mm y={world_xy[1]*1000:+.1f}mm"], active_color))
                    current_xy = np.asarray(world_xy, dtype=float)
                    tracked = True
                else:
                    shapes.append((shape_kind, shape_repr, DIM_COLOR))
            annotated = annotate(raw, shapes, det_labels)
            if not tracked:
                cv2.putText(annotated, f"{label}: not visible this step - coasting on last known position",
                            (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COAST_TEXT_COLOR, 2, cv2.LINE_AA)
            hold = FINAL_HOLD_S if step_i == len(z_schedule) - 1 else STEP_HOLD_S
            self.hold_frame(annotated, hold)
            print(f"[vision-pipeline] {label} live-track z={z:.3f}: "
                  f"{'tracked' if tracked else 'coasting'} -> ({current_xy[0]:.4f}, {current_xy[1]:.4f})")
        return current_xy

    def close(self):
        self.writer.release()


def main():
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "out/vision_pipeline.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = VisionPipelineRecorder(out_path)

    for qd_name, port_num in SESSION:
        qd_xy = rec.live_track_and_record(TRAY_HOVER_XY, TRAY_HOVER_Z, QD_TRACK_FLOOR_Z,
                                           QD_NEST_HARDCODED[qd_name][:2], rec.detect_hex_shapes,
                                           QD_MATCH_RADIUS_M, QD_TRACK_STEPS, qd_name,
                                           "polygon", HEX_COLOR)
        grasp_point = np.array([qd_xy[0], qd_xy[1], QD_GRASP_Z])
        diff_mm = np.linalg.norm(grasp_point - (QD_NEST_HARDCODED[qd_name] + GRASP_OFFSET)) * 1000
        print(f"[vision-pipeline] {qd_name} final tracked grasp point (vs hardcoded, diff={diff_mm:.2f}mm)")

        port_xy = rec.live_track_and_record(MANIFOLD_HOVER_XY, MANIFOLD_HOVER_Z, PORT_TRACK_FLOOR_Z,
                                             PORTS_HARDCODED[port_num][:2], rec.detect_hole_shapes,
                                             PORT_MATCH_RADIUS_M, PORT_TRACK_STEPS, f"port_{port_num}",
                                             "circle", HOLE_COLOR)
        port = np.array([port_xy[0], port_xy[1], PORT_Z])
        diff_mm = np.linalg.norm(port - PORTS_HARDCODED[port_num]) * 1000
        print(f"[vision-pipeline] port_{port_num} final tracked position (vs hardcoded, diff={diff_mm:.2f}mm)")

        rec.install(qd_name, grasp_point, port)
        print(f"[vision-pipeline] {qd_name} installed in port_{port_num}")

    rec.close()
    print(f"\nFrames recorded: {rec.frame_count} ({rec.frame_count/FPS:.1f}s at {FPS}fps)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
