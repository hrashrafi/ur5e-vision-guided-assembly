"""
Records one continuous video of the full session: starting from the arm's
"ready" pose, picking up all 4 QDs one at a time and inserting each into its
own manifold port, then returning the arm to the exact ready pose it started
from.

Between task phases (ready -> first QD, port -> next QD, last port -> ready)
the arm flies via a safe overhead waypoint instead of moving directly.
PickSequence/TransportSequence's own internal moves were already collision-
tested, but a direct move for these outer transitions - especially ready's
folded pose straight into a reach down into the tray - genuinely
self-collides (measured up to 8 simultaneous arm/gripper contacts). Splitting
each transition into reorient-in-place -> rise to a safe height -> descend
(see fly_safe()) measured zero self-contacts across the whole run.

Usage:
  venv/bin/python scripts/record_full_session.py [--out out/full_session.mp4]
  venv/bin/python scripts/record_full_session.py --live   # also opens one
      interactive window, split into a live third-person robot view (left
      half) and a wrist-cam dashboard (right half: detection overlay on
      top, a telemetry panel + START button along the bottom) - the arm
      doesn't move until START is clicked/'s' is pressed. Same video is
      still recorded as always - see qd_sim/viz/live_dashboard.py.
"""

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

# Makes `python scripts/record_full_session.py` work when run from the
# project root without PYTHONPATH set - qd_sim/ is a sibling of scripts/,
# and Python only auto-adds the running script's own directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qd_sim.sim.env import SimEnv
from qd_sim.robot import arm_controller
from qd_sim.robot.arm_controller import ArmController
from qd_sim.robot.gripper_controller import GripperController
from qd_sim.robot.kinematics import site_pose
from qd_sim.tasks.pick_sequence import PickSequence, GRIPPER_DOWN_QUAT
from qd_sim.tasks.transport_sequence import TransportSequence
from qd_sim.tasks.kinematic_weld import KinematicWeld
from qd_sim.control.screw_controller import ScrewController
from qd_sim.utils import transforms as tf
from qd_sim.vision.annotate import DIM_COLOR, HEX_COLOR, HOLE_COLOR, OCCUPIED_COLOR
from qd_sim.vision.camera_intrinsics import intrinsics_from_mj_camera
from qd_sim.vision.ray_cast import pixel_to_world_on_plane
from qd_sim.vision.shape_detector import detect_hexagons, detect_holes, is_hole_occupied
from qd_sim.viz.live_dashboard import LiveDashboard


class LiveSessionAborted(Exception):
    """Raised (only when --live) when the user presses 'q'/Esc in either
    window - caught in main() to clean up the video writer and both
    windows before exiting, instead of a half-written file and windows
    left open."""

THREAD_FACE_LOCAL_OFFSET = np.array([0, 0, 0.0429])

# Reference-only: the real QD/port positions used at runtime come from
# live_track_descend() below (shape detection on a wrist-cam render,
# re-checked at each step while descending), not these constants. Kept here
# as (a) a known-good comparison printed alongside each vision detection,
# since in simulation the "right answer" is knowable in advance (these are
# exactly world.xml's authored positions), and (b) the coarse hover
# position and initial match seed live_track_descend() starts from.
QD_NEST_HARDCODED = {
    "qd_1": np.array([0.42, -0.23, 0.0559]),
    "qd_2": np.array([0.48, -0.23, 0.0559]),
    "qd_3": np.array([0.42, -0.17, 0.0559]),
    "qd_4": np.array([0.48, -0.17, 0.0559]),
}
PORTS_HARDCODED = {
    1: np.array([0.4575, 0.175, 0.050]),
    2: np.array([0.5025, 0.175, 0.050]),
    3: np.array([0.5475, 0.175, 0.050]),
    4: np.array([0.5925, 0.175, 0.050]),
}
# qd_N -> port N, in order
SESSION = [("qd_1", 1), ("qd_2", 2), ("qd_3", 3), ("qd_4", 4)]

# Grasps the flange, not the smooth shaft further up - a round shaft gives
# the closing pads nothing to mechanically lock against, and the QD could
# visibly twist mid-grip before the kinematic weld locks its pose.
# pick_sequence.py's GRIPPER_DOWN_QUAT already yaw-aligns to the flange's
# flats; this just sets grasp height. Found by sweeping the real gap
# between the pad's bottom face and the flange's bottom edge - lands the
# pad tip about 0.26mm above the flange edge, a little clearance rather
# than flush or into the thread above.
GRASP_OFFSET = np.array([0.0, 0.0, -0.0125])

# See qd_sim/control/screw_controller.py for the control-law reasoning.
#
# Real thread spec, from the connector's datasheet: 7/8"-14 UNF male
# straight thread (ISO 11926-3 / SAE J1926 ORB), 14 TPI. The mesh only
# shows a cosmetic ridge pattern in the thread region, not a real
# tessellated helix, so the pitch can't be measured from geometry alone -
# the datasheet is the source of truth here, and its dimensions match this
# project's own direct mesh measurements to within 0.02mm.
QD_THREAD_TPI = 14
QD_THREAD_PITCH_M = 0.0254 / QD_THREAD_TPI  # 1.8143mm/turn (25.4mm/14)
# Engagement depth = distance from the flange's thread-side face (mesh
# z=42.9mm) to the true tip (z=54.9mm), from slicing the mesh directly -
# independently confirmed by the datasheet's own "Thread Length: 12.0mm".
ENGAGEMENT_DEPTH_M = 0.0120
EXPECTED_TURNS = ENGAGEMENT_DEPTH_M / QD_THREAD_PITCH_M  # ~6.61 turns
SCREW_ANGULAR_SPEED_RAD_S = 2.5 * np.pi  # 1.25 rev/s - no real-world spec
                                           # for driving speed, picked for
                                           # watchable pacing.
# Time to ramp to/from full speed = SCREW_ANGULAR_SPEED_RAD_S / this.
# ScrewController's own default (a 0.5s ramp) still looks abrupt on the
# arm, so this slows it to roughly a 2.5s ramp - kept as a fixed
# acceleration (not a fixed ramp time) so it scales automatically if the
# speed above changes.
SCREW_ANGULAR_ACCEL_RAD_S2 = np.pi

# Separate, faster speed for the post-screw "unwind" - safe to raise
# independently of SCREW_ANGULAR_SPEED_RAD_S since unwind is a purely
# cosmetic derotation of the empty tool, not constrained by real thread
# engagement (see ScrewController.unwind_tick()).
UNWIND_ANGULAR_SPEED_RAD_S = 2 * SCREW_ANGULAR_SPEED_RAD_S
UNWIND_ANGULAR_ACCEL_RAD_S2 = 6 * SCREW_ANGULAR_ACCEL_RAD_S2

# Height every inter-task transition rises to before translating - well
# above the tray/manifold/arm's own folded pose, verified to route every
# big reconfiguration clear of self-collision. Matches TRAY_HOVER_Z on
# purpose, so every transit is monotonic: rise once, translate across at
# height, then descend - rather than climbing past the hover height only
# for the tracking descent to immediately reverse it.
SAFE_Z = 0.50

# --- Vision detection (live_track_descend(), called from within main()
# below). Both the QD and port cases fly to a coarse "look roughly here"
# hover, then descend in steps, re-detecting the real features (a QD's hex
# flange, a manifold's port holes - no printed markers) at each one,
# closed-loop, rather than a single shot at hover height. Both hovers use
# the ordinary GRIPPER_DOWN_QUAT orientation (wrist_cam is mounted to look
# straight down at that same orientation).
#
# wrist_cam's mount has a real lateral offset from the pinch site even
# though it looks straight down, so its optical center doesn't sit
# directly above whatever the hover targets actually position. Measured
# directly and left uncorrected this silently distorts every detection's
# framing - TRAY_HOVER_XY/MANIFOLD_HOVER_XY below are each (feature
# center) minus this offset, so the camera's true optical center lands on
# the actual feature center.
WRIST_CAM_OPTICAL_OFFSET_XY_M = (0.0577, -0.0688)

_TRAY_CENTER_XY = (0.45, -0.20)  # tray center - average of QD_NEST_HARDCODED
TRAY_HOVER_XY = (_TRAY_CENTER_XY[0] - WRIST_CAM_OPTICAL_OFFSET_XY_M[0],
                  _TRAY_CENTER_XY[1] - WRIST_CAM_OPTICAL_OFFSET_XY_M[1])
TRAY_HOVER_Z = 0.50  # higher than needed for framing alone, to fight a
                       # real accuracy issue: at any off-vertical viewing
                       # angle the QD's shaft top visually shifts relative
                       # to the flange beneath it, pulling the detected
                       # centroid off-center - worse further from the
                       # camera axis. Hovering higher shrinks that angular
                       # deviation and roughly halves the mean error with
                       # no change to the detection rate.
_MANIFOLD_CENTER_XY = (0.525, 0.175)  # port row center - average of PORTS_HARDCODED
MANIFOLD_HOVER_XY = (_MANIFOLD_CENTER_XY[0] - WRIST_CAM_OPTICAL_OFFSET_XY_M[0],
                      _MANIFOLD_CENTER_XY[1] - WRIST_CAM_OPTICAL_OFFSET_XY_M[1])
MANIFOLD_HOVER_Z = 0.30  # frames all 4 port holes with margin.
# Known, fixed world-Z heights vision ray-casts detected pixels onto (see
# qd_sim/vision/ray_cast.py) - the scene doesn't leave these uncertain
# (QDs rest on a known floor height, the manifold's top face height is
# fixed), only X/Y needs refining from the detected shape.
#
# This Z must be the actual height of the visible surface the camera is
# looking at, not just any Z the caller wants back - the ray-cast
# intersects the pixel's real camera ray with this plane, so the wrong Z
# here silently produces wrong X/Y too. Using the grasp height instead of
# the flange top's real height was a real bug early on; ray-casting to the
# flange's own top height (QD_FLANGE_TOP_Z below) fixes it, and the final
# grasp point is built at the separate, correct grasp Z afterward.
QD_FLANGE_TOP_Z = QD_NEST_HARDCODED["qd_1"][2] - 0.0349  # the flange's own top face height
QD_GRASP_Z = QD_NEST_HARDCODED["qd_1"][2] + GRASP_OFFSET[2]  # flange grasp height
PORT_Z = PORTS_HARDCODED[1][2]  # manifold top face height

# Closed-loop visual tracking parameters - see main()'s live_track_descend()
# for how these are used. Module-level so record_vision_pipeline.py can
# import the exact same values rather than risk drifting from them.
QD_TRACK_FLOOR_Z = 0.28    # hex detection is clean down to here; below this
                             # the hex's pixel footprint grows past the
                             # detector's max area and its shape distorts,
                             # so this keeps margin above that cliff.
PORT_TRACK_FLOOR_Z = 0.20  # same idea for hole detection.
QD_TRACK_STEPS = 4
PORT_TRACK_STEPS = 3
QD_MATCH_RADIUS_M = 0.030   # under half the 60mm tray grid spacing
PORT_MATCH_RADIUS_M = 0.020  # under half the 45mm port row spacing

WIDTH, HEIGHT = 1920, 1440
# Separate, smaller resolution for the vision (wrist-cam) renderer - faster,
# and shape_detector.py's thresholds are calibrated specifically against
# this resolution, so keep them in sync if this ever changes.
VISION_WIDTH, VISION_HEIGHT = 960, 720
FPS = 30
PHYSICS_HZ = 500
STEPS_PER_FRAME = PHYSICS_HZ // FPS
GRIPPER_OPEN_RAMP_TICKS = 90   # ramped open (mirrors PickSequence's
                                 # GRASP_RAMP_TICKS close) instead of an
                                 # instant ctrl snap, which looked like the
                                 # gripper popping open.

# Third-person-only cosmetic override - swaps in a darker/richer manifold
# material for the recorded video and dashboard's robot-view half only, by
# showing geom group 5 (the pretty overlay) and hiding group 1 (the
# vision-safe color). Every vision-facing render in this file always uses
# the unmodified default scene options, so this swap is invisible to
# detection by construction.
THIRD_PERSON_SCENE_OPTION = mujoco.MjvOption()
THIRD_PERSON_SCENE_OPTION.geomgroup[1] = 0  # hide the vision-safe manifold color
THIRD_PERSON_SCENE_OPTION.geomgroup[5] = 1  # show the pretty overlay instead


def disable_qd_collision(model, qd_name):
    """Turn off collision for a placed QD's geoms - call once it's seated
    and released. A placed QD is done: it only needs to look right from
    here on, and leaving its collision on is a real problem, not cosmetic
    - later QDs sit on ports only 45mm apart, close enough that installing
    a neighbor clips an already-placed one. The QD itself can't move (it's
    a kinematically-welded mocap body), but the gripper very much can: a
    mocap body still exerts a real reaction force on anything that touches
    it, so the controller ends up fighting a physical shove on every later
    install."""
    for suffix in ("col_far", "col_flange", "col_thread"):
        geom_id = model.geom(f"{qd_name}_{suffix}").id
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0


def tilt_deg(data, model, body_name):
    R = data.body(body_name).xmat.reshape(3, 3)
    return np.degrees(np.arccos(np.clip((R @ np.array([0, 0, -1]))[2], -1, 1)))


def quat_close(q1, q2, tol_deg=1.0):
    _, angle_err = tf.pose_error(np.zeros(3), q1, np.zeros(3), q2)
    return angle_err < tol_deg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out/full_session.mp4")
    parser.add_argument("--live", action="store_true",
                         help="Open an interactive live Robot View + Dashboard "
                              "(wrist-cam overlay/telemetry, gated behind a START "
                              "button) while recording the same video as always.")
    parser.add_argument("--auto-start", action="store_true",
                         help="With --live, skip the START button/'s' key wait and begin "
                              "the session immediately - for generating a recording "
                              "unattended, not for normal interactive use.")
    parser.add_argument("--live-out", default="out/live_dashboard.mp4",
                         help="With --live, also save the dashboard window itself "
                              "(both halves, overlay, telemetry panel) to this video "
                              "file, at the same pacing as the plain --out recording.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = SimEnv()
    arm = ArmController(env.model, "gripper/pinch")
    gripper = GripperController(env.model)
    weld = KinematicWeld(env.model)
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    vision_renderer = mujoco.Renderer(env.model, height=VISION_HEIGHT, width=VISION_WIDTH)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (WIDTH, HEIGHT))
    live_out_path = None
    if args.live:
        live_out_path = Path(args.live_out)
        live_out_path.parent.mkdir(parents=True, exist_ok=True)
    dash = LiveDashboard(record_path=live_out_path, record_fps=FPS) if args.live else None

    # --- Live-dashboard state (all no-ops when dash is None) -----------
    # live_wrist_state holds whatever the most recent live-tracking step
    # classified for display; session_status holds the telemetry
    # build_hud_lines() reads each frame. Both are just display bookkeeping
    # - never read by anything that affects the actual robot motion.
    live_wrist_state = {"img": None, "shapes": [], "labels": [], "coast_text": None}
    session_status = {"qd": None, "port": None, "diff_mm": None, "tilt_deg": None, "completed": 0}
    # Outside a live-tracking step, record_frame() falls back to a plain
    # wrist_cam render for the dashboard. Re-rendering it every other frame
    # rather than every frame (LIVE_PLAIN_WRIST_STRIDE) halves that cost -
    # the wrist view barely changes tick-to-tick outside active tracking,
    # so reusing the last frame for one tick out of two isn't visible.
    LIVE_PLAIN_WRIST_STRIDE = 2
    plain_wrist_cache = [None]

    def reset_live_wrist_state():
        live_wrist_state.update(img=None, shapes=[], labels=[], coast_text=None)
        plain_wrist_cache[0] = None

    def render_wrist_plain():
        vision_renderer.update_scene(env.data, camera="wrist_cam")
        return cv2.cvtColor(vision_renderer.render(), cv2.COLOR_RGB2BGR)

    def build_hud_lines():
        from qd_sim.viz.live_dashboard import PANEL_HEADING, PANEL_TEXT
        lines = [(f"Phase: {phase_label[0]}", PANEL_HEADING)]
        if session_status["qd"] is not None:
            lines.append((f"Current QD/port: {session_status['qd']} -> port {session_status['port']}", PANEL_TEXT))
        if session_status["diff_mm"] is not None:
            lines.append((f"Last tracked vs. reference: {session_status['diff_mm']:.2f}mm", PANEL_TEXT))
        if session_status["tilt_deg"] is not None:
            lines.append((f"Last grasp tilt: {session_status['tilt_deg']:.2f}deg", PANEL_TEXT))
        lines.append((f"QDs installed so far: {session_status['completed']}/{len(SESSION)}", PANEL_TEXT))
        lines.append((f"Self-collision contacts (peak): {self_collision_peak}", PANEL_TEXT))
        lines.append((f"Frame: {frame_count}  ({frame_count / FPS:.1f}s)", PANEL_TEXT))
        return lines

    frame_count = 0
    # Arm/gripper self-collision tracker, checked every physics step for
    # the whole run so the final report reflects the actual session, not
    # just the transitions this was originally tuned against.
    ARM_BODY_PREFIXES = ("shoulder", "upper_arm", "forearm", "wrist", "gripper", "base")
    self_collision_peak = 0
    self_collision_pairs = set()

    phase_label = ["init"]

    def set_phase(label):
        phase_label[0] = label

    def record_frame():
        nonlocal frame_count
        renderer.update_scene(env.data, camera="third_person_cam", scene_option=THIRD_PERSON_SCENE_OPTION)
        img = renderer.render()
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frame_count += 1
        if dash is not None:
            wrist_img = live_wrist_state["img"]
            if wrist_img is None:
                if plain_wrist_cache[0] is None or frame_count % LIVE_PLAIN_WRIST_STRIDE == 0:
                    plain_wrist_cache[0] = render_wrist_plain()
                wrist_img = plain_wrist_cache[0]
            still_running = dash.update(img, wrist_img, build_hud_lines(),
                                         shapes=live_wrist_state["shapes"],
                                         labels=live_wrist_state["labels"],
                                         coast_text=live_wrist_state["coast_text"])
            if not still_running:
                raise LiveSessionAborted("User pressed quit in the live window.")

    def check_self_collision():
        nonlocal self_collision_peak
        n = 0
        for c in env.data.contact[: env.data.ncon]:
            b1 = env.model.body(env.model.geom_bodyid[c.geom1]).name
            b2 = env.model.body(env.model.geom_bodyid[c.geom2]).name
            if b1.startswith(ARM_BODY_PREFIXES) and b2.startswith(ARM_BODY_PREFIXES) and b1 != b2:
                n += 1
                self_collision_pairs.add(tuple(sorted((b1, b2))))
        if n > self_collision_peak:
            self_collision_peak = n

    def run_until(step_fn, is_done_fn, max_steps):
        for i in range(max_steps):
            step_fn()
            env.step(1)
            weld.update(env.data)  # no-op until engage() has been called
            check_self_collision()
            if i % STEPS_PER_FRAME == 0:
                record_frame()
            if is_done_fn():
                return i
        return max_steps

    # Cruise pace for the smooth Cartesian trajectory planner - centralized
    # in ArmController (see DEFAULT_CRUISE_SPEED_MPS there).
    CRUISE_SPEED_MPS = arm_controller.DEFAULT_CRUISE_SPEED_MPS
    CRUISE_ANGULAR_SPEED_DEG_S = arm_controller.DEFAULT_CRUISE_ANGULAR_SPEED_DEG_S
    MIN_MOVE_STEPS = arm_controller.DEFAULT_MIN_MOVE_STEPS

    def plan_steps(dist_m, angle_deg):
        duration_s = max(dist_m / CRUISE_SPEED_MPS, angle_deg / CRUISE_ANGULAR_SPEED_DEG_S)
        return max(MIN_MOVE_STEPS, int(duration_s / env.dt))

    def move_to(pos, quat, max_steps=8000, pos_tol_m=None, angle_tol_deg=None):
        arm.set_smooth_target_paced(env.data, pos, quat, env.dt,
                                     cruise_speed_mps=CRUISE_SPEED_MPS,
                                     cruise_angular_speed_deg_s=CRUISE_ANGULAR_SPEED_DEG_S,
                                     min_steps=MIN_MOVE_STEPS)
        tol_kwargs = {}
        if pos_tol_m is not None:
            tol_kwargs["pos_tol_m"] = pos_tol_m
        if angle_tol_deg is not None:
            tol_kwargs["angle_tol_deg"] = angle_tol_deg
        return run_until(lambda: arm.step(env.data, env.dt),
                          lambda: arm.is_converged(env.data, **tol_kwargs), max_steps)

    # Still used by live_track_descend()'s own exit handoff below (a plain
    # reactive set_target_pose()). fly_safe() itself no longer needs a
    # loose tolerance now that its rise/translate/descend legs are one
    # continuous planned trajectory rather than three separately-converging
    # move_to() calls.
    FLY_LOOSE_POS_TOL_M = 0.03
    FLY_LOOSE_ANGLE_TOL_DEG = 8.0

    def fly_safe(target_xy, target_z, quat=GRIPPER_DOWN_QUAT, safe_z=SAFE_Z):
        """Reorient in place (if needed), then rise straight up at the
        current xy to safe_z, translate at safe_z to target_xy, then
        descend to (target_xy, target_z) - see module docstring for why a
        direct diagonal move isn't used: the rise guarantees no horizontal
        movement happens until the arm has already cleared safe_z, well
        above any installed QD's real height.

        The three legs after reorientation - rise, translate, descend -
        are one continuous trajectory (ArmController.set_smooth_path()):
        a single accelerate-cruise-decelerate profile governs the whole
        detour, so the arm passes through the via-points at cruising
        speed with no stop in between, while still visiting them in the
        required order. Only reorientation (rare - the gripper stays
        pointed down almost the whole session) is a separate, fully-
        converging move first.

        The completion check is arm.is_near_final(), not is_converged() -
        this returns once the arm is close to the hover point while still
        cruising, since the caller (live_track_descend()) immediately sets
        its own next target and that handoff preserves whatever velocity
        the arm already has.

        When the destination is above safe_z (e.g. the tray hover, which
        matches SAFE_Z), the mid-height translate waypoint is skipped -
        otherwise the final climb into the hover would be purely vertical,
        and the tracking descent that follows would immediately reverse it:
        a pure direction reversal that passes through zero speed. Going
        diagonally instead keeps horizontal motion alive the whole way, and
        is just as safe since the whole diagonal stays at or above safe_z."""
        cur_pos, cur_quat = site_pose(env.data, arm.site_id)
        if not quat_close(cur_quat, quat):
            move_to(cur_pos, quat)
        cur_pos, _ = site_pose(env.data, arm.site_id)
        waypoints = [([cur_pos[0], cur_pos[1], safe_z], quat)]
        if target_z < safe_z:
            # Destination is below the safe height, so the translate must
            # happen up at safe_z and the descent only afterwards - the
            # whole collision-avoidance point of this function.
            waypoints.append(([target_xy[0], target_xy[1], safe_z], quat))
        waypoints.append(([target_xy[0], target_xy[1], target_z], quat))
        total_dist = sum(np.linalg.norm(np.array(waypoints[i][0]) - (cur_pos if i == 0 else np.array(waypoints[i - 1][0])))
                          for i in range(len(waypoints)))
        arm.set_smooth_path(env.data, waypoints, plan_steps(total_dist, 0.0))
        run_until(lambda: arm.step(env.data, env.dt), lambda: arm.is_near_final(env.data), 12000)

    arm.reset(env.data)
    gripper.reset(env.data)
    ready_pos, ready_quat = site_pose(env.data, arm.site_id)
    ready_arm_qpos = env.data.qpos[arm.qpos_idx].copy()

    def thread_face_z(qd_name):
        qd_pos = env.data.body(qd_name).xpos.copy()
        qd_R = env.data.body(qd_name).xmat.reshape(3, 3).copy()
        return (qd_pos + qd_R @ THREAD_FACE_LOCAL_OFFSET)[2]

    # --- Closed-loop visual tracking --------------------------------
    #
    # The arm flies to a known coarse hover (still fixed/assumed - vision
    # only refines from it, never replaces it), then descends toward
    # track_floor_z, re-rendering and re-detecting continuously and
    # refining the X/Y estimate toward whichever detected candidate is
    # nearest the current one (never a different, still-visible QD/hole -
    # see match_radius_m below). Below the height where a detector
    # reliably sees anything (the *_TRACK_FLOOR_Z constants above), the
    # last good estimate is simply kept rather than guessed further. The
    # existing PickSequence/TransportSequence then take over the final
    # fixed-target approach/grasp/insert from wherever tracking leaves the
    # arm. record_vision_pipeline.py imports these same constants directly
    # so the two scripts can't drift apart.

    def detect_hexagons_world():
        """Render wrist_cam right now and return (world_xy_list, raw_img,
        shape_pairs, occupied) - world_xy_list is the world (x, y) of every
        currently-visible hex flange, ray-cast to the flange's true top
        height; shape_pairs is [(world_xy, polygon)] in the same order, for
        the --live dashboard's overlay; occupied is always [] here (no such
        concept for QDs in the tray) - present only so this and
        detect_empty_holes_world() share one return shape.

        Always requests return_polygons=True, even when dash is None - not
        a second detection pass, since detect_hexagons() computes the
        polygon internally regardless (already needed for the vertex-count
        check); the flag only changes what's returned."""
        img = render_wrist_plain()
        cam_id = env.model.camera("wrist_cam").id
        K, _ = intrinsics_from_mj_camera(env.model, "wrist_cam", VISION_WIDTH, VISION_HEIGHT)
        shape_pairs = []
        for (px, py), polygon in detect_hexagons(img, return_polygons=True):
            world_xy = pixel_to_world_on_plane(px, py, K, env.data.cam_xpos[cam_id],
                                                env.data.cam_xmat[cam_id], QD_FLANGE_TOP_Z)[:2]
            shape_pairs.append((world_xy, polygon))
        return [xy for xy, _ in shape_pairs], img, shape_pairs, []

    def detect_empty_holes_world():
        """Same idea for manifold holes: render wrist_cam now, return
        (world_xy_list, raw_img, shape_pairs, occupied) - world_xy_list is
        the world (x, y) of every currently empty hole (is_hole_occupied()
        filters out already-installed ports), ray-cast to the manifold's
        known top height; the extra two are for --live's overlay only,
        same free-lunch principle as detect_hexagons_world()."""
        img = render_wrist_plain()
        cam_id = env.model.camera("wrist_cam").id
        K, _ = intrinsics_from_mj_camera(env.model, "wrist_cam", VISION_WIDTH, VISION_HEIGHT)
        shape_pairs, occupied = [], []
        for px, py, r in detect_holes(img, return_radius=True):
            if is_hole_occupied(img, px, py):
                occupied.append((px, py, r))
            else:
                world_xy = pixel_to_world_on_plane(px, py, K, env.data.cam_xpos[cam_id],
                                                    env.data.cam_xmat[cam_id], PORT_Z)[:2]
                shape_pairs.append((world_xy, (px, py, r)))
        return [xy for xy, _ in shape_pairs], img, shape_pairs, occupied

    def live_track_descend(hover_xy, hover_z, track_floor_z, seed_xy,
                            detect_world_fn, match_radius_m, label, max_steps=12000,
                            shape_kind=None, active_color=None):
        """Fly to the coarse (hover_xy, hover_z), then drive smoothly and
        continuously down to track_floor_z in one motion - the arm's normal
        accel/speed-limited motion, not a series of discrete stop-and-
        recheck waypoints.

        Vision runs continuously alongside the motion: once per rendered
        frame (the same STEPS_PER_FRAME cadence used for recording
        elsewhere, matching how a real camera feed would actually be
        processed), re-detect and refine the live X/Y target, feeding it
        straight into the arm's current Cartesian target. Z is commanded
        straight to track_floor_z from the start, so the arm's own natural
        accel/decel profile governs the descent's pace and the camera still
        sees every real intermediate height along the way.

        Whichever detected candidate is nearest the current target is
        adopted, but only if within match_radius_m - farther detections are
        ignored outright, since that's likely a different QD/hole still
        visible, not the one being tracked. When nothing is detected, the
        target is simply left unchanged - coasting on the latest good fix.
        seed_xy anchors the very first match before any tracking history
        exists. Converges (arm.is_converged()'s normal tight tolerance)
        once actually at (current_xy - camera offset, track_floor_z) - that
        final pose is what PickSequence/TransportSequence take over from.

        shape_kind/active_color (--live only): when given, also feeds the
        live dashboard's wrist-cam overlay each detection tick, built from
        the same detect_world_fn() call already made for tracking above."""
        set_phase(f"fly_hover:{label}")
        fly_safe(hover_xy, hover_z)
        set_phase(f"track:{label}")
        current_xy = np.array(seed_xy, dtype=float)
        tick = 0

        def step_fn():
            nonlocal current_xy, tick
            if tick % STEPS_PER_FRAME == 0:
                candidates, raw_img, shape_pairs, occupied = detect_world_fn()
                if candidates:
                    best = min(candidates, key=lambda p: np.linalg.norm(np.asarray(p) - current_xy))
                    dist = np.linalg.norm(np.asarray(best) - current_xy)
                    if dist < match_radius_m:
                        current_xy = np.asarray(best, dtype=float)
                        status = f"tracked (moved {dist*1000:.1f}mm)"
                    else:
                        status = f"ignored a detection {dist*1000:.1f}mm away (not the tracked object)"
                else:
                    status = "no detection - coasting on last known position"
                cur_z, _ = site_pose(env.data, arm.site_id)
                print(f"[vision] {label} live-track z={cur_z[2]:.3f}: {status} -> ({current_xy[0]:.4f}, {current_xy[1]:.4f})")
                # Target the pinch at (current_xy - the wrist_cam mount
                # offset), not at current_xy itself, so the camera - not
                # the pinch - stays centered above the tracked target, same
                # compensation the hover point already applies once up
                # front. Skipping this at every step leaves a monotonic
                # drift and a noticeably worse grasp tilt.
                cam_target = current_xy - np.array(WRIST_CAM_OPTICAL_OFFSET_XY_M)
                arm.set_target_pose(np.array([cam_target[0], cam_target[1], track_floor_z]), GRIPPER_DOWN_QUAT)

                if dash is not None and shape_kind is not None:
                    shapes_draw, labels_draw, found_active = [], [], False
                    for world_xy, shape_repr in shape_pairs:
                        d = np.linalg.norm(np.asarray(world_xy) - current_xy)
                        if d < match_radius_m and not found_active:
                            shapes_draw.append((shape_kind, shape_repr, active_color))
                            anchor = (np.mean(shape_repr, axis=0) if shape_kind == "polygon"
                                      else shape_repr[:2])
                            labels_draw.append((anchor[0], anchor[1],
                                                 [f"TRACKING {label}",
                                                  f"x={world_xy[0]*1000:+.1f}mm y={world_xy[1]*1000:+.1f}mm"],
                                                 active_color))
                            found_active = True
                        else:
                            shapes_draw.append((shape_kind, shape_repr, DIM_COLOR))
                    for px, py, r in occupied:
                        shapes_draw.append(("circle", (px, py, r), OCCUPIED_COLOR))
                    live_wrist_state["img"] = raw_img
                    live_wrist_state["shapes"] = shapes_draw
                    live_wrist_state["labels"] = labels_draw
                    live_wrist_state["coast_text"] = (
                        None if found_active else f"{label}: not visible this step - coasting")
            arm.step(env.data, env.dt)
            tick += 1

        # Loose exit tolerance: the downstream state (PickSequence's
        # APPROACH / TransportSequence's TO_PORT_ABOVE) immediately sets
        # its own fresh target the moment this returns, and that state's
        # own handoff into the final descend is already loose too.
        run_until(step_fn, lambda: arm.is_converged(env.data, pos_tol_m=FLY_LOOSE_POS_TOL_M,
                                                      angle_tol_deg=FLY_LOOSE_ANGLE_TOL_DEG), max_steps)
        return current_xy

    if dash is not None and not args.auto_start:
        set_phase("idle")
        renderer.update_scene(env.data, camera="third_person_cam", scene_option=THIRD_PERSON_SCENE_OPTION)
        preview_third = renderer.render()
        preview_wrist = render_wrist_plain()
        subtitle = ["Queued: " + ", ".join(f"{q}->port{p}" for q, p in SESSION)]
        if not dash.wait_for_start(preview_third, preview_wrist, subtitle):
            writer.release()
            dash.close()
            out_path.unlink(missing_ok=True)  # nothing real was ever recorded - don't leave an empty .mp4
            print("Live session cancelled before start - no video written.")
            return

    try:
        for qd_name, port_num in SESSION:
            session_status["qd"] = qd_name
            session_status["port"] = port_num
            qd_xy = live_track_descend(TRAY_HOVER_XY, TRAY_HOVER_Z, QD_TRACK_FLOOR_Z,
                                        QD_NEST_HARDCODED[qd_name][:2], detect_hexagons_world,
                                        QD_MATCH_RADIUS_M, qd_name,
                                        shape_kind="polygon", active_color=HEX_COLOR)
            reset_live_wrist_state()
            grasp_point = np.array([qd_xy[0], qd_xy[1], QD_GRASP_Z])
            diff_mm = np.linalg.norm(grasp_point - (QD_NEST_HARDCODED[qd_name] + GRASP_OFFSET)) * 1000
            session_status["diff_mm"] = diff_mm
            print(f"[vision] {qd_name} final tracked grasp point: {grasp_point} (vs hardcoded, diff={diff_mm:.2f}mm)")

            # No separate fly_safe() to the grasp point here -
            # live_track_descend already brought the arm down to
            # QD_TRACK_FLOOR_Z directly above the tracked X/Y; the pick
            # sequence commands its own approach point and handles the
            # remaining descent itself.
            pick = PickSequence(arm, gripper, grasp_point)
            pick.start(env.data, env.dt)
            prev_state = pick.state

            def pick_step():
                nonlocal prev_state
                set_phase(f"pick:{pick.state}")
                pick.tick(env.data, env.dt)
                if prev_state == "GRASP" and pick.state == "LIFT":
                    weld.engage(env.data, qd_name, arm.site_id)
                    # Disable the QD's own collision the instant it's
                    # welded, not just after it's placed - from here on the
                    # weld provides 100% of the QD's positional control,
                    # and real contact with the gripper serves no purpose:
                    # the QD is a mocap body immune to being pushed, so if
                    # the closed pads ever penetrate its collision volume
                    # the contact solver escalates the correction force
                    # every tick trying to resolve it (measured climbing
                    # from a normal ~36N to over 14,000N within about 20
                    # ticks during screwing).
                    disable_qd_collision(env.model, qd_name)
                    # Freeze the gripper's own ctrl target at its true,
                    # physically-contact-limited position the same tick.
                    # PickSequence's grasp hold always commands full close
                    # regardless of contact - real contact with the QD was
                    # the only thing holding that back, and
                    # disable_qd_collision() just removed it. Without this
                    # the fingers keep closing straight through the QD.
                    gripper.hold_here(env.data)
                prev_state = pick.state

            steps = run_until(pick_step, pick.is_done, 15000)
            session_status["tilt_deg"] = tilt_deg(env.data, env.model, qd_name)
            print(f"[{qd_name}] Pick done in {steps} steps, tilt={session_status['tilt_deg']:.2f}deg")

            port_xy = live_track_descend(MANIFOLD_HOVER_XY, MANIFOLD_HOVER_Z, PORT_TRACK_FLOOR_Z,
                                          PORTS_HARDCODED[port_num][:2], detect_empty_holes_world,
                                          PORT_MATCH_RADIUS_M, f"port_{port_num}",
                                          shape_kind="circle", active_color=HOLE_COLOR)
            reset_live_wrist_state()
            port = np.array([port_xy[0], port_xy[1], PORT_Z])
            diff_mm = np.linalg.norm(port - PORTS_HARDCODED[port_num]) * 1000
            session_status["diff_mm"] = diff_mm
            print(f"[vision] port_{port_num} final tracked position: {port} (vs hardcoded, diff={diff_mm:.2f}mm)")
            # No separate centering fly_safe() needed here either - see
            # the pick case above; the transport sequence commands its own
            # transit point and handles the remaining descent. Measured,
            # not assumed - see TransportSequence's module docstring for
            # why this replaced a hardcoded flange-specific constant.
            pinch_pos, _ = site_pose(env.data, arm.site_id)
            pinch_to_thread_face_offset_m = pinch_pos[2] - thread_face_z(qd_name)
            transport = TransportSequence(arm, port, pinch_to_thread_face_offset_m)
            transport.start(env.data, env.dt)

            def transport_step():
                set_phase(f"transport:{transport.state}")
                transport.tick(env.data, env.dt)

            steps = run_until(transport_step, transport.is_done, 20000)
            print(f"[{qd_name}] Transport done in {steps} steps, tilt={tilt_deg(env.data, env.model, qd_name):.2f}deg")

            # Kinematic thread-advance: the real screwing motion - see
            # qd_sim/control/screw_controller.py for the full reasoning.
            # Starts turning right from wherever transport's pre-insert
            # point left off, not from a separate approach-first move: at
            # this stretched-out reach the arm settles into a persistent
            # few-mm droop under its own weight, and a separate big
            # Cartesian jump toward the manifold would overshoot into that
            # droop before correcting back - a visible dive toward the
            # manifold with real risk of the gripper reaching it early. The
            # screw controller's own per-tick advance is already small-step
            # and gradual, so starting it early removes that risky move.
            # reverse=False (default) - verified via world coordinates, not
            # by eye: this direction rotates clockwise viewed from above
            # while descending, matching the standard right-hand "tightening"
            # convention.
            screw = ScrewController(arm, env.model, ENGAGEMENT_DEPTH_M, EXPECTED_TURNS,
                                     SCREW_ANGULAR_SPEED_RAD_S,
                                     angular_accel_rad_s2=SCREW_ANGULAR_ACCEL_RAD_S2, reverse=False,
                                     unwind_speed_rad_s=UNWIND_ANGULAR_SPEED_RAD_S,
                                     unwind_accel_rad_s2=UNWIND_ANGULAR_ACCEL_RAD_S2)
            screw.start(env.data)
            set_phase(f"screw_turn:{qd_name}")
            steps = run_until(lambda: screw.tick(env.data, env.dt), lambda: thread_face_z(qd_name) <= port[2], 40000)
            # Ramp rotation/descent back down to a full stop rather than
            # freezing instantly (a real reaction-torque jolt otherwise -
            # see ScrewController's module docstring). A brief hold
            # afterward, turning and gripper both untouched, gives the stop
            # a clear beat before anything else starts.
            screw.begin_stop(env.data)
            set_phase(f"screw_decel:{qd_name}")
            decel_steps = run_until(lambda: screw.decelerate_tick(env.data, env.dt), screw.is_stopped, 4000)
            # From here through the end of the gripper opening, the arm is
            # not touched at all - ctrl just holds whatever it last was.
            # Calling arm.step() through this window kept the position
            # servos "live" even with an unchanging target, and that small
            # residual correction was enough to visibly disturb the
            # already-seated (still welded) QD.
            #
            # 60 ticks gives a real, visible beat (0.12s) so the stop reads
            # as deliberate. record_frame() is gated the same way every
            # other loop in this file is - at 500Hz physics but a 30fps
            # writer, an ungated call would play back far slower than real
            # time.
            for i in range(60):
                env.step(1)
                weld.update(env.data)
                check_self_collision()
                if i % STEPS_PER_FRAME == 0:
                    record_frame()
            print(f"[{qd_name}] Insert done in {steps}+{decel_steps} steps, thread_face_z={thread_face_z(qd_name):.4f} "
                  f"(target {port[2]:.4f}), tilt={tilt_deg(env.data, env.model, qd_name):.2f}deg")

            # Let go: open the gripper on a ramp (mirroring PickSequence's
            # ramped close), arm still untouched, then stop the weld - the
            # QD stays exactly where it is (mocap bodies never move on
            # their own), and the gripper is now free to fly off without
            # dragging it along.
            start_ctrl = env.data.ctrl[gripper.actuator_id]
            for tick in range(GRIPPER_OPEN_RAMP_TICKS):
                frac = (tick + 1) / GRIPPER_OPEN_RAMP_TICKS
                gripper.close(env.data, ctrl_value=start_ctrl + frac * (gripper.OPEN_CTRL - start_ctrl))
                env.step(1)
                weld.update(env.data)
                check_self_collision()
                if tick % STEPS_PER_FRAME == 0:
                    record_frame()
            weld.release()
            screw.finish()  # unlock wrist_3 - the arm needs full 6-DOF control again to fly elsewhere
            disable_qd_collision(env.model, qd_name)

            # Rise straight up to a safe height before unwinding the tool
            # joint's rotation, not after. Unwinding first at the low
            # insertion height is harmless (the QD's already released) but
            # looks wrong - the gripper visibly spins right at the height
            # of the other installed QDs, reading as about to collide even
            # though it never does. Rising first matches how this would
            # look done by hand: retract clear, then reset orientation.
            cur_pos, cur_quat = site_pose(env.data, arm.site_id)
            set_phase(f"post_screw_rise:{qd_name}")
            move_to([cur_pos[0], cur_pos[1], SAFE_Z], cur_quat)

            # Unwind the tool joint back to the nearest angle that looks
            # the same as its pre-screw one - not the full amount screwed
            # (see ScrewController.start_unwind's docstring) - now that the
            # arm is safely above the other QDs, before flying to the next
            # one. Runs sequentially rather than concurrently with the next
            # flight: doing both at once needs the flight trajectory to
            # compensate for screw_adapter's own contribution to the
            # gripper's pose (ArmController.step()'s extra_quat_correction
            # and ScrewController.unwind_compensation_rotation() already
            # support this), but the compensation only stays valid while a
            # trajectory is actively re-sampling - once it finishes early,
            # the correction keeps compounding on itself with nothing to
            # reset it. Not worth chasing for the roughly one second per QD
            # it would save; UNWIND_ANGULAR_ACCEL_RAD_S2 already keeps the
            # sequential unwind fast with no concurrency risk.
            screw.start_unwind(env.data)
            set_phase(f"unwind:{qd_name}")
            steps = run_until(lambda: screw.unwind_tick(env.data, env.dt), lambda: screw.unwind_done(env.data), 8000)
            print(f"[{qd_name}] Unwound in {steps} steps")
            session_status["completed"] += 1

        # All 4 QDs placed - fly back to the exact pose the arm started
        # from. Reorient to ready_quat while still up at the safe height,
        # then descend already in that final orientation - reorienting
        # down at ready_pos itself measured a real self-collision (gripper
        # base mount vs wrist_2), the same class of problem fly_safe's
        # other transitions are already routed around.
        set_phase("fly_ready")
        fly_safe(ready_pos[:2], SAFE_Z, quat=GRIPPER_DOWN_QUAT)
        move_to([ready_pos[0], ready_pos[1], SAFE_Z], ready_quat)
        move_to(ready_pos, ready_quat)

        # Belt-and-suspenders exact snap to the ready keyframe's own joint
        # angles (IK reaching the same Cartesian pose doesn't guarantee
        # bit-identical joint angles). Capped at 800 ticks: the loop's
        # tolerance is never fully reachable in practice - the arm settles
        # into a small steady-state droop under its own weight and never
        # improves past it, so this just keeps comfortable margin past
        # that plateau without burning extra ticks.
        env.data.ctrl[arm.actuator_idx] = ready_arm_qpos
        for i in range(800):
            env.step(1)
            weld.update(env.data)
            check_self_collision()
            if i % STEPS_PER_FRAME == 0:
                record_frame()
            if np.allclose(env.data.qpos[arm.qpos_idx], ready_arm_qpos, atol=0.001):
                break
        print(f"Returned to ready pose in {i} steps")

        # hold the final frame for a beat so the video doesn't end abruptly
        for _ in range(30):
            record_frame()

        print(f"\nFrames recorded: {frame_count} ({frame_count/FPS:.1f}s at {FPS}fps)")
        print(f"Peak arm/gripper self-collision contacts anywhere in the run: "
              f"{self_collision_peak} (pairs involved: {self_collision_pairs or 'none'})")
        writer.release()
        print(f"Saved: {out_path}")
    except LiveSessionAborted as e:
        print(f"\nLive session aborted: {e}")
        print(f"Frames recorded before abort: {frame_count} ({frame_count/FPS:.1f}s at {FPS}fps)")
        writer.release()
        print(f"Partial video saved: {out_path}")
    finally:
        if dash is not None:
            dash.close()


if __name__ == "__main__":
    main()
