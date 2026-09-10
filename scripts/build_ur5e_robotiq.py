"""
One-time composition of the UR5e arm + Robotiq 2F-85 gripper into a single
MJCF file via mujoco.MjSpec.attach() (spec-level composition, not manual XML
splicing - this avoids actuator/joint-name collisions between the two source
models by namespacing every gripper element under a "gripper/" prefix).

The gripper is attached at the UR5e's existing "attachment_site" (defined at
the end of wrist_3_link in ur5e.xml) - the standard menagerie convention for
mounting end-effectors.

A Franka Emika Hand (true parallel-jaw, no pad tilt) was tried here first in
place of the 2F-85, but its fingertip pads turned out too small/flat to
reliably grip this QD's round flange. Back on the 2F-85, which actually
holds the part - see pick_sequence.py's GRASP_RAMP_TICKS for the slow-close
fix layered on top of it.

Writes scene/ur5e_robotiq.xml, self-contained with its own assets/ dir (mesh
files copied alongside it) so it can be loaded standalone via
mujoco.MjModel.from_xml_path() without depending on mujoco_menagerie's
internal directory layout.

Also consolidates the QD/manifold CAD meshes into that same scene/assets/
dir, since MuJoCo resolves an included fragment's relative mesh paths
against the including (top-level) file's directory, not the fragment's own
- scene/world.xml needs its meshes sitting next to scene_real.xml too.
Re-run this script any time scene/assets/ needs rebuilding from scratch.

Usage:
  venv/bin/python scripts/build_ur5e_robotiq.py
"""

import shutil
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
UR5E_DIR = ROOT / "mujoco_menagerie" / "universal_robots_ur5e"
GRIPPER_DIR = ROOT / "mujoco_menagerie" / "robotiq_2f85"
UR5E_XML = UR5E_DIR / "ur5e.xml"
GRIPPER_XML = GRIPPER_DIR / "2f85.xml"
OUT_XML = ROOT / "scene" / "ur5e_robotiq.xml"
OUT_ASSETS = OUT_XML.parent / "assets"

# Non-robot assets that scene/world.xml needs, consolidated into the same
# scene/assets/ dir (see module docstring for why).
EXTRA_ASSET_FILES = [
    ROOT / "assets" / "cad" / "qd" / "QD.stl",
    ROOT / "assets" / "cad" / "manifold" / "manifold.obj",
]


def main():
    OUT_XML.parent.mkdir(parents=True, exist_ok=True)

    arm = mujoco.MjSpec.from_file(str(UR5E_XML))
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))

    site = arm.site("attachment_site")
    if site is None:
        raise RuntimeError("ur5e.xml has no 'attachment_site' - check the source model")

    # A dedicated rotary tool adapter, mounted at the arm's own
    # attachment_site, with the gripper mounted on it rather than directly
    # on wrist_3_link - so the screwing motion can spin this one joint
    # instead of the arm's own wrist_3. wrist_3_joint has a hard +-2*pi
    # limit (real UR5e hardware), nowhere near enough travel for a
    # multi-turn continuous screw, and driving rotation through the arm's
    # own IK while pointing straight down sits right at its wrist
    # singularity. A separate joint sidesteps both: unlimited range, and
    # the arm's own 6 joints never need to reorient for screwing at all.
    # Positioned at the exact same pos/quat as attachment_site, so none of
    # the gripper's own grasp-point geometry needed to change.
    screw_adapter = arm.body("wrist_3_link").add_body(
        name="screw_adapter", pos=site.pos, quat=site.quat,
    )
    screw_adapter.add_joint(
        name="screw_adapter_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        limited=False,
        damping=0.1,
        armature=0.01,
    )
    adapter_site = screw_adapter.add_site(name="screw_adapter_site")

    arm.add_actuator(
        name="screw_adapter",
        target="screw_adapter_joint",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        # kp=5000, kd=10, not the UR5e's stock wrist-joint gains (kp=500,
        # kd=100) this started as a copy of. Those are tuned for slow
        # repositioning; screw_adapter instead cruises continuously for
        # seconds at a time, and the stock gains left it tens of degrees
        # behind the commanded angle at cruise - invisible to
        # screw_controller.py's own ramp-down logic (which only tracks the
        # commanded angle), so "stop" was getting declared while the joint
        # was still spinning. These gains cut that lag to a fraction of a
        # degree while staying well inside the actuator's torque budget.
        gainprm=[5000] + [0] * 9,
        biasprm=[0, -5000, -10] + [0] * 7,
        forcerange=[-28, 28],
        ctrllimited=False,
    )

    arm.attach(gripper, prefix="gripper/", site=adapter_site)

    # wrist_2_link and the gripper's base_mount sit at a fraction of a
    # nanometer of separation in the original menagerie models too - just
    # floating-point noise around an already-exactly-touching design, but
    # adding screw_adapter into the kinematic chain was enough to flip its
    # sign into a real (if tiny) overlap. Excluded explicitly, the same way
    # 2f85.xml already excludes its own by-design near-zero-clearance pairs.
    arm.add_exclude(bodyname1="wrist_2_link", bodyname2="gripper/base_mount")

    # Eye-in-hand wrist camera, mounted on wrist_3_link rather than further
    # down the chain on the gripper - wrist_3_link is screw_adapter's
    # parent, so the camera stays fixed relative to the arm and doesn't
    # spin along with the gripper while screwing. pos/quat point it
    # straight down when the arm is in its normal gripper-down orientation.
    wrist_3 = arm.body("wrist_3_link")
    wrist_3.add_camera(
        name="wrist_cam",
        pos=[0.09, 0.2108, 0],
        quat=[0.64085638, 0.64085638, -0.29883624, 0.29883624],
        fovy=55,
    )
    # Small cosmetic housing + lens disc so there's something physical
    # where the camera is - a MuJoCo <camera> has no visual body of its
    # own. Offset back along the camera's local +Z so the housing sits
    # behind the lens rather than blocking its view.
    cam_quat = [0.64085638, 0.64085638, -0.29883624, 0.29883624]
    wrist_3.add_geom(
        name="wrist_cam_housing", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.014, 0.014, 0.011], pos=[0, 0.06, -0.07], quat=cam_quat,
        contype=0, conaffinity=0, group=2, material="gripper/metal",
    )
    wrist_3.add_geom(
        name="wrist_cam_lens", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.006, 0.0008, 0], pos=[0, 0.071, -0.07], quat=cam_quat,
        contype=0, conaffinity=0, group=2, material="gripper/black",
    )

    # attach() warns and keeps the parent's <option> values on conflict. The
    # gripper's contact behavior was authored against cone="elliptic"
    # impratio="10" (see 2f85.xml <option>) - adopt those for the merged
    # model rather than silently keeping the arm-only defaults.
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10

    # fingers_actuator is left at 2f85.xml's own stock gain/force values -
    # the QD is a mocap body carried by a kinematic weld once grasped (see
    # world.xml), not held by contact force, so the gripper's only real job
    # is closing against the QD's surface, not generating grip pressure.

    # Sanity-compile in-memory before writing anything to disk.
    model = arm.compile()
    print(f"Compiled OK: {model.nq} qpos, {model.nu} actuators, {model.nbody} bodies")
    print("Actuators:", [model.actuator(i).name for i in range(model.nu)])

    # to_file() writes the XML with meshdir="assets/" but does NOT copy the
    # referenced mesh files - both source dirs' assets/ get consolidated
    # here (their filenames don't collide) so the output is loadable
    # standalone, independent of mujoco_menagerie's internal layout.
    arm.to_file(str(OUT_XML))

    if OUT_ASSETS.exists():
        shutil.rmtree(OUT_ASSETS)
    OUT_ASSETS.mkdir(parents=True)
    for src_dir in (UR5E_DIR / "assets", GRIPPER_DIR / "assets"):
        for f in src_dir.iterdir():
            shutil.copy2(f, OUT_ASSETS / f.name)
    for f in EXTRA_ASSET_FILES:
        shutil.copy2(f, OUT_ASSETS / f.name)
    print(f"Copied {len(list(OUT_ASSETS.iterdir()))} files into {OUT_ASSETS}")
    print(f"Wrote {OUT_XML}")


if __name__ == "__main__":
    main()
