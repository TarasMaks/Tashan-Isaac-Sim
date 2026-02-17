# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim.sensors.physx import _range_sensor
from isaacsim.core.prims import RigidPrim
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import Articulation, SingleArticulation, XFormPrim

from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
import omni.kit.app as APP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
import rerun as rr


class ScenarioTemplate:
    def __init__(self):
        pass

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self):
        pass


"""
Scenario 4: Panda robot pick-and-place with dual Tashan TS-F-A tactile sensors.

Two TS-F-A sensors are placed at WORLD level (/World/TipLeft, /World/TipRight)
and attached to the Panda fingers via FixedJoints that target each sensor's
base_link (a real rigid body).  This avoids nested-rigid-body errors that
occur when sensors are parented under finger prims.

The sensors keep their full internal physics (base_link → tip_link → pad →
pad_1..pad_7 connected by joints).  Only the root_joint and
ArticulationRootAPI are removed so they don't conflict with the Panda
articulation.

Asset loading is split into two phases:
  Phase 1 (load_robot) — BEFORE World.reset.  Loads robot + sensors,
      removes sensor articulation roots, creates FixedJoints.
  Phase 2 (_build_sensor_scene) — AFTER World.reset.  Creates the
      cube and RigidPrim contact wrappers.
"""


class ExampleScenario(ScenarioTemplate):
    # Grasp state-machine phases
    PHASE_APPROACH = 0
    PHASE_LOWER = 1
    PHASE_GRASP = 2
    PHASE_LIFT = 3
    PHASE_HOLD = 4

    # Phase durations in physics steps (at 60 Hz)
    APPROACH_STEPS = 90       # 1.5 s
    LOWER_STEPS = 90          # 1.5 s
    GRASP_STEPS = 120         # 2.0 s
    LIFT_STEPS = 120          # 2.0 s

    # Channel labels for the 11-channel TS-F-A output
    CHANNEL_NAMES = [
        "Proximity",
        "Normal Force (N)",
        "Tangential Force (N)",
        "Tangential Direction (deg)",
        "Capacitance Ch1",
        "Capacitance Ch2",
        "Capacitance Ch3",
        "Capacitance Ch4",
        "Capacitance Ch5",
        "Capacitance Ch6",
        "Capacitance Ch7",
    ]

    # Prim paths — sensors at WORLD level (avoids nested-rigid-body issues).
    PANDA_PRIM_PATH = "/World/Panda"
    LEFT_FINGER_PATH = "/World/Panda/panda_leftfinger"
    RIGHT_FINGER_PATH = "/World/Panda/panda_rightfinger"
    LEFT_SENSOR_PATH = "/World/TipLeft"
    RIGHT_SENSOR_PATH = "/World/TipRight"
    CUBE_PRIM_PATH = "/World/TransparentCube"

    # Quaternion helpers  (w, x, y, z)  — Gf.Quatf(real, Gf.Vec3f(imag))
    _SQRT2_2 = 0.7071068

    def __init__(self):
        self._articulation = None
        self._panda_robot = None
        self._running_scenario = False
        self._time = 0.0
        self._phase = self.PHASE_APPROACH
        self._phase_step = 0

        # Per-finger sensor data
        self.sensorFrameDataLeft = []
        self.sensorFrameDataRight = []
        self.sensorBufferLeft = []
        self.sensorBufferRight = []

        self._load_register_sensor()

    # ------------------------------------------------------------------
    # Public lifecycle (called by UIBuilder)
    # ------------------------------------------------------------------
    def load_robot(self):
        """Phase 1: called from setup_scene_fn BEFORE World.reset.

        Loads the Franka robot, both Tashan sensors at world level,
        removes the sensors' articulation roots, and creates FixedJoints
        between each finger and the sensor's base_link (a real rigid body).
        """
        stage = get_current_stage()
        ts_usd = os.path.join(os.path.dirname(__file__), "../assets/TS-F-A.usd")

        # ---- 1. Load Franka robot ----
        panda_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        print(f"[Scenario4] Loading Panda from {panda_usd}")
        add_reference_to_stage(usd_path=panda_usd, prim_path=self.PANDA_PRIM_PATH)

        # ---- 2. Load sensors at WORLD level (no parenting) ----
        # No initial transform — the FixedJoint will snap them to the
        # correct position and orientation on the first physics step.
        add_reference_to_stage(usd_path=ts_usd, prim_path=self.LEFT_SENSOR_PATH)
        add_reference_to_stage(usd_path=ts_usd, prim_path=self.RIGHT_SENSOR_PATH)
        print(f"[Scenario4] Loaded Tashan sensors from {ts_usd}")

        # ---- 3. Disable the sensors' own articulation ----
        # Remove ArticulationRootAPI so the sensor is not its own
        # articulation (which would prevent cross-articulation joints).
        # Deactivate root_joint (sensor no longer fixed to the world).
        for sensor_path in [self.LEFT_SENSOR_PATH, self.RIGHT_SENSOR_PATH]:
            sensor_prim = stage.GetPrimAtPath(sensor_path)
            if sensor_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                sensor_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                print(f"[Scenario4] Removed ArticulationRootAPI from {sensor_path}")

            joint_path = f"{sensor_path}/root_joint"
            joint_prim = stage.GetPrimAtPath(joint_path)
            if joint_prim.IsValid():
                if joint_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    joint_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    print(f"[Scenario4] Removed ArticulationRootAPI from {joint_path}")
                joint_prim.SetActive(False)
                print(f"[Scenario4] Deactivated {joint_path}")

        # ---- 4. FixedJoints: finger ↔ sensor base_link ----
        # base_link is a real rigid body inside the sensor USD, so the
        # FixedJoint has valid body targets on both sides.
        #
        # localPos0 = fingertip offset in the finger's local frame.
        # localRot0 = rotation applied at the attachment point on the finger.
        #   The sensor pads face +Z in the sensor's default frame.
        #   To make the left-finger pads face inward (toward the right finger)
        #   we rotate the joint frame so the sensor's +Z maps to the
        #   finger's inward direction.
        #
        #   Rx(+90°) maps +Z → −Y.   Rx(−90°) maps +Z → +Y.
        #
        #   The Franka left finger is on the +Y side, so inward = −Y → Rx(+90°).
        #   The Franka right finger is on the −Y side, so inward = +Y → Rx(−90°).
        #
        # We try BOTH +90 and −90 — if user reports outward, swap them.
        sq = self._SQRT2_2
        rot_left  = Gf.Quatf(sq, Gf.Vec3f(sq, 0, 0))   # Rx(+90°)
        rot_right = Gf.Quatf(sq, Gf.Vec3f(-sq, 0, 0))   # Rx(−90°)

        self._create_fixed_joint(
            stage,
            joint_path="/World/FixedJointLeft",
            body0_path=self.LEFT_FINGER_PATH,
            body1_path=f"{self.LEFT_SENSOR_PATH}/base_link",
            local_pos0=Gf.Vec3f(0, 0, 0.04),
            local_rot0=rot_left,
        )
        self._create_fixed_joint(
            stage,
            joint_path="/World/FixedJointRight",
            body0_path=self.RIGHT_FINGER_PATH,
            body1_path=f"{self.RIGHT_SENSOR_PATH}/base_link",
            local_pos0=Gf.Vec3f(0, 0, 0.04),
            local_rot0=rot_right,
        )
        print("[Scenario4] FixedJoints created (finger ↔ sensor base_link)")

        # ---- 5. Create Articulation wrapper ----
        self._panda_robot = Articulation(prim_paths_expr=self.PANDA_PRIM_PATH, name="panda_robot")

    def setup_scenario(self, articulation, object_prim):
        """Phase 2: called from setup_post_load_fn AFTER World.reset."""
        self._build_sensor_scene()
        self._articulation = self._panda_robot
        self._running_scenario = True
        self._phase = self.PHASE_APPROACH
        self._phase_step = 0
        set_camera_view(eye=[1.0, 0.8, 0.8], target=[0.4, 0.0, 0.4])
        rr.init("tashan_panda_pick_demo", spawn=True)

    def teardown_scenario(self):
        self._time = 0.0
        self._articulation = None
        self._running_scenario = False
        self._phase = self.PHASE_APPROACH
        self._phase_step = 0
        self.sensorBufferLeft = []
        self.sensorBufferRight = []

    def update_scenario(self, step: float, step_ind: int):
        from register_sensor import TSsensor

        # Read both finger sensors
        self.sensorFrameDataLeft = TSsensor(self.tactile_left, self.range_left)
        self.sensorFrameDataRight = TSsensor(self.tactile_right, self.range_right)

        # Buffer every frame (no cap - STOP triggers plot)
        self.sensorBufferLeft.append(list(self.sensorFrameDataLeft))
        self.sensorBufferRight.append(list(self.sensorFrameDataRight))

        self._time += step
        self._execute_grasp_sequence(step_ind)
        self._update_rerun_visualization()

    def draw_data(self):
        """Generate matplotlib PNG plots for all 11 channels on both fingers."""
        if not self.sensorBufferLeft or not self.sensorBufferRight:
            return

        current_path = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(current_path, "../sensor_data")
        os.makedirs(data_dir, exist_ok=True)

        left = np.array(self.sensorBufferLeft)
        right = np.array(self.sensorBufferRight)
        x = np.arange(len(left))

        # --- Figure 1: Force & Proximity (both fingers) ---
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 1, 1)
        plt.title("Left Finger - Force & Proximity")
        plt.plot(x, left[:, 0], label="Proximity", linestyle="-")
        plt.plot(x, left[:, 1], label="Normal Force (N)", linestyle="-")
        plt.plot(x, left[:, 2], label="Tangential Force (N)", linestyle="--")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.title("Right Finger - Force & Proximity")
        plt.plot(x, right[:, 0], label="Proximity", linestyle="-")
        plt.plot(x, right[:, 1], label="Normal Force (N)", linestyle="-")
        plt.plot(x, right[:, 2], label="Tangential Force (N)", linestyle="--")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        path1 = os.path.join(data_dir, "panda_pick_forces.png")
        plt.savefig(path1, dpi=150)
        plt.close()
        print(f"Force plot saved to {path1}")

        # --- Figure 2: Capacitance channels (both fingers) ---
        if left.shape[1] >= 11:
            plt.figure(figsize=(14, 10))

            plt.subplot(2, 1, 1)
            plt.title("Left Finger - Capacitance Channels")
            for i in range(7):
                plt.plot(x, left[:, 4 + i], label=f"Ch {i + 1}")
            plt.xlabel("Time step")
            plt.ylabel("Capacitance")
            plt.legend(ncol=4)
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.title("Right Finger - Capacitance Channels")
            for i in range(7):
                plt.plot(x, right[:, 4 + i], label=f"Ch {i + 1}")
            plt.xlabel("Time step")
            plt.ylabel("Capacitance")
            plt.legend(ncol=4)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            path2 = os.path.join(data_dir, "panda_pick_capacitance.png")
            plt.savefig(path2, dpi=150)
            plt.close()
            print(f"Capacitance plot saved to {path2}")

        # --- Figure 3: All 11 channels side-by-side ---
        n_channels = min(left.shape[1], 11)
        fig, axes = plt.subplots(
            n_channels, 1,
            figsize=(14, 2.5 * n_channels),
            sharex=True,
        )
        fig.suptitle(
            "All Tashan Sensor Channels - Left vs Right Finger",
            fontsize=14,
        )

        for ch in range(n_channels):
            ax = axes[ch]
            ax.plot(x, left[:, ch], label="Left", linewidth=1)
            ax.plot(x, right[:, ch], label="Right", linewidth=1, linestyle="--")
            label = self.CHANNEL_NAMES[ch] if ch < len(self.CHANNEL_NAMES) else f"Ch {ch}"
            ax.set_ylabel(label, fontsize=8)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time step")
        plt.tight_layout()
        path3 = os.path.join(data_dir, "panda_pick_all_channels.png")
        plt.savefig(path3, dpi=150)
        plt.close()
        print(f"All-channels plot saved to {path3}")

    # ------------------------------------------------------------------
    # Grasp state machine
    # ------------------------------------------------------------------
    def _execute_grasp_sequence(self, step_ind):
        if self._articulation is None:
            return

        # Joint targets for the 7 arm DOFs
        approach_arm = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.4, 0.785])
        lower_arm = np.array([0.0, 0.15, 0.0, -1.35, 0.0, 1.87, 0.785])
        lift_arm = np.array([0.0, -0.6, 0.0, -1.8, 0.0, 2.4, 0.785])

        fingers_open = np.array([0.04, 0.04])
        fingers_closed = np.array([0.005, 0.005])

        self._phase_step += 1

        if self._phase == self.PHASE_APPROACH:
            targets = np.concatenate([approach_arm, fingers_open])
            self._articulation.apply_action(
                ArticulationAction(joint_positions=targets)
            )
            if self._phase_step >= self.APPROACH_STEPS:
                self._phase = self.PHASE_LOWER
                self._phase_step = 0

        elif self._phase == self.PHASE_LOWER:
            targets = np.concatenate([lower_arm, fingers_open])
            self._articulation.apply_action(
                ArticulationAction(joint_positions=targets)
            )
            if self._phase_step >= self.LOWER_STEPS:
                self._phase = self.PHASE_GRASP
                self._phase_step = 0

        elif self._phase == self.PHASE_GRASP:
            targets = np.concatenate([lower_arm, fingers_closed])
            self._articulation.apply_action(
                ArticulationAction(joint_positions=targets)
            )
            if self._phase_step >= self.GRASP_STEPS:
                self._phase = self.PHASE_LIFT
                self._phase_step = 0

        elif self._phase == self.PHASE_LIFT:
            targets = np.concatenate([lift_arm, fingers_closed])
            self._articulation.apply_action(
                ArticulationAction(joint_positions=targets)
            )
            if self._phase_step >= self.LIFT_STEPS:
                self._phase = self.PHASE_HOLD
                self._phase_step = 0

        elif self._phase == self.PHASE_HOLD:
            targets = np.concatenate([lift_arm, fingers_closed])
            self._articulation.apply_action(
                ArticulationAction(joint_positions=targets)
            )

    # ------------------------------------------------------------------
    # Sensor library loader (shared with other scenarios)
    # ------------------------------------------------------------------
    def _load_register_sensor(self):
        try:
            version = APP.get_app().get_app_version()
            print("Isaac Sim Version:", version)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if version not in ("4.5.0", "5.0.0"):
                print("TaShan sensor not supported on version", version)

            ts_lib_path = os.path.join(current_dir, "ts_sensor_lib", "isaac-sim-" + version)
            if ts_lib_path not in sys.path:
                sys.path.insert(0, ts_lib_path)

            try:
                from register_sensor import TSsensor
                print("TS sensor callback registered successfully via TSensor module")
            except ImportError as e:
                print(f"Failed to import TSensor module from {ts_lib_path}: {e}")

        except Exception as e:
            print(f"Failed to initialize TS sensor callback: {e}")
            return False

    # ------------------------------------------------------------------
    # Rerun telemetry
    # ------------------------------------------------------------------
    def _update_rerun_visualization(self):
        # Left finger - all 11 channels
        rr.log("left/proximity", rr.Scalar(self.sensorFrameDataLeft[0]))
        rr.log("left/force_normal", rr.Scalar(self.sensorFrameDataLeft[1]))
        rr.log("left/force_tangential", rr.Scalar(self.sensorFrameDataLeft[2]))
        rr.log("left/tangential_dir", rr.Scalar(self.sensorFrameDataLeft[3]))
        if len(self.sensorFrameDataLeft) >= 11:
            for i in range(7):
                rr.log(f"left/capacitance/ch_{i + 1}", rr.Scalar(self.sensorFrameDataLeft[4 + i]))

        # Right finger - all 11 channels
        rr.log("right/proximity", rr.Scalar(self.sensorFrameDataRight[0]))
        rr.log("right/force_normal", rr.Scalar(self.sensorFrameDataRight[1]))
        rr.log("right/force_tangential", rr.Scalar(self.sensorFrameDataRight[2]))
        rr.log("right/tangential_dir", rr.Scalar(self.sensorFrameDataRight[3]))
        if len(self.sensorFrameDataRight) >= 11:
            for i in range(7):
                rr.log(f"right/capacitance/ch_{i + 1}", rr.Scalar(self.sensorFrameDataRight[4 + i]))

        # Grasp phase indicator
        rr.log("grasp/phase", rr.Scalar(self._phase))

    # ------------------------------------------------------------------
    # FixedJoint helper
    # ------------------------------------------------------------------
    @staticmethod
    def _create_fixed_joint(stage, joint_path, body0_path, body1_path,
                            local_pos0=Gf.Vec3f(0), local_pos1=Gf.Vec3f(0),
                            local_rot0=None, local_rot1=None):
        """Create a USD FixedJoint between two rigid bodies."""
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
        joint.CreateLocalPos0Attr().Set(local_pos0)
        joint.CreateLocalPos1Attr().Set(local_pos1)
        if local_rot0 is not None:
            joint.CreateLocalRot0Attr().Set(local_rot0)
        if local_rot1 is not None:
            joint.CreateLocalRot1Attr().Set(local_rot1)
        return joint

    # ------------------------------------------------------------------
    # Scene assembly (phase 2 — after World.reset)
    # ------------------------------------------------------------------
    def _build_sensor_scene(self):
        """Create the transparent cube and RigidPrim wrappers for sensor
        pads.  All USD loading already happened in load_robot (phase 1)
        so the physics system has processed the rigid bodies."""

        # ---- 1. Create transparent cube ----
        cube_size = 0.04  # 4 cm
        self.cube = DynamicCuboid(
            prim_path=self.CUBE_PRIM_PATH,
            name="transparent_cube",
            position=np.array([0.4, 0.0, cube_size / 2 + 0.001]),
            scale=np.array([cube_size, cube_size, cube_size]),
            color=np.array([0.2, 0.6, 1.0]),
            mass=0.1,
        )
        self._make_prim_transparent(self.CUBE_PRIM_PATH, opacity=0.35)

        # ---- 2. Set up tactile sensor interfaces ----
        cube_filter = [self.CUBE_PRIM_PATH]

        self.range_left = f"{self.LEFT_SENSOR_PATH}/pad_4/LightBeam_Sensor"
        self.tactile_left = RigidPrim(
            prim_paths_expr=f"{self.LEFT_SENSOR_PATH}/pad_[1-7]",
            name="finger_tactile_left",
            contact_filter_prim_paths_expr=cube_filter,
            max_contact_count=7 * 5,
        )

        self.range_right = f"{self.RIGHT_SENSOR_PATH}/pad_4/LightBeam_Sensor"
        self.tactile_right = RigidPrim(
            prim_paths_expr=f"{self.RIGHT_SENSOR_PATH}/pad_[1-7]",
            name="finger_tactile_right",
            contact_filter_prim_paths_expr=cube_filter,
            max_contact_count=7 * 5,
        )

    # ------------------------------------------------------------------
    # Transparent material helper
    # ------------------------------------------------------------------
    @staticmethod
    def _make_prim_transparent(prim_path, opacity=0.35):
        """Create and bind a translucent UsdPreviewSurface material."""
        stage = get_current_stage()

        mat_path = prim_path + "/TransparentMat"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.6, 1.0))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        prim = stage.GetPrimAtPath(prim_path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
