"""Franka Deformable Demo with Tashan TS-F-A Tactile Sensors

Extends the standard FrankaDeformableDemo to integrate Tashan TS-F-A
tactile sensors on the Franka robot gripper fingers, with real-time Rerun
visualization of force, proximity, and capacitance data.

Each Franka gripper gets two TS-F-A sensors (left and right finger),
providing per-finger:
  - Proximity sensing (index 0)
  - Normal force in N (index 1)
  - Tangential force in N (index 2)
  - Tangential force direction in degrees (index 3)
  - 7-channel raw capacitance (indices 4-10)

Sensor data is streamed to Rerun for live monitoring. When the native
TSsensor library is unavailable, the demo falls back to reading net
contact forces from the PhysX tensor API rigid body views.

Usage:
    Register this module as a PhysX demo, or import and instantiate
    FrankaDeformableSensorDemo directly within an Isaac Sim session.
"""

import carb
import math
import numpy as np
import omni
from pxr import Gf, Sdf, UsdGeom, UsdShade, PhysxSchema, UsdPhysics
import omni.physxdemos as demo
from omni.physx.scripts import deformableUtils, utils, physicsUtils
from omni.physx.scripts.assets_paths import AssetFolders
import omni.physx.bindings._physx as physx_settings_bindings
from omni.physxdemos.utils import franka_helpers
from omni.physxdemos.utils import numpy_utils
import os
import sys

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("[FrankaDeformableSensorDemo] rerun-sdk not installed, visualization disabled")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Load Tashan native sensor library (best-effort)
# ---------------------------------------------------------------------------
_ts_sensor_loaded = False
try:
    import omni.kit.app as _APP
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _isaac_version = _APP.get_app().get_app_version()
    _ts_lib_path = os.path.join(_current_dir, "ts_sensor_lib", "isaac-sim-" + _isaac_version)
    if _ts_lib_path not in sys.path:
        sys.path.insert(0, _ts_lib_path)
    from register_sensor import TSsensor
    _ts_sensor_loaded = True
    print(f"[FrankaDeformableSensorDemo] TSsensor loaded (Isaac Sim {_isaac_version})")
except Exception as _e:
    print(f"[FrankaDeformableSensorDemo] TSsensor not available: {_e}")

_RIGIDPRIM_AVAILABLE = False
try:
    from isaacsim.core.prims import RigidPrim
    _RIGIDPRIM_AVAILABLE = True
except ImportError:
    pass

deformable_beta_on = carb.settings.get_settings().get_as_bool(
    physx_settings_bindings.SETTING_ENABLE_DEFORMABLE_BETA
)


# ---------------------------------------------------------------------------
# Helper functions (same as original FrankaDeformableDemo)
# ---------------------------------------------------------------------------

def orientation_error(desired, current):
    cc = numpy_utils.quat_conjugate(current)
    q_r = numpy_utils.quat_mul(desired, cc)
    return q_r[:, 0:3] * np.sign(q_r[:, 3])[:, None]


def cube_grasping_yaw(q, corners):
    """Returns horizontal rotation required to grasp cube."""
    rc = numpy_utils.quat_rotate(q, corners)
    yaw = (np.arctan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = np.cos(theta)
    x = np.zeros_like(w)
    y = np.zeros_like(w)
    z = np.sin(theta)
    yaw_quats = np.stack([x, y, z, w], axis=-1)
    return yaw_quats


# ---------------------------------------------------------------------------
# TashanFingerSensorManager
# ---------------------------------------------------------------------------

class TashanFingerSensorManager:
    """Manages Tashan TS-F-A sensors attached to Franka gripper fingers.

    Responsibilities:
      - Attach TS-F-A sensor USD models to left/right finger prims on stage
      - Create tensor-API rigid body views for reading finger contact forces
      - Optionally create RigidPrim objects for the native TSsensor callback
      - Initialize and update Rerun visualisation streams
      - Store per-finger data buffers for matplotlib plotting
    """

    SENSOR_CHANNELS = 11  # proximity, normal, tangential, direction, cap1..cap7

    def __init__(self, num_envs, sensor_usd_path):
        self.num_envs = num_envs
        self.sensor_usd_path = sensor_usd_path

        # Per-finger sensor data: key = (env_idx, "left"|"right")
        self.sensor_data = {}
        self._tactile_prims = {}
        self._range_paths = {}

        # Tensor API views (set later)
        self.left_fingers = None
        self.right_fingers = None

        self._step_count = 0
        self._rerun_initialized = False

        # Rolling data buffers for plotting
        self.sensor_buffers = {}
        self.max_buffer_size = 1000

    # ------------------------------------------------------------------
    # Stage setup
    # ------------------------------------------------------------------

    def attach_sensors_to_stage(self, stage, envs_scope_path):
        """Attach TS-F-A sensor USD models to every Franka finger on the stage.

        The sensor is referenced as a child of each finger prim and positioned
        on the inner face so that its tactile pads face the opposing finger.
        """
        for i in range(self.num_envs):
            env_path = envs_scope_path.AppendChild(f"env_{i}")
            for side in ("left", "right"):
                self._attach_single_sensor(stage, env_path, side)

    def _attach_single_sensor(self, stage, env_path, side):
        """Attach one TS-F-A sensor to a single finger."""
        finger_name = f"panda_{side}finger"
        sensor_path = env_path.AppendPath(f"franka/{finger_name}/ts_sensor")

        sensor_prim = stage.DefinePrim(sensor_path)
        sensor_prim.GetReferences().AddReference(self.sensor_usd_path)

        sensor_xform = UsdGeom.Xform.Define(stage, sensor_path)

        # Position on inner face of finger tip.
        # The Franka USD (at 0.01 world scale) has finger geometry in local
        # centimetre-ish coords.  The translate and orient values below place
        # the sensor pad facing the opposing finger.
        if side == "left":
            sensor_xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.6, 3.8))
            orient = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0).GetQuat())
        else:
            sensor_xform.AddTranslateOp().Set(Gf.Vec3f(0.0, -0.6, 3.8))
            orient = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90.0).GetQuat())

        sensor_xform.AddOrientOp().Set(orient)
        sensor_xform.AddScaleOp().Set(Gf.Vec3f(0.4, 0.4, 0.4))

    # ------------------------------------------------------------------
    # Tensor / physics initialisation (called from on_tensor_start)
    # ------------------------------------------------------------------

    def init_tensor_views(self, sim):
        """Create rigid body views so we can query finger contact forces."""
        self.left_fingers = sim.create_rigid_body_view(
            "/World/envs/*/franka/panda_leftfinger"
        )
        self.right_fingers = sim.create_rigid_body_view(
            "/World/envs/*/franka/panda_rightfinger"
        )

    def init_rigid_prim_sensors(self):
        """Attempt to create RigidPrim objects for the native TSsensor callback.

        This is best-effort: if the isaacsim.core.prims.RigidPrim import or
        the prim-path expression fails (e.g. the TS-F-A USD structure does
        not match expectations), the demo falls back to tensor-API forces.
        """
        if not (_RIGIDPRIM_AVAILABLE and _ts_sensor_loaded):
            return

        for i in range(self.num_envs):
            for side in ("left", "right"):
                finger_name = f"panda_{side}finger"
                sensor_base = f"/World/envs/env_{i}/franka/{finger_name}/ts_sensor"
                try:
                    tactile = RigidPrim(
                        prim_paths_expr=f"{sensor_base}/pad_[1-7]",
                        name=f"tactile_{side}_{i}",
                        contact_filter_prim_paths_expr=[
                            f"/World/envs/env_{i}/box",
                        ],
                        max_contact_count=7 * 5,
                    )
                    range_path = f"{sensor_base}/pad_4/LightBeam_Sensor"
                    self._tactile_prims[(i, side)] = tactile
                    self._range_paths[(i, side)] = range_path
                except Exception as exc:
                    print(
                        f"[TashanSensor] RigidPrim creation failed for env {i} "
                        f"{side}: {exc}  (falling back to tensor API)"
                    )

    # ------------------------------------------------------------------
    # Rerun
    # ------------------------------------------------------------------

    def init_rerun(self):
        """Spawn a Rerun viewer and open a recording stream."""
        if not RERUN_AVAILABLE:
            return
        rr.init("franka_deformable_tactile", spawn=True)
        self._rerun_initialized = True
        print("[FrankaDeformableSensorDemo] Rerun viewer spawned")

    # ------------------------------------------------------------------
    # Per-step reading
    # ------------------------------------------------------------------

    def read_sensors(self):
        """Read sensor data from every finger, every environment."""
        self._step_count += 1

        # Tensor API net contact forces (always available as fallback)
        try:
            left_forces = self.left_fingers.get_net_contact_forces()
            right_forces = self.right_fingers.get_net_contact_forces()
        except Exception:
            # If contact forces are unavailable, use zero arrays
            left_forces = np.zeros((self.num_envs, 3), dtype=np.float32)
            right_forces = np.zeros((self.num_envs, 3), dtype=np.float32)

        for i in range(self.num_envs):
            for side in ("left", "right"):
                forces = left_forces[i] if side == "left" else right_forces[i]
                key = (i, side)

                # --- Try native TSsensor first (full 11-channel) -----------
                if key in self._tactile_prims and _ts_sensor_loaded:
                    try:
                        raw = TSsensor(
                            self._tactile_prims[key],
                            self._range_paths[key],
                        )
                        self.sensor_data[key] = np.array(raw, dtype=np.float32)
                        self._buffer_data(key)
                        continue
                    except Exception:
                        pass  # fall through to tensor-API path

                # --- Fallback: derive channels from tensor API forces ------
                force_normal = float(np.abs(forces[2]))
                force_tangential = float(
                    np.sqrt(forces[0] ** 2 + forces[1] ** 2)
                )
                force_direction = (
                    float(np.degrees(np.arctan2(forces[1], forces[0]))) % 360.0
                )
                force_mag = float(np.linalg.norm(forces))

                data = np.zeros(self.SENSOR_CHANNELS, dtype=np.float32)
                data[0] = 0.0                     # proximity (unavailable)
                data[1] = force_normal
                data[2] = force_tangential
                data[3] = force_direction
                # Distribute magnitude across 7 capacitance channels
                for ch in range(7):
                    data[4 + ch] = force_mag / 7.0

                self.sensor_data[key] = data
                self._buffer_data(key)

    def _buffer_data(self, key):
        buf = self.sensor_buffers.setdefault(key, [])
        buf.append(self.sensor_data[key].copy())
        if len(buf) > self.max_buffer_size:
            buf.pop(0)

    # ------------------------------------------------------------------
    # Rerun logging
    # ------------------------------------------------------------------

    def update_rerun(self):
        """Log current sensor data to Rerun for every finger."""
        if not self._rerun_initialized:
            return

        for i in range(self.num_envs):
            for side in ("left", "right"):
                key = (i, side)
                if key not in self.sensor_data:
                    continue

                data = self.sensor_data[key]
                prefix = f"franka/env_{i}/{side}_finger"

                rr.log(f"{prefix}/force_normal", rr.Scalar(float(data[1])))
                rr.log(f"{prefix}/force_tangential", rr.Scalar(float(data[2])))
                rr.log(f"{prefix}/proximity", rr.Scalar(float(data[0])))
                rr.log(f"{prefix}/force_direction", rr.Scalar(float(data[3])))

                if len(data) >= self.SENSOR_CHANNELS:
                    for ch in range(7):
                        rr.log(
                            f"{prefix}/capacitance/ch_{ch + 1}",
                            rr.Scalar(float(data[4 + ch])),
                        )

                total_force = float(np.sqrt(data[1] ** 2 + data[2] ** 2))
                rr.log(f"{prefix}/total_force", rr.Scalar(total_force))

        # Convenience: combined grip force for env 0
        l_key = (0, "left")
        r_key = (0, "right")
        if l_key in self.sensor_data and r_key in self.sensor_data:
            grip = float(
                self.sensor_data[l_key][1] + self.sensor_data[r_key][1]
            )
            rr.log("franka/env_0/grip_force_sum", rr.Scalar(grip))

    # ------------------------------------------------------------------
    # Matplotlib output
    # ------------------------------------------------------------------

    def draw_data(self, env_idx=0):
        """Generate a matplotlib plot of buffered sensor data for *env_idx*."""
        if not MATPLOTLIB_AVAILABLE:
            print("[TashanSensor] matplotlib not available, skipping plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        for side_idx, side in enumerate(("left", "right")):
            key = (env_idx, side)
            buf = self.sensor_buffers.get(key)
            if not buf:
                continue

            arr = np.array(buf)
            steps = np.arange(len(arr))
            ax = axes[side_idx]
            ax.set_title(f"Env {env_idx} – {side.capitalize()} Finger Sensor")
            ax.plot(steps, arr[:, 1], label="Normal Force (N)", linestyle="-")
            ax.plot(steps, arr[:, 2], label="Tangential Force (N)", linestyle="--")
            ax.plot(steps, arr[:, 0], label="Proximity", linestyle="-.")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Physics Step")
        fig.tight_layout()

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sensor_data")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"franka_finger_sensors_env{env_idx}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[TashanSensor] Plot saved to {out_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        self._tactile_prims.clear()
        self._range_paths.clear()
        self.sensor_data.clear()
        self.sensor_buffers.clear()
        self.left_fingers = None
        self.right_fingers = None


# ---------------------------------------------------------------------------
# FrankaDeformableSensorDemo
# ---------------------------------------------------------------------------

class FrankaDeformableSensorDemo(demo.AsyncDemoBase):
    """Franka deformable-object grasping demo with Tashan TS-F-A finger sensors.

    This is a drop-in extension of the standard FrankaDeformableDemo.  All
    original grasping behaviour is preserved; the additions are:

      * TS-F-A sensor USD models referenced onto each gripper finger
      * Per-step sensor data acquisition (native TSsensor or tensor-API fallback)
      * Real-time Rerun visualisation of force / proximity / capacitance
      * Matplotlib plot export on shutdown
    """

    title = "Franka Deformable with Tashan Sensors"
    category = demo.Categories.COMPLEX_SHOWCASES
    short_description = (
        "Franka arms with Tashan TS-F-A tactile sensors grasping deformable cubes"
    )
    description = (
        "The Franka robot arms use Jacobian-based inverse kinematics to lift "
        "and drop a red deformable cube in a bowl of green deformable cubes.  "
        "Tashan TS-F-A tactile sensors on each gripper finger provide real-time "
        "force, proximity, and capacitance feedback, visualised in Rerun."
    )

    kit_settings = {
        "persistent/app/viewport/displayOptions": demo.get_viewport_minimal_display_options_int(),
    }

    params = {
        "Num_Frankas": demo.IntParam(4, 1, 20, 1),
        "Num_Greens": demo.IntParam(9, 0, 9, 1),
    }

    demo_camera = Sdf.Path("/World/Camera")

    def __init__(self):
        super().__init__(enable_tensor_api=True, enable_fabric=False)
        self._reset_hydra_instancing_on_shutdown = False
        self.loop = True

        self.asset_paths = {
            "franka": demo.get_demo_asset_path(
                AssetFolders.FRANKA_DEFORMABLE,
                "SubUSDs/Franka/franka_alt_fingers.usd",
            ),
            "table": demo.get_demo_asset_path(
                AssetFolders.FRANKA_DEFORMABLE,
                "SubUSDs/Table_Rounded/Table_Rounded.usd",
            ),
            "stool": demo.get_demo_asset_path(
                AssetFolders.FRANKA_DEFORMABLE,
                "SubUSDs/Franka_Stool/Franka_Stool.usd",
            ),
            "cube": demo.get_demo_asset_path(
                AssetFolders.FRANKA_DEFORMABLE, "SubUSDs/box_high.usd"
            ),
            "bowl": demo.get_demo_asset_path(
                AssetFolders.FRANKA_DEFORMABLE, "SubUSDs/bowl_plate.usd"
            ),
            "cube_materials": demo.get_demo_asset_path(
                AssetFolders.FRANKA_DEFORMABLE,
                "SubUSDs/deformable_cubes_materials.usd",
            ),
        }
        self.demo_base_usd_url = demo.get_demo_asset_path(
            AssetFolders.FRANKA_DEFORMABLE, "StagingDeformable.usd"
        )

        # Tashan sensor asset
        self._ts_sensor_usd = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "assets", "TS-F-A.usd"
        )
        self._sensor_manager = None

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def on_startup(self):
        sceneGraphInstancingEnabled = carb.settings.get_settings().get(
            "/persistent/omnihydra/useSceneGraphInstancing"
        )
        if not sceneGraphInstancingEnabled:
            carb.settings.get_settings().set(
                "/persistent/omnihydra/useSceneGraphInstancing", True
            )
            self._reset_hydra_instancing_on_shutdown = True

    # ------------------------------------------------------------------
    # Deformable cube creation (unchanged from original)
    # ------------------------------------------------------------------

    def create_jello_cube(
        self,
        stage,
        path,
        name,
        position,
        size,
        mesh_path,
        phys_material_path,
        grfx_material,
    ):
        if deformable_beta_on:
            xform_path = path.AppendChild(name)
            xform = UsdGeom.Xform.Define(stage, xform_path)
            skinMesh_path = xform_path.AppendChild("mesh")
            stage.DefinePrim(skinMesh_path).GetReferences().AddReference(mesh_path)
            skinMesh = UsdGeom.Mesh.Define(stage, skinMesh_path)
            skinMesh.AddTranslateOp().Set(position)
            skinMesh.AddOrientOp().Set(Gf.Quatf(1.0))
            skinMesh.AddScaleOp().Set(Gf.Vec3f(size, size, size))

            simMesh_path = xform_path.AppendChild("simMesh")
            collMesh_path = xform_path.AppendChild("collMesh")

            deformableUtils.create_auto_volume_deformable_hierarchy(
                stage,
                root_prim_path=xform_path,
                simulation_tetmesh_path=simMesh_path,
                collision_tetmesh_path=collMesh_path,
                cooking_src_mesh_path=skinMesh_path,
                simulation_hex_mesh_enabled=True,
                cooking_src_simplification_enabled=True,
                set_visibility_with_guide_purpose=True,
            )
            xform.GetPrim().GetAttribute("physxDeformableBody:resolution").Set(3)
            xform.GetPrim().ApplyAPI("PhysxBaseDeformableBodyAPI")
            xform.GetPrim().GetAttribute("physxDeformableBody:selfCollision").Set(False)
            xform.GetPrim().GetAttribute(
                "physxDeformableBody:solverPositionIterationCount"
            ).Set(self.pos_iterations)

            collMeshPrim = stage.GetPrimAtPath(collMesh_path)
            physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(collMeshPrim)
            physxCollisionAPI.GetContactOffsetAttr().Set(0.02)
            physxCollisionAPI.CreateRestOffsetAttr().Set(0.001)

            physicsUtils.add_physics_material_to_prim(
                stage, xform.GetPrim(), phys_material_path
            )
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=xform_path,
                material_path=grfx_material,
                strength=None,
            )
        else:
            cube_path = path.AppendChild(name)
            stage.DefinePrim(cube_path).GetReferences().AddReference(mesh_path)
            skinMesh = UsdGeom.Mesh.Define(stage, cube_path)
            skinMesh.AddTranslateOp().Set(position)
            skinMesh.AddOrientOp().Set(Gf.Quatf(1.0))
            skinMesh.AddScaleOp().Set(Gf.Vec3f(size, size, size))
            deformableUtils.add_physx_deformable_body(
                stage,
                cube_path,
                simulation_hexahedral_resolution=3,
                collision_simplification=True,
                self_collision=False,
                solver_position_iteration_count=self.pos_iterations,
            )
            physicsUtils.add_physics_material_to_prim(
                stage, skinMesh.GetPrim(), phys_material_path
            )
            physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(skinMesh.GetPrim())
            physxCollisionAPI.GetContactOffsetAttr().Set(0.02)
            physxCollisionAPI.CreateRestOffsetAttr().Set(0.001)
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=cube_path,
                material_path=grfx_material,
                strength=None,
            )

    # ------------------------------------------------------------------
    # Scene creation
    # ------------------------------------------------------------------

    def create(self, stage, Num_Frankas, Num_Greens):  # noqa: N803 – matches base API
        self.defaultPrimPath = stage.GetDefaultPrim().GetPath()
        self.stage = stage
        self.num_envs = Num_Frankas
        self.num_greens = Num_Greens

        # --- Physics scene -------------------------------------------------
        scene = UsdPhysics.Scene.Define(
            stage, self.defaultPrimPath.AppendChild("physicsScene")
        )
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.807)
        utils.set_physics_scene_asyncsimrender(scene.GetPrim(), False)
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physxSceneAPI.CreateFrictionOffsetThresholdAttr().Set(0.001)
        physxSceneAPI.CreateFrictionCorrelationDistanceAttr().Set(0.0005)
        physxSceneAPI.CreateGpuTotalAggregatePairsCapacityAttr().Set(10 * 1024)
        physxSceneAPI.CreateGpuFoundLostPairsCapacityAttr().Set(10 * 1024)
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(64 * 1024 * 1024)
        physxSceneAPI.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(10 * 1024)
        self.pos_iterations = 20
        self.vel_iterations = 1
        physxSceneAPI.GetMaxPositionIterationCountAttr().Set(self.pos_iterations)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Ground plane
        utils.addPlaneCollider(stage, "/World/physicsGroundPlaneCollider", "Z")

        # --- Deformable materials ------------------------------------------
        deformable_material_path = omni.usd.get_stage_next_free_path(
            stage, "/DeformableBodyMaterial", True
        )
        deformable_material2_path = omni.usd.get_stage_next_free_path(
            stage, "/DeformableBodyMaterial2", True
        )

        if deformable_beta_on:
            deformableUtils.add_deformable_material(
                stage,
                deformable_material_path,
                youngs_modulus=10000000.0,
                poissons_ratio=0.499,
                dynamic_friction=1.0,
                density=300.0,
            )
            mat_prim = stage.GetPrimAtPath(deformable_material_path)
            mat_prim.ApplyAPI("PhysxDeformableMaterialAPI")
            mat_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(
                0.0001
            )

            deformableUtils.add_deformable_material(
                stage,
                deformable_material2_path,
                youngs_modulus=4000000.0,
                poissons_ratio=0.499,
                dynamic_friction=0.05,
                density=100.0,
            )
            mat2_prim = stage.GetPrimAtPath(deformable_material2_path)
            mat2_prim.ApplyAPI("PhysxDeformableMaterialAPI")
            mat2_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(
                0.005
            )
        else:
            deformableUtils.add_deformable_body_material(
                stage,
                deformable_material_path,
                youngs_modulus=10000000.0,
                poissons_ratio=0.499,
                damping_scale=0.0,
                elasticity_damping=0.0001,
                dynamic_friction=1.0,
                density=300,
            )
            deformableUtils.add_deformable_body_material(
                stage,
                deformable_material2_path,
                youngs_modulus=4000000.0,
                poissons_ratio=0.499,
                damping_scale=0.0,
                elasticity_damping=0.005,
                dynamic_friction=0.05,
                density=100,
            )

        # Franka finger friction material
        mat_prim = UsdShade.Material.Define(
            stage, "/World/FrankaFingerPhysicsMaterial"
        )
        finger_material = UsdPhysics.MaterialAPI.Apply(mat_prim.GetPrim())
        finger_material.CreateStaticFrictionAttr().Set(1.0)
        finger_material.CreateDynamicFrictionAttr().Set(1.0)

        # --- Layout parameters ---------------------------------------------
        table_scale = 0.8
        table_height = 0.762 * table_scale
        table_position = Gf.Vec3f(0.5, 0.0, 0.0)
        env_spacing = 2
        num_envs_per_side = max(1, round(math.sqrt(self.num_envs)))
        row_half_length = (num_envs_per_side - 1) * env_spacing * 0.5

        self.box_size = 0.05
        box_loc = Gf.Vec3f(table_position) + Gf.Vec3f(
            0.0, 0.0, table_height + self.box_size * 3.5
        )

        # --- Camera --------------------------------------------------------
        cam = UsdGeom.Camera.Define(stage, self.demo_camera)
        cam_height = 1.1
        location = Gf.Vec3f(row_half_length + 6.5, row_half_length + 4.0, cam_height)
        target = Gf.Vec3f(0.0, 0.0, cam_height - 0.4)
        delta = target - location
        rotZ = math.degrees(math.atan2(-delta[0], delta[1]))
        rotX = math.degrees(
            math.atan2(delta[2], math.sqrt(delta[0] ** 2.0 + delta[1] ** 2.0))
        )
        rotZQ = Gf.Quatf(Gf.Rotation(Gf.Vec3d([0, 0, 1]), rotZ).GetQuat())
        rotXQ = Gf.Quatf(Gf.Rotation(Gf.Vec3d([1, 0, 0]), rotX + 90).GetQuat())
        physicsUtils.setup_transform_as_scale_orient_translate(cam)
        physicsUtils.set_or_add_translate_op(cam, translate=location)
        physicsUtils.set_or_add_orient_op(cam, orient=rotZQ * rotXQ)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))

        # --- Deformable cube materials (visual) ----------------------------
        looks_path = self.defaultPrimPath.AppendChild("Looks")
        stage.DefinePrim(looks_path).GetReferences().AddReference(
            self.asset_paths["cube_materials"], "/Looks"
        )
        green_glass_path = looks_path.AppendChild("GreenGlass")
        red_glass_path = looks_path.AppendChild("RedGlass")

        # --- Franka drive parameters ---------------------------------------
        self.default_dof_pos, drive_params = self.get_franka_parameters()

        # --- Create environments -------------------------------------------
        envsScopePath = self.defaultPrimPath.AppendPath("envs")
        UsdGeom.Scope.Define(stage, envsScopePath)

        for i in range(self.num_envs):
            col_number = i % num_envs_per_side
            row_number = i // num_envs_per_side
            x_pos = -row_half_length + col_number * env_spacing
            y_pos = -row_half_length + row_number * env_spacing
            env_xform = UsdGeom.Xform.Define(
                stage, envsScopePath.AppendChild(f"env_{i}")
            )
            env_xform.AddTranslateOp().Set(Gf.Vec3f(x_pos, y_pos, 0.0))

            # -- Franka stool --
            franka_stool_position = Gf.Vec3f(-0.027, 0.0, -0.001)
            stool_path = env_xform.GetPath().AppendChild("franka_stool")
            stage.DefinePrim(stool_path).GetReferences().AddReference(
                self.asset_paths["stool"]
            )
            stool_xform = UsdGeom.Xform.Get(stage, stool_path)
            utils.setStaticCollider(
                stool_xform.GetPrim(), approximationShape="boundingCube"
            )
            assert stool_xform
            physicsUtils.set_or_add_translate_op(
                stool_xform, translate=franka_stool_position
            )

            # -- Franka --
            franka_path = env_xform.GetPath().AppendChild("franka")
            assert stage.DefinePrim(franka_path).GetReferences().AddReference(
                self.asset_paths["franka"], "/panda"
            )
            franka_xform = UsdGeom.Xform.Get(stage, franka_path)
            assert franka_xform
            physicsUtils.set_or_add_translate_op(
                franka_xform, translate=Gf.Vec3f(0.0, 0.0, table_height - 0.4)
            )
            physicsUtils.set_or_add_scale_op(
                franka_xform, scale=Gf.Vec3f(0.01)
            )
            franka_helpers.configure_franka_drives(
                stage, franka_path, self.default_dof_pos, drive_params
            )
            physxArticulationAPI = PhysxSchema.PhysxArticulationAPI.Apply(
                franka_xform.GetPrim()
            )
            physxArticulationAPI.GetSolverPositionIterationCountAttr().Set(
                self.pos_iterations
            )
            physxArticulationAPI.GetSolverVelocityIterationCountAttr().Set(
                self.vel_iterations
            )

            # Finger friction binding
            for finger in ("panda_leftfinger", "panda_rightfinger"):
                geom_path = f"{env_xform.GetPath()}/franka/{finger}/geometry"
                prim = stage.GetPrimAtPath(geom_path)
                bindingAPI = UsdShade.MaterialBindingAPI.Apply(prim)
                bindingAPI.Bind(
                    mat_prim, UsdShade.Tokens.weakerThanDescendants, "physics"
                )

            # Finger joint velocity limits
            for joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
                joint = PhysxSchema.PhysxJointAPI.Get(
                    stage,
                    f"{env_xform.GetPath()}/franka/panda_hand/{joint_name}",
                )
                joint.GetMaxJointVelocityAttr().Set(1)

            # -- Table --
            table_path = env_xform.GetPath().AppendPath("table")
            assert stage.DefinePrim(table_path).GetReferences().AddReference(
                self.asset_paths["table"]
            )
            xform = UsdGeom.Xform.Get(stage, table_path)
            utils.setStaticCollider(
                xform.GetPrim(), approximationShape="boundingCube"
            )
            physicsUtils.set_or_add_translate_op(xform, translate=table_position)
            physicsUtils.set_or_add_scale_op(
                xform, scale=Gf.Vec3f(table_scale)
            )

            # -- Bowl --
            bowl_path = env_xform.GetPath().AppendPath("bowl")
            assert stage.DefinePrim(bowl_path).GetReferences().AddReference(
                self.asset_paths["bowl"], "/World/bowl_plate"
            )
            xform = UsdGeom.Xform.Get(stage, bowl_path)
            xform.AddTranslateOp().Set(
                table_position + Gf.Vec3f(0.0, 0.0, table_height)
            )
            rigidbody_api = PhysxSchema.PhysxRigidBodyAPI.Apply(
                stage.GetPrimAtPath(bowl_path.AppendChild("bowl_plate"))
            )
            rigidbody_api.CreateSolverPositionIterationCountAttr(self.pos_iterations)
            rigidbody_api.CreateSolverVelocityIterationCountAttr(self.vel_iterations)

            # -- Red deformable cube --
            self.create_jello_cube(
                stage,
                env_xform.GetPath(),
                "box",
                box_loc,
                self.box_size,
                self.asset_paths["cube"],
                deformable_material_path,
                red_glass_path,
            )

            # -- Green extras --
            extras_path = env_xform.GetPath().AppendChild("extras")
            ex_spacing = 1.5
            for z in range(3):
                for y in range(3):
                    for x in range(3):
                        if x == 1 & y == 1 & z == 1:
                            continue
                        j = x + y * 3 + z * 3 * 3
                        if j > 13:
                            j -= 1
                        if j >= self.num_greens:
                            break
                        self.create_jello_cube(
                            stage,
                            extras_path,
                            f"extra_{j}",
                            box_loc
                            + Gf.Vec3f(
                                (x - 1) * ex_spacing,
                                (y - 1) * ex_spacing,
                                (z - 1) * ex_spacing,
                            )
                            * self.box_size,
                            self.box_size,
                            self.asset_paths["cube"],
                            deformable_material2_path,
                            green_glass_path,
                        )

        # ==================================================================
        # >>> TASHAN SENSOR ATTACHMENT <<<
        # Attach TS-F-A sensors to every Franka finger after all envs are built
        # ==================================================================
        self._sensor_manager = TashanFingerSensorManager(
            num_envs=self.num_envs,
            sensor_usd_path=self._ts_sensor_usd,
        )
        self._sensor_manager.attach_sensors_to_stage(stage, envsScopePath)
        print(
            f"[FrankaDeformableSensorDemo] Attached TS-F-A sensors to "
            f"{self.num_envs * 2} fingers ({self.num_envs} envs × 2 fingers)"
        )

        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)

    # ------------------------------------------------------------------
    # Tensor API start
    # ------------------------------------------------------------------

    def on_tensor_start(self, tensorApi):
        sim = tensorApi.create_simulation_view("numpy")
        sim.set_subspace_roots("/World/envs/*")

        # Franka articulation view
        self.frankas = sim.create_articulation_view("/World/envs/*/franka")
        self.franka_indices = np.arange(self.frankas.count, dtype=np.int32)

        # Set default DOF positions
        init_dof_pos = np.stack(
            self.num_envs * [np.array(self.default_dof_pos, dtype=np.float32)]
        )
        self.frankas.set_dof_positions(init_dof_pos, self.franka_indices)

        # End-effector (hand) view
        self.hands = sim.create_rigid_body_view("/World/envs/*/franka/panda_hand")
        init_hand_transforms = self.hands.get_transforms().copy()
        self.init_pos = init_hand_transforms[:, :3]
        self.init_rot = init_hand_transforms[:, 3:]

        # Deformable box view
        if deformable_beta_on:
            self.boxes = sim.create_volume_deformable_body_view("/World/envs/*/box")
        else:
            self.boxes = sim.create_soft_body_view("/World/envs/*/box")

        # Box corner coords for grasping angle
        box_half_size = 0.5 * self.box_size
        corner_coord = np.array([box_half_size, box_half_size, box_half_size])
        self.corners = np.stack(self.num_envs * [corner_coord])

        self.init_box_transforms = self.boxes.get_transforms().copy()

        self.down_dir = np.array([0, 0, -1]).reshape(3, 1)
        self.down_q = np.stack(self.num_envs * [np.array([1.0, 0.0, 0.0, 0.0])])

        # ==================================================================
        # >>> TASHAN SENSOR INITIALISATION <<<
        # ==================================================================
        if self._sensor_manager is not None:
            self._sensor_manager.init_tensor_views(sim)
            self._sensor_manager.init_rigid_prim_sensors()
            self._sensor_manager.init_rerun()

        # Prevent GC
        self.sim = sim

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def on_physics_step(self, dt):
        # --- Box transforms ------------------------------------------------
        box_transforms = self.boxes.get_transforms()
        box_pos = box_transforms[:, :3]

        # --- End-effector transforms ---------------------------------------
        hand_transforms = self.hands.get_transforms()
        hand_pos = hand_transforms[:, :3]
        hand_rot = hand_transforms[:, 3:]

        # --- Franka DOF state & Jacobians ----------------------------------
        dof_pos = self.frankas.get_dof_positions()
        jacobians = self.frankas.get_jacobians()

        # --- Direction from hand to box ------------------------------------
        to_box = box_pos - hand_pos
        box_dist = np.linalg.norm(to_box, axis=1)
        box_dir = to_box / box_dist[:, None]
        box_dot = (box_dir @ self.down_dir).flatten()

        grasp_offset = 0.10

        # --- Grip detection ------------------------------------------------
        gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
        gripped = (gripper_sep < self.box_size) & (box_dist < grasp_offset * 2.0)

        # --- Unreachable check ---------------------------------------------
        from_start = box_pos - self.init_box_transforms[:, :3]
        start_dist = np.linalg.norm(from_start, axis=1)
        unreachable = start_dist > 0.5

        # --- Goal position -------------------------------------------------
        above_box = (box_dot >= 0.9) & (box_dist < grasp_offset * 3)
        grasp_pos = box_pos.copy()
        grasp_pos[:, 2] = np.where(
            above_box,
            box_pos[:, 2] + grasp_offset,
            box_pos[:, 2] + grasp_offset * 2.5,
        )
        goal_pos = np.where(gripped[:, None], self.init_pos, grasp_pos)
        goal_rot = self.down_q

        # --- IK (damped least squares) -------------------------------------
        pos_err = goal_pos - hand_pos
        max_err = 0.05
        pos_err_mag = np.linalg.norm(pos_err, axis=1)[:, None] + 0.0001
        pos_err = np.where(
            pos_err_mag > max_err, pos_err * max_err / pos_err_mag, pos_err
        )
        orn_err = orientation_error(goal_rot, hand_rot)
        dpose = np.concatenate([pos_err, orn_err], -1)[:, None].transpose(0, 2, 1)

        franka_hand_index = 8
        j_eef = jacobians[:, franka_hand_index - 1, :]
        j_eef_T = np.transpose(j_eef, (0, 2, 1))
        d = 0.05
        lmbda = np.eye(6) * (d ** 2)
        u = (
            j_eef_T @ np.linalg.inv(j_eef @ j_eef_T + lmbda) @ dpose
        ).reshape(self.num_envs, 9)

        pos_targets = dof_pos + u

        # --- Gripper control -----------------------------------------------
        if self.loop:
            close_gripper = np.where(
                gripped,
                (box_dist < grasp_offset * 1.5),
                (box_dist < grasp_offset * 1.1),
            )
        else:
            close_gripper = box_dist < grasp_offset

        grip0 = 0.015
        grip1 = 0.05
        grip_acts = np.where(
            close_gripper[:, None],
            [[grip0, grip0]] * self.num_envs,
            [[grip1, grip1]] * self.num_envs,
        )
        pos_targets[:, 7:9] = grip_acts

        # Reset if unreachable
        pos_targets = np.where(
            unreachable[:, None], self.default_dof_pos, pos_targets
        )

        self.frankas.set_dof_position_targets(pos_targets, self.franka_indices)

        # ==================================================================
        # >>> TASHAN SENSOR UPDATE <<<
        # Read all finger sensors and push data to Rerun
        # ==================================================================
        if self._sensor_manager is not None:
            self._sensor_manager.read_sensors()
            self._sensor_manager.update_rerun()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def on_shutdown(self):
        # Save sensor plots before cleanup
        if self._sensor_manager is not None:
            self._sensor_manager.draw_data(env_idx=0)
            self._sensor_manager.cleanup()
            self._sensor_manager = None

        self.frankas = None
        self.hands = None
        self.boxes = None
        self.sim = None

        if self._reset_hydra_instancing_on_shutdown:
            carb.settings.get_settings().set(
                "/persistent/omnihydra/useSceneGraphInstancing", False
            )
        super().on_shutdown()

    # ------------------------------------------------------------------
    # Franka parameters (unchanged from original)
    # ------------------------------------------------------------------

    def get_franka_parameters(self):
        default_dof_pos = [0.0, 0.0, 0.0, -0.95, 0.0, 1.12, 0.0, 0.02, 0.02]
        revolute_drive_params = {
            "stiffness": 2000.0,
            "damping": 50.0,
            "maxForce": 3000.0,
        }
        linear_drive_params = {
            "stiffness": 2000.0,
            "damping": 50.0,
            "maxForce": 3000.0,
        }

        drive_params = [revolute_drive_params] * 7
        drive_params.extend([linear_drive_params] * 2)
        drive_params[5]["stiffness"] = 800.0
        drive_params[6]["stiffness"] = 800.0

        return default_dof_pos, drive_params
