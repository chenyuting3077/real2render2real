"""OpenArm task scenes for R2R2R."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from real2render2real.isaaclab_viser.configs.scene_configs.openarm_base_cfg import PAD_HEIGHT, OpenArmBaseCfg

dir_path = os.path.dirname(os.path.realpath(__file__))
outputs_dir = os.path.join(dir_path, "../../../../outputs")

PRINGLES_USD = os.path.join(
    outputs_dir,
    "pringles/state_rigid_20261104_193955/usd/"
    "sugarfine_3Dgs7000_densityestim02_sdfnorm02_level005_decim200000_normalconsistency01_gaussperface6"
    "_fixed_normals_centered.usd",
)
# The centred mesh's lowest point (after cutting the unscanned bottom at z=-0.116) is 0.116 m below its origin.
PRINGLES_BOTTOM_OFFSET = 0.116


@configclass
class OpenArmPringlesPickCfg(OpenArmBaseCfg):
    """Right-arm pick of the scanned Pringles can."""

    pringles = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pringles",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PRINGLES_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=16,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        # On the pad between the grippers (clear of the right hand at home); the simulator re-places it on reset.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, -0.10, PAD_HEIGHT + PRINGLES_BOTTOM_OFFSET + 0.002)),
    )
