"""RoboCasa panda_omron modality config for Isaac-GR00T N1.7 finetune.

Registered for EmbodimentTag.NEW_EMBODIMENT and loaded via
`launch_finetune_n17.py --modality_config_path .../robocasa_config_n17.py`
(same mechanism as LeIsaac's leisaac_config_n17.py).

Why this is a hand-written modality config and not the N1.5 fork's
`panda_omron` DataConfig:
- The N1.5 robocasa-benchmark fork added `PandaOmronDataConfig` + a manual
  quaternion->rotation_6d path in state_action.py. N1.7 refactored all of that
  and ships the rotation machinery natively (ActionFormat.XYZ_ROT6D /
  XYZ_ROTVEC, ActionType.EEF). So no fork code needs porting — we just declare
  the keys here and N1.7's StateActionProcessor handles normalization.

Keys mirror the dataset's meta/modality.json exactly
(target/atomic/OpenCabinet .../lerobot, codebase v2.1):
  state  (16) = base_position 3 + base_rotation 4 + eef_pos_rel 3
                + eef_rot_rel 4 + gripper_qpos 2
  action (12) = base_motion 4 + control_mode 1 + eef_position 3
                + eef_rotation 3 + gripper_close 1
  video  (3)  = robot0_agentview_left / _right / eye_in_hand
  language    = annotation.human.task_description

ACTION REPRESENTATION — the one modeling decision, made deliberately:
  RoboCasa actions are ALREADY deltas (robosuite OSC_POSE delta commands;
  verified from stats.json: eef_pos/eef_rot centered at ~0, small std).
  N1.7 only relativizes a group when `rep == RELATIVE AND use_relative_action`
  (state_action_processor.py:169,351,460). To AVOID double-relativizing
  already-delta actions we set every action group to rep=ABSOLUTE +
  type=NON_EEF + format=DEFAULT, i.e. "normalize the raw values as-is, no SE(3)
  conversion". This faithfully reproduces the proven N1.5 robocasa recipe
  (min_max on raw deltas). The launcher also sets use_relative_action=False as
  a belt-and-suspenders guard.

  We intentionally do NOT use ActionType.EEF + XYZ_ROT6D here: that path does
  SE(3) relative composition meant for ABSOLUTE eef poses, not for delta
  commands, and it expects a single combined eef pose key rather than the
  separate position/rotation keys this dataset has.

action_horizon = 40: N1.7's Gr00tN1d7Config hard-defaults action_horizon=40;
the data-side action delta_indices must match (loader derives
horizon = max(delta) - min(delta) + 1). This is the one change vs the N1.5
config (which used range(16)).

NOTE: only one modality config can be registered for a given EmbodimentTag at a
time — do not import this alongside another NEW_EMBODIMENT config in one run.

VALIDATION BEFORE TRUSTING A RUN: do an open-loop replay probe (feed training
frames through preproc -> policy -> postproc and compare predicted vs recorded
actions), the same leading-indicator discipline used for the ACT work. A wrong
rep/format shows up as large eef L1 there before you waste GPU-hours.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def _raw_delta_action():
    """rep=ABSOLUTE + NON_EEF + DEFAULT: normalize raw delta values, no
    relativization, no rotation conversion. Used for every RoboCasa action
    group (see module docstring)."""
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


robocasa_panda_omron_n17_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "base_position",
            "base_rotation",
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),  # N1.7 action_horizon=40
        modality_keys=[
            "base_motion",
            "control_mode",
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
        ],
        action_configs=[
            _raw_delta_action(),  # base_motion   (static for OpenCabinet)
            _raw_delta_action(),  # control_mode  (constant for OpenCabinet)
            _raw_delta_action(),  # end_effector_position  (xyz delta)
            _raw_delta_action(),  # end_effector_rotation  (rotvec delta)
            _raw_delta_action(),  # gripper_close
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    robocasa_panda_omron_n17_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
