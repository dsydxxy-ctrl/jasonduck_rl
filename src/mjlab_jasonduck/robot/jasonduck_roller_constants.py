"""Independent JasonDuck skates: CAD inertia, JasonDuck HLS1910 servos, four passive wheels."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from math import radians
import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab_jasonduck.robot.jasonduck_constants import JASONDUCK_HOME_FRAME, actuators

JASONDUCK_ROLLER_XML = Path(__file__).parent / 'jasonduck_roller/robot_backlash.xml'
# Lowest tire vertex at the roller-specific HOME pose touches the floor.
JASONDUCK_ROLLER_STAND_Z = 0.15481704568257137
JASONDUCK_WHEEL_RADIUS = 0.015  # 30 mm CAD tires; upstream MicroDuck uses 35 mm tires.
JASONDUCK_ROLLER_HOME_FRAME = deepcopy(JASONDUCK_HOME_FRAME)
JASONDUCK_ROLLER_HOME_FRAME.pos = (0.0, 0.0, JASONDUCK_ROLLER_STAND_Z)
# Roller HOME is independent of the wheel-less Walk posture.
JASONDUCK_ROLLER_HOME_FRAME.joint_pos = {
    'left_hip_yaw': radians(0),
    'left_hip_roll': radians(0),
    'left_hip_pitch': radians(-10),
    'left_knee': radians(10),
    'left_ankle': radians(20),
    'neck_pitch': radians(25),
    'head_pitch': radians(25),
    'head_yaw': radians(0),
    'head_roll': radians(0),
    'right_hip_yaw': radians(0),
    'right_hip_roll': radians(0),
    'right_hip_pitch': radians(10),
    'right_knee': radians(-10),
    'right_ankle': radians(-20),
    r'^passive_.*wheel$': 0.0,
    r'^passive_.*_backlash$': 0.0,
}

def get_roller_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(JASONDUCK_ROLLER_XML))

# Use the same HLS1910 fit, P and delay as the wheel-less JasonDuck.
_roller_actuators = deepcopy(actuators)

JASONDUCK_ROLLER_ROBOT_CFG = EntityCfg(
    spec_fn=get_roller_spec,
    init_state=JASONDUCK_ROLLER_HOME_FRAME,
    collisions=(),  # Named wheel/non-wheel masks are explicitly defined in the XML.
    articulation=EntityArticulationInfoCfg(
        actuators=(_roller_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
