import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from typing import List, Dict
from .types import Pose

def get_local_pose(parent_pose: Pose, child_pose: Pose) -> Pose:
    """
    Calculate the local transform (Pos, Quat) of a child relative to a parent.
    """
    # 1. Calculate inverse of parent rotation (Conjugate quaternion: w, -x, -y, -z)
    inv_parent_quat = np.array([parent_pose.quat[0], -parent_pose.quat[1], -parent_pose.quat[2], -parent_pose.quat[3]])

    # 2. Calculate relative position: R_parent_inv * (P_child - P_parent)
    diff_pos = child_pose.pos - parent_pose.pos
    local_pos = np.zeros(3)
    mujoco.mju_rotVecQuat(local_pos, diff_pos, inv_parent_quat)

    # 3. Calculate relative rotation: R_parent_inv * R_child
    local_quat = np.zeros(4)
    mujoco.mju_mulQuat(local_quat, inv_parent_quat, child_pose.quat)

    return Pose(local_pos, local_quat)

def rotation_to_quaternion(rx: float, ry: float, rz: float) -> List[float]:
    """
    Convert XYZ Euler angles (radians) to quaternion (w, x, y, z).
    """
    r = Rotation.from_euler('xyz', [rx, ry, rz], degrees=False)
    quat = r.as_quat()  # Returns (x, y, z, w)
    # Convert to Mujoco format (w, x, y, z)
    return [quat[3], quat[0], quat[1], quat[2]]

def print_auxiliaries(auxiliaries: List[Dict]):
    """
    Print processed auxiliary structures in a table format.
    """
    if not auxiliaries:
        print("List is empty.")
        return

    row_fmt = "{:<5} | {:<5} | {:<20} | {:<10} | {:<20} | {:<20}"
    header = row_fmt.format("ID", "Type", "Position", "Actuator", "Parent Body", "Current Body")

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for aux in auxiliaries:
        # Format coordinate display
        pos_str = f"[{aux['position'][0]:.2f}, {aux['position'][1]:.2f}, {aux['position'][2]:.2f}]"

        # Get attributes with defaults
        has_motor = "Yes" if aux.get("has_actuator") else "No"
        p_body = str(aux.get("parent_body", "N/A"))
        c_body = str(aux.get("current_body", "N/A"))

        print(row_fmt.format(
            aux['id'],
            aux['joint_type'],
            pos_str,
            has_motor,
            p_body,
            c_body
        ))

    print("-" * len(header))

def print_topology(trees: List[Dict]):
    """
    Print the topology tree structure visually.
    """
    if not trees:
        print("Topology tree is empty.")
        return

    print("=" * 40)
    print("PHYSICS TOPOLOGY TREE")
    print("=" * 40)

    def _print_node(node: Dict, prefix: str = "", is_last: bool = True, is_root: bool = False):
        body_id = node["body_id"]
        children = node["children"]
        link = node["link_data"]

        # 1. Construct current line content
        connector = ""
        if not is_root:
            connector = "└── " if is_last else "├── "

        # 2. Extract connection info (if not root)
        info_str = f"[{body_id}]"
        if link:
            motor = " (Motor)" if link.get('has_actuator') else ""
            # Display format: └── [JointType: ID] -> [BodyID] (Motor?)
            display_str = f"{prefix}{connector} -> \033[1m{info_str}\033[0m{motor}"
        else:
            # Root node display
            display_str = f"{prefix}{info_str} (Root)"

        print(display_str)

        # 3. Calculate prefix for children
        if is_root:
            new_prefix = prefix
        else:
            new_prefix = prefix + ("    " if is_last else "│   ")

        # 4. Recursively print children
        count = len(children)
        for i, child in enumerate(children):
            _print_node(child, new_prefix, i == count - 1, is_root=False)

    # Iterate over all trees (usually one forest, but could be multiple independent structures)
    for tree in trees:
        _print_node(tree, is_root=True)
        print("-" * 20)
