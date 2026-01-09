import os
import numpy as np
import mujoco
from typing import List, Dict, Set, Optional

from .types import Pose
from .utils import get_local_pose

class MujocoBuilderWithMesh:
    def __init__(self, model_name: str = "robot", asset_dir: str = "assets"):
        self.asset_dir = asset_dir
        self.model_name = model_name

        # 1. Initialize Spec
        self.spec = mujoco.MjSpec()
        self.spec.modelname = model_name

        # 2. Basic Configuration
        self.spec.compiler.degree = False
        self.spec.compiler.inertiafromgeom = True
        self.spec.option.gravity = [0, 0, -9.81]
        self.spec.option.timestep = 0.002

        # 3. State Tracking
        self.processed_ids: Set[str] = set()
        self.instance_map: Dict[str, Dict] = {}

        # Ensure asset directory exists
        if not os.path.exists(self.asset_dir):
            os.makedirs(self.asset_dir, exist_ok=True)

    def _write_obj_file(self, filename: str, vertices: np.ndarray, faces: np.ndarray):
        filepath = os.path.join(self.asset_dir, filename)

        # Fix: If 1D array, reshape to (N, 3)
        if vertices.ndim == 1:
            vertices = vertices.reshape(-1, 3)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Generated for {self.model_name}\n")
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (1-indexed)
            num_faces = len(faces) // 3
            for i in range(num_faces):
                idx = i * 3
                f.write(f"f {faces[idx] + 1} {faces[idx + 1] + 1} {faces[idx + 2] + 1}\n")
        return filename

    def _prepare_assets(self, instances: List[Dict]):
        """
        1. Build ID mapping
        2. Write Mesh data to files
        3. Register Mesh assets in Spec
        """
        print(f"Processing assets for {len(instances)} instances...")

        for inst in instances:
            uid = inst['unique_id']
            self.instance_map[uid] = inst

            # Process all Sub-Meshes for this instance
            for i, mesh_data in enumerate(inst['meshes']):
                # Construct filename: instanceID_subIndex.obj
                mesh_name = f"{uid}_m{i}"
                filename = f"{mesh_name}.obj"

                # Write to disk
                self._write_obj_file(filename, mesh_data['vertices'], mesh_data['faces'])

                # Add Asset to MuJoCo Spec
                mesh_asset = self.spec.add_mesh()
                mesh_asset.name = mesh_name
                mesh_asset.file = os.path.join(self.asset_dir, filename)

    def _get_joint_enum(self, raw_type: str):
        t = str(raw_type).lower()
        if "slider" in t or "2" in t:
            return mujoco.mjtJoint.mjJNT_SLIDE
        if "hinge" in t:
            return mujoco.mjtJoint.mjJNT_HINGE
        return None

    def _get_abs_pose_from_instance(self, body_id: str) -> Pose:
        body_pos = np.array([0.0, 0.0, 0.0])
        body_quat = np.array([1.0, 0.0, 0.0, 0.0])

        if body_id in self.instance_map:
            inst = self.instance_map[body_id]
            body_pos = np.array(inst['position'])
            if 'rotation' in inst:
                # Convert Euler to Quat (assuming XYZ order as in original script)
                mujoco.mju_euler2Quat(body_quat, np.array(inst['rotation']), "XYZ")
        
        return Pose(body_pos, body_quat)

    def _add_body_recursive(self, parent_body, node: Dict, parent_abs_pose: Pose):
        body_id = node["body_id"]
        link_data = node.get("link_data")
        self.processed_ids.add(body_id)

        # 1. Create Body (Container)
        # -------------------------------------------------
        # Get Body Pose from Instance (Input is absolute coords, radians)
        current_abs_pose = self._get_abs_pose_from_instance(body_id)
        
        rel_pose = get_local_pose(parent_abs_pose, current_abs_pose)
        
        current_body = parent_body.add_body()
        current_body.name = body_id
        current_body.pos = rel_pose.pos
        current_body.quat = rel_pose.quat
        current_body.mass = 1.0

        # 2. Add Geoms (Support multiple Meshes per Body)
        # -------------------------------------------------
        if body_id in self.instance_map:
            inst = self.instance_map[body_id]
            # The index 'i' must match logic in _prepare_assets
            for i in range(len(inst['meshes'])):
                geom = current_body.add_geom()
                geom.type = mujoco.mjtGeom.mjGEOM_MESH
                # Matches resource name generated in _prepare_assets: "ID_m0", "ID_m1"...
                geom.meshname = f"{body_id}_m{i}"
                geom.pos = np.zeros(3)  # Default: mesh origin coincides with body
                geom.rgba = np.array([0.8, 0.8, 0.8, 1.0])

        # 3. Setup Joint & Actuator (from Auxiliary)
        # -------------------------------------------------
        if link_data:
            j_type = self._get_joint_enum(link_data.get("joint_type", ""))
            
            # Parse range string "{min, max}"
            raw_range = link_data.get("actuator_range", "{0,0}")
            if isinstance(raw_range, str):
                ctrl_range_min, ctrl_range_max = map(float, raw_range.strip('{}').split(','))
            else:
                 ctrl_range_min, ctrl_range_max = 0.0, 0.0


            if j_type is not None:
                joint = current_body.add_joint()
                joint.name = f"joint_{body_id}"
                joint.type = j_type
                joint.range[:] = [ctrl_range_min, ctrl_range_max]

                if "position" in link_data:
                    joint_pos = np.array(link_data["position"])
                    # Construct Joint Absolute Pose (Rotation default unit quat, strictly position relative)
                    # Note: Original code treated joint position as absolute-like relative to parent? 
                    # Actually original code calculated local pose relative to current body.
                    
                    # Wait, let's re-read the original logic carefully.
                    # Original: 
                    # joint_abs_pose = Pose(joint_pos, identity)
                    # joint_local_pose = get_local_pose(current_abs_pose, joint_abs_pose)
                    # This implies 'position' in auxiliary is in WORLD coordinates (or at least same frame as body pos).
                    
                    joint_abs_pose = Pose(joint_pos, np.array([1., 0., 0., 0.]))
                    joint_local_pose = get_local_pose(current_abs_pose, joint_abs_pose)
                    joint.pos = joint_local_pose.pos

                if "rotation" in link_data:
                    joint.axis = np.array(link_data["rotation"])
                else:
                    joint.axis = np.array([0, 0, 1])

                if link_data.get("has_actuator"):
                    act = self.spec.add_actuator()
                    act.name = f"actuator_{body_id}"
                    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                    act.target = joint.name
                    kp = 2000
                    act.dyntype = mujoco.mjtDyn.mjDYN_NONE
                    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                    act.gainprm[0] = kp
                    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                    act.biasprm[1] = -kp  # -kp * qpos
                    act.ctrlrange[:] = [ctrl_range_min, ctrl_range_max]

        # 4. Recurse Children
        for child in node.get("children", []):
            self._add_body_recursive(current_body, child, current_abs_pose)

    def build(self, topology_forest: List[Dict], instances: List[Dict]):
        """
        Main build function
        """
        # 1. Prepare assets (Generate OBJ and register Mesh assets)
        self._prepare_assets(instances)

        # 2. Build Topology Tree (Handle Parent-Child Bodies)
        world_pose = Pose() # Default origin and unit rotation
        
        for root_node in topology_forest:
            self._add_body_recursive(self.spec.worldbody, root_node, world_pose)

        # 3. Handle Orphan/Fixed Bodies
        # -------------------------------------------------
        for inst in instances:
            uid = inst['unique_id']

            # If ID not processed in topology tree, treat as orphan/fixed
            if uid not in self.processed_ids:
                # A. Create Body
                body = self.spec.worldbody.add_body()
                body.name = uid
                body.pos = np.array(inst['position'])
                body.mass = 1.0

                quat = np.zeros(4)
                if 'rotation' in inst:
                    mujoco.mju_euler2Quat(quat, np.array(inst['rotation']), "XYZ")
                else:
                    quat = np.array([1., 0., 0., 0.])
                body.quat = quat

                # B. Add Geoms
                for i in range(len(inst['meshes'])):
                    geom = body.add_geom()
                    geom.type = mujoco.mjtGeom.mjGEOM_MESH
                    geom.meshname = f"{uid}_m{i}"
                    geom.rgba = np.array([0.5, 0.5, 1.0, 1.0])  # Blue for orphans

    def save_xml(self, filename: str) -> str:
        xml_str = self.spec.to_xml()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        return xml_str
