import json
import os
import numpy as np
from typing import List, Dict

# Field mapping configuration: Original Field Name -> New Field Name
PROPERTY_MAPPING = {
    "hkglwt": "parent_body",      # Slide block associated model -> Parent Body
    "hgzzmx": "current_body",     # Slide rail termination model -> Current Body
    "hingeActuator": "has_actuator", # Has motor
    "mdfwtest": "actuator_range"  # Motor range (Optional optimization)
}

def normalize_auxiliaries(json_path: str) -> List[Dict]:
    """
    Read JSON file and parse auxiliary structure data used for joints and actuators.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON format error: {e}")

    # Compatible with direct list or structure contained in customAuxiliaries object
    raw_list = data.get("customAuxiliaries", []) if isinstance(data, dict) else data
    if not isinstance(raw_list, list):
        return []

    normalized_list = []

    for item in raw_list:
        # 1. Extract basic transform info
        pos = item.get("position", {})
        rot = item.get("rotate", {})

        # Build basic object structure
        aux_obj = {
            "id": item.get("instanceId"),
            "joint_type": item.get("auxiliaryType"),
            "position": [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)],
            "rotation": [rot.get("x", 0), rot.get("y", 0), rot.get("z", 0)]
        }

        # 2. Process business properties (paramBizProperties)
        # Convert list to flattened key-value pairs and perform renaming
        raw_props = item.get("paramBizProperties", [])

        for prop in raw_props:
            original_key = prop.get("key")
            raw_value = prop.get("value")
            val_type = prop.get("valueType")

            # Value type conversion
            if val_type == "boolean":
                final_value = True if str(raw_value).lower() == "true" else False
            else:
                final_value = raw_value

            # Key mapping and assignment
            target_key = PROPERTY_MAPPING.get(original_key, original_key)
            aux_obj[target_key] = final_value

        normalized_list.append(aux_obj)

    return normalized_list

def parse_model_instances(json_path: str) -> List[Dict]:
    """
    Parse editor3d JSON file to extract model instances with mesh data.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'modelInstances' not in data:
        print("❌ Error: 'modelInstances' field not found in JSON file")
        return []
    
    instances = []
    
    for idx, instance in enumerate(data['modelInstances']):
        unique_id = instance.get('uniqueId', f'instance_{idx}')
        
        # Extract position
        pos_dict = instance.get('position', {})
        position = (
            pos_dict.get('x', 0.0),
            pos_dict.get('y', 0.0),
            pos_dict.get('z', 0.0)
        )
        
        # Extract rotation (radians)
        rot_dict = instance.get('rotate', {})
        rotation = (
            rot_dict.get('x', 0.0),
            rot_dict.get('y', 0.0),
            rot_dict.get('z', 0.0)
        )
        
        # Extract meshes
        meshes_data = []
        for mesh_idx, mesh in enumerate(instance.get('meshes', [])):
            if 'coordinateArray' in mesh and 'coordinateIndexArray' in mesh:
                vertices = np.array(mesh['coordinateArray'], dtype=np.float64)
                faces = np.array(mesh['coordinateIndexArray'], dtype=np.int32)
                
                num_vertices = len(vertices) // 3
                num_faces = len(faces) // 3
                
                meshes_data.append({
                    'vertices': vertices,
                    'faces': faces,
                    'num_vertices': num_vertices,
                    'num_faces': num_faces,
                    'mesh_index': mesh_idx
                })
        
        if meshes_data:  # Only add instances with mesh data
            instances.append({
                'unique_id': unique_id,
                'position': position,
                'rotation': rotation,
                'meshes': meshes_data,
                'instance_index': idx
            })
            
    return instances
