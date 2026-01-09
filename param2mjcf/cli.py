#!/usr/bin/env python3
import argparse
import sys
import os

from .loader import normalize_auxiliaries, parse_model_instances
from .topology import build_topology
from .generator import MujocoBuilderWithMesh
from .utils import print_auxiliaries, print_topology

def main():
    parser = argparse.ArgumentParser(description="Convert Editor3D Parametric Model (JSON) to MJCF (XML)")
    parser.add_argument("input_json", help="Path to the input JSON file (e.g., editor3dJson.json)")
    parser.add_argument("--output", "-o", default="model.xml", help="Output MJCF XML file path")
    parser.add_argument("--assets", "-a", default="assets", help="Directory to save generated assets (.obj files)")
    parser.add_argument("--model-name", "-n", default="param_robot", help="Name of the MuJoCo model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed topology info")

    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' does not exist.")
        sys.exit(1)

    print(f"Loading {args.input_json}...")

    # 1. Parse Data
    try:
        model_instances = parse_model_instances(args.input_json)
        auxiliaries = normalize_auxiliaries(args.input_json)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

    if args.verbose:
        print("\n--- Auxiliaries Data ---")
        print_auxiliaries(auxiliaries)

    # 2. Build Topology
    topology_forest = build_topology(auxiliaries)
    
    if args.verbose:
        print("\n--- Topology Tree ---")
        print_topology(topology_forest)

    # 3. Generate MJCF
    print(f"\nBuilding MJCF structure '{args.model_name}'...")
    try:
        builder = MujocoBuilderWithMesh(model_name=args.model_name, asset_dir=args.assets)
        builder.build(topology_forest, model_instances)
        
        # 4. Save
        builder.save_xml(args.output)
        print(f"Success! MJCF saved to '{args.output}'")
        print(f"Assets saved in '{args.assets}/'")
        
    except Exception as e:
        print(f"Error building MJCF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
