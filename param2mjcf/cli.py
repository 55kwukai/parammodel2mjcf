#!/usr/bin/env python3
import argparse
import sys
import os

from .loader import normalize_auxiliaries, parse_model_instances
from .topology import build_topology
from .generator import MujocoBuilderWithMesh
from .utils import print_auxiliaries, print_topology

def main():
    parser = argparse.ArgumentParser(
        description="Convert Editor3D Parametric Model (JSON) to MJCF (XML)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-infer output paths from input directory)
  param2mjcf examples/three_drawers/input.json

  # Specify custom output file
  param2mjcf examples/three_drawers/input.json -o custom_output.xml

  # Specify all paths explicitly
  param2mjcf input.json -o output.xml -a assets/ -n my_model
        """
    )
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument("--output", "-o", default=None,
                       help="Output MJCF XML file path (default: <input_dir>/output.xml)")
    parser.add_argument("--assets", "-a", default=None,
                       help="Directory to save generated assets (default: <input_dir>/assets/)")
    parser.add_argument("--model-name", "-n", default=None,
                       help="Name of the MuJoCo model (default: derived from input directory name)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed topology info")

    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' does not exist.")
        sys.exit(1)

    # Smart path inference
    input_dir = os.path.dirname(args.input_json)
    input_basename = os.path.basename(args.input_json)

    # If no output specified, infer from input directory
    if args.output is None:
        if input_dir:
            args.output = os.path.join(input_dir, "output.xml")
        else:
            args.output = "output.xml"

    # If no assets directory specified, infer from input directory
    if args.assets is None:
        if input_dir:
            args.assets = os.path.join(input_dir, "assets")
        else:
            args.assets = "assets"

    # If no model name specified, infer from directory name
    if args.model_name is None:
        if input_dir:
            # Use the parent directory name (e.g., "three_drawers" from "examples/three_drawers")
            dir_name = os.path.basename(input_dir)
            args.model_name = dir_name if dir_name else "model"
        else:
            # Use input filename without extension
            args.model_name = os.path.splitext(input_basename)[0]

    print(f"Loading {args.input_json}...")
    print(f"Output will be saved to: {args.output}")
    print(f"Assets will be saved to: {args.assets}/")
    print(f"Model name: {args.model_name}")

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
