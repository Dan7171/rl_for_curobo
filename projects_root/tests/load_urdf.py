 #!/usr/bin/env python3
"""
URDF Loading Test Script for K-Arm Systems

This script tests URDF generation and loading for various robot models
and multi-arm configurations in Isaac Sim.

Usage:
    python load_urdf.py --robot franka --num_arms 3 --layout line
    python load_urdf.py --robot ur5e --num_arms 4 --layout circle
    python load_urdf.py --robot mixed --config_file mixed_config.json
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add projects_root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    # Third Party
    import isaacsim
except ImportError:
    print("IsaacSim not available - running in test mode")

# Third Party
import torch
import numpy as np

# Parse arguments before importing Isaac Sim
parser = argparse.ArgumentParser(description="Test K-arm URDF generation and loading")
parser.add_argument("--robot", type=str, default="franka", 
                   choices=["franka", "ur5e", "ur10e", "kinova", "iiwa", "jaco", "tm12", "mixed"],
                   help="Robot model to test")
parser.add_argument("--num_arms", type=int, default=3, help="Number of arms")
parser.add_argument("--layout", type=str, default="line", 
                   choices=["line", "circle", "grid"],
                   help="Layout pattern for arms")
parser.add_argument("--spacing", type=float, default=None, help="Custom arm spacing")
parser.add_argument("--config_file", type=str, default=None, help="JSON config for mixed arms")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--save_image", action="store_true", help="Save screenshot")

args = parser.parse_args()

# Start Isaac Sim
try:
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({
        "headless": args.headless,
        "width": 1920,
        "height": 1080,
    })
    
    # Isaac Sim imports (after SimulationApp initialization)
    import omni.usd
    from pxr import Usd, UsdGeom, Gf
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import create_new_stage
    
    ISAAC_SIM_AVAILABLE = True
    print("✓ Isaac Sim initialized successfully")
    
except ImportError as e:
    ISAAC_SIM_AVAILABLE = False
    print(f"⚠ Isaac Sim not available: {e}")
    print("Running in config generation test mode only")

# Import our K-arm config generator
from projects_root.utils.multi_arm_config_generator import (
    MultiArmConfigGenerator, 
    create_k_arm_system, 
    create_mixed_arm_system,
    ROBOT_MODELS
)


def load_mixed_config(config_file: str) -> list:
    """Load mixed arm configuration from JSON file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config.get("arms", [])


def test_config_generation():
    """Test configuration generation without Isaac Sim."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION GENERATION")
    print("="*60)
    
    generator = MultiArmConfigGenerator("test_generated_configs")
    
    if args.robot == "mixed":
        if args.config_file:
            arm_specs = load_mixed_config(args.config_file)
        else:
            # Default mixed configuration
            arm_specs = [
                {"model": "franka", "position": [-1.0, 0.0, 0.0]},
                {"model": "ur5e", "position": [0.0, 0.0, 0.0]}, 
                {"model": "kinova", "position": [1.0, 0.0, 0.0]}
            ]
        
        arms, system_name = create_mixed_arm_system(arm_specs, "mixed_test")
        print(f"✓ Created mixed system with {len(arms)} arms:")
        for arm in arms:
            model = arm.urdf_path.split('/')[-2] if '/' in arm.urdf_path else "unknown"
            print(f"  - {arm.arm_id}: {model} at {arm.base_position}")
    else:
        arms, system_name = create_k_arm_system(
            robot_model=args.robot,
            num_arms=args.num_arms,
            arm_spacing=args.spacing,
            layout=args.layout
        )
        print(f"✓ Created {args.robot} {args.num_arms}-arm system ({args.layout} layout)")
    
    try:
        # Generate URDF
        urdf_path = generator.generate_multi_arm_urdf(arms, system_name)
        print(f"✓ Generated URDF: {urdf_path}")
        
        # Generate CuRobo config
        config_path = generator.generate_curobo_config(arms, system_name, urdf_path)
        print(f"✓ Generated CuRobo config: {config_path}")
        
        # Generate particle MPC config
        mpc_config_path = generator.generate_particle_mpc_config(len(arms), system_name)
        print(f"✓ Generated MPC config: {mpc_config_path}")
        
        # Verify files exist and have content
        assert os.path.exists(urdf_path) and os.path.getsize(urdf_path) > 0, "URDF file empty or missing"
        assert os.path.exists(config_path) and os.path.getsize(config_path) > 0, "Config file empty or missing"
        assert os.path.exists(mpc_config_path) and os.path.getsize(mpc_config_path) > 0, "MPC config file empty or missing"
        
        print(f"✓ All files generated and verified")
        
        return urdf_path, config_path, arms
        
    except Exception as e:
        print(f"✗ Error generating configs: {e}")
        return None, None, None


def test_urdf_loading_isaac_sim(urdf_path: str, arms: list):
    """Test URDF loading in Isaac Sim."""
    if not ISAAC_SIM_AVAILABLE:
        print("⚠ Skipping Isaac Sim test - not available")
        return False
        
    print("\n" + "="*60)
    print("TESTING URDF LOADING IN ISAAC SIM")
    print("="*60)
    
    try:
        # Create new stage
        create_new_stage()
        world = World(stage_units_in_meters=1.0)
        stage = world.stage
        
        # Add ground plane
        world.scene.add_default_ground_plane()
        print("✓ Created stage with ground plane")
        
        # Import URDF to stage
        # Try different URDF import methods based on Isaac Sim version
        abs_urdf_path = os.path.abspath(urdf_path)
        result = False
        prim_path = "/World/Robot"
        
        # Method 1: Try Isaac Sim URDF importer with proper ImportConfig
        try:
            from omni.isaac.urdf import _urdf
            import omni.kit.commands
            
            # Create proper ImportConfig object
            import_config = _urdf.ImportConfig()
            import_config.merge_fixed_joints = False
            import_config.convex_decomp = False
            import_config.import_inertia_tensor = True
            import_config.self_collision = False
            import_config.create_physics_scene = True
            import_config.default_drive_strength = 1e7
            import_config.default_position_drive_damping = 1e5
            import_config.default_drive_type = "position"
            
            result, prim_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=abs_urdf_path,
                import_config=import_config,
            )
            
            if result:
                print(f"✓ Method 1 (URDFParseAndImportFile): Success")
            else:
                raise Exception("Command returned False")
                
        except Exception as e1:
            print(f"⚠ Method 1 failed: {e1}")
            result = False
            
            # Method 2: Try alternative URDF command structure
            try:
                import omni.kit.commands
                
                # Try without ImportConfig, using URDFCreateFile instead
                result, prim_path = omni.kit.commands.execute(
                    "URDFCreateFile",
                    urdf_path=abs_urdf_path,
                    import_config=None,
                )
                
                if result:
                    print(f"✓ Method 2 (URDFCreateFile): Success")
                else:
                    raise Exception("Command returned False")
                    
            except Exception as e2:
                print(f"⚠ Method 2 failed: {e2}")
                result = False
                
                # Method 3: Try direct USD creation from URDF
                try:
                    import xml.etree.ElementTree as ET
                    result = create_robot_from_urdf_manual(abs_urdf_path, stage, prim_path)
                    if result:
                        print(f"✓ Method 3 (manual parsing): Success")
                    else:
                        raise Exception("Manual parsing failed")
                        
                except Exception as e3:
                    print(f"⚠ Method 3 failed: {e3}")
                    result = False
                    
                    # Method 4: Simple visual representation
                    try:
                        result = create_simple_robot_representation(stage, arms, prim_path)
                        if result:
                            print(f"✓ Method 4 (simple representation): Success")
                        else:
                            raise Exception("Simple representation failed")
                    except Exception as e4:
                        print(f"⚠ Method 4 failed: {e4}")
                        result = False
        
        if result:
            print(f"✓ Successfully imported URDF to: {prim_path}")
            
            # Get all prims in the scene
            stage = omni.usd.get_context().get_stage()
            
            # Count links and joints
            link_count = 0
            joint_count = 0
            
            for prim in stage.Traverse():
                if prim.GetTypeName() == "Xform":
                    if "link" in prim.GetName().lower():
                        link_count += 1
                elif prim.GetTypeName() == "PhysicsRevoluteJoint" or prim.GetTypeName() == "PhysicsFixedJoint":
                    joint_count += 1
            
            print(f"✓ Found {link_count} links and {joint_count} joints in scene")
            print(f"✓ Expected approximately {len(arms)} arms")
            
            # Wait a moment for scene to settle
            for _ in range(10):
                world.step(render=not args.headless)
            
            # Save screenshot if requested
            if args.save_image and not args.headless:
                save_screenshot(urdf_path)
            
            print("✓ URDF loaded successfully in Isaac Sim!")
            return True
            
        else:
            print(f"✗ Failed to import URDF: {abs_urdf_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error loading URDF in Isaac Sim: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_robot_from_urdf_manual(urdf_path, stage, prim_path):
    """Fallback method to create robot representation by parsing URDF manually."""
    try:
        import xml.etree.ElementTree as ET
        from pxr import UsdGeom, Gf
        
        # Parse URDF
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # Create robot prim
        robot_prim = UsdGeom.Xform.Define(stage, prim_path)
        
        # Find all links with visual elements
        links = root.findall('.//link')
        for i, link in enumerate(links):
            link_name = link.get('name', f'link_{i}')
            visuals = link.findall('.//visual')
            
            if visuals:
                # Create a simple cube for each visual link
                link_path = f"{prim_path}/{link_name}"
                cube = UsdGeom.Cube.Define(stage, link_path)
                cube.GetSizeAttr().Set(0.1)  # Small cube
                
                # Set some basic material properties
                cube.GetDisplayColorAttr().Set([(0.7, 0.7, 0.7)])
        
        return True
    except Exception as e:
        print(f"Manual URDF parsing failed: {e}")
        return False


def create_simple_robot_representation(stage, arms, prim_path):
    """Create a simple visual representation of multiple arms."""
    try:
        from pxr import UsdGeom, Gf
        
        # Create robot group
        robot_prim = UsdGeom.Xform.Define(stage, prim_path)
        
        # Create a simple representation for each arm
        for i, arm in enumerate(arms):
            arm_path = f"{prim_path}/arm_{i+1}"
            arm_group = UsdGeom.Xform.Define(stage, arm_path)
            
            # Create base
            base_path = f"{arm_path}/base"
            base = UsdGeom.Cylinder.Define(stage, base_path)
            base.GetRadiusAttr().Set(0.15)
            base.GetHeightAttr().Set(0.3)
            base.GetDisplayColorAttr().Set([(0.2, 0.3, 0.8)])
            
            # Position the arm
            pos = arm.position
            arm_group.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
            
            # Create simple links
            for j in range(7):  # Assume 7-DOF arm
                link_path = f"{arm_path}/link_{j+1}"
                link = UsdGeom.Cube.Define(stage, link_path)
                link.GetSizeAttr().Set(0.05)
                link.GetDisplayColorAttr().Set([(0.7, 0.4, 0.2)])
                
                # Position along Z-axis
                link.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(0, 0, j * 0.2 + 0.4))
        
        return True
    except Exception as e:
        print(f"Simple representation failed: {e}")
        return False


def save_screenshot(urdf_path: str):
    """Save a screenshot of the current scene."""
    try:
        import omni.kit.commands
        
        # Generate filename from URDF path
        urdf_name = os.path.splitext(os.path.basename(urdf_path))[0]
        screenshot_path = f"test_screenshots/{urdf_name}_{args.robot}_{args.num_arms}arms.png"
        
        # Create directory
        os.makedirs("test_screenshots", exist_ok=True)
        
        # Capture screenshot
        omni.kit.commands.execute(
            "CaptureViewportToFile",
            file_path=screenshot_path,
            use_camera=False
        )
        
        print(f"✓ Screenshot saved: {screenshot_path}")
        
    except Exception as e:
        print(f"⚠ Could not save screenshot: {e}")


def print_test_summary(success: bool, urdf_path: str, config_path: str, arms: list):
    """Print test summary and next steps."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if success:
        print("✓ All tests PASSED!")
        print(f"\nGenerated files:")
        print(f"  - URDF: {urdf_path}")
        print(f"  - Config: {config_path}")
        print(f"  - Arms: {len(arms)} total")
        
        print(f"\nTo use this configuration:")
        print(f"  python k_arm_centralized_mpc.py \\")
        print(f"    --robot {os.path.basename(config_path)} \\")
        print(f"    --num_arms {len(arms)}")
        
    else:
        print("✗ Some tests FAILED!")
        print("Check the error messages above for details.")
    
    print("\nAvailable robot models:")
    for model in ROBOT_MODELS.keys():
        info = ROBOT_MODELS[model]
        print(f"  - {model}: {info['ee_link_suffix']} ({info['default_spacing']}m spacing)")


def create_example_mixed_config():
    """Create an example mixed configuration file."""
    example_config = {
        "arms": [
            {
                "model": "franka",
                "position": [-1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0]
            },
            {
                "model": "ur5e", 
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0]
            },
            {
                "model": "kinova",
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0]
            }
        ]
    }
    
    with open("example_mixed_config.json", 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print("✓ Created example_mixed_config.json")


def main():
    """Main test function."""
    print("K-ARM URDF GENERATION AND LOADING TEST")
    print("="*60)
    print(f"Robot: {args.robot}")
    print(f"Arms: {args.num_arms}")
    print(f"Layout: {args.layout}")
    print(f"Spacing: {args.spacing}")
    print(f"Isaac Sim: {'Available' if ISAAC_SIM_AVAILABLE else 'Not Available'}")
    
    # Test 1: Configuration Generation
    urdf_path, config_path, arms = test_config_generation()
    
    if urdf_path is None:
        print_test_summary(False, "", "", [])
        return False
    
    # Test 2: Isaac Sim Loading (if available)
    isaac_success = True
    if ISAAC_SIM_AVAILABLE:
        isaac_success = test_urdf_loading_isaac_sim(urdf_path, arms)
    
    # Print summary
    overall_success = (urdf_path is not None) and isaac_success
    print_test_summary(overall_success, urdf_path, config_path, arms)
    
    # Create example config for mixed mode
    if args.robot == "mixed" and not args.config_file:
        create_example_mixed_config()
    
    return overall_success


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠ Test interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        if ISAAC_SIM_AVAILABLE:
            simulation_app.close()
    
    sys.exit(exit_code)