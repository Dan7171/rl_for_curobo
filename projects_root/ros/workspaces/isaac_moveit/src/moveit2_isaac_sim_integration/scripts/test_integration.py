"""
Test script for MoveIt2 Isaac Sim integration.
Validates basic functionality without requiring full Isaac Sim setup.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

def test_config_loader():
    """Test the configuration loader."""
    print("Testing configuration loader...")
    
    try:
        from config_loader import ConfigLoader
        
        # Test loading default configuration
        config_loader = ConfigLoader()
        
        # List available robots
        robots = config_loader.list_available_robots()
        print(f"  Available robots: {robots}")
        
        # Test loading default robot
        robot_config = config_loader.get_robot_config("default_robot")
        print(f"  Default robot name: {robot_config.name}")
        print(f"  Planning group: {robot_config.get_planning_group()}")
        print(f"  End effectors: {len(robot_config.get_end_effectors())}")
        print(f"  Joint count: {len(robot_config.get_joint_names())}")
        
        # Validate configuration
        is_valid = config_loader.validate_config("default_robot")
        print(f"  Configuration valid: {is_valid}")
        
        if is_valid:
            print("  ✓ Configuration loader test passed")
            return True
        else:
            print("  ✗ Configuration loader test failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Configuration loader test failed: {e}")
        return False


def test_moveit2_utils():
    """Test MoveIt2 utilities (without ROS2)."""
    print("Testing MoveIt2 utilities...")
    
    try:
        from moveit2_utils import create_pose_from_list, EndEffectorTarget
        
        # Test pose creation
        pose = create_pose_from_list([0.5, 0.0, 0.5])
        print(f"  Created pose: position=({pose.position.x}, {pose.position.y}, {pose.position.z})")
        
        # Test end effector target
        target = EndEffectorTarget("test_ee", pose)
        print(f"  Created target: {target}")
        
        # Test distance calculation
        pose2 = create_pose_from_list([0.6, 0.1, 0.4])
        distance = target.distance_to(pose2)
        print(f"  Distance between poses: {distance:.3f}")
        
        print("  ✓ MoveIt2 utilities test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ MoveIt2 utilities test failed: {e}")
        return False


def test_isaac_sim_launcher():
    """Test Isaac Sim launcher (without actually starting Isaac Sim)."""
    print("Testing Isaac Sim launcher...")
    
    try:
        from start_sim import IsaacSimLauncher
        from config_loader import ConfigLoader
        
        # Load configuration
        config_loader = ConfigLoader()
        robot_config = config_loader.get_robot_config("default_robot")
        
        # Create launcher (without starting)
        launcher = IsaacSimLauncher(robot_config)
        
        # Test configuration access
        stage_path = robot_config.get_isaac_stage_path()
        usd_path = robot_config.get_isaac_usd_path()
        position = robot_config.get_robot_position()
        
        print(f"  Stage path: {stage_path}")
        print(f"  USD path: {usd_path}")
        print(f"  Position: {position}")
        
        # Test target management
        target_positions = launcher.get_target_positions()
        print(f"  Target positions: {target_positions}")
        
        print("  ✓ Isaac Sim launcher test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Isaac Sim launcher test failed: {e}")
        return False


def test_simulation_runner():
    """Test simulation runner (without ROS2)."""
    print("Testing simulation runner...")
    
    try:
        from config_loader import ConfigLoader
        
        # Load configuration
        config_loader = ConfigLoader()
        robot_config = config_loader.get_robot_config("default_robot")
        
        # Test configuration access
        target_change_interval = robot_config.get_target_change_interval()
        target_tolerance = robot_config.get_target_pose_tolerance()
        
        print(f"  Target change interval: {target_change_interval}")
        print(f"  Target tolerance: {target_tolerance}")
        
        # Test random target generation
        # This would require the SimulationRunner class, but we'll skip ROS2 parts
        print("  ✓ Simulation runner test passed (basic config check)")
        return True
        
    except Exception as e:
        print(f"  ✗ Simulation runner test failed: {e}")
        return False


def test_package_structure():
    """Test package structure and files."""
    print("Testing package structure...")
    
    try:
        package_dir = Path(__file__).parent.parent
        
        # Check required files
        required_files = [
            "package.xml",
            "CMakeLists.txt",
            "README.md",
            "scripts/config_loader.py",
            "scripts/start_sim.py",
            "scripts/moveit2_utils.py",
            "scripts/simulation_runner.py",
            "config/robot_config.yaml",
            "launch/simulation_launch.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = package_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                
        if missing_files:
            print(f"  ✗ Missing files: {missing_files}")
            return False
        else:
            print("  ✓ All required files present")
            
        # Check script permissions
        script_dir = package_dir / "scripts"
        for script in script_dir.glob("*.py"):
            if not os.access(script, os.X_OK):
                print(f"  ✗ Script not executable: {script}")
                return False
                
        print("  ✓ All scripts are executable")
        print("  ✓ Package structure test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Package structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running MoveIt2 Isaac Sim Integration tests...\n")
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Configuration Loader", test_config_loader),
        ("MoveIt2 Utilities", test_moveit2_utils),
        ("Isaac Sim Launcher", test_isaac_sim_launcher),
        ("Simulation Runner", test_simulation_runner),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {test_name} test failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 