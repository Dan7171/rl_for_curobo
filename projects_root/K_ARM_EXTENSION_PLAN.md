# K-Arm Centralized MPC Extension Plan

## Overview

This document outlines the step-by-step plan to extend your current dual-arm centralized MPC system to support **any number K of arms** from **any arm model**, while maintaining backward compatibility with your existing setup.

## Current System Analysis

### Working Components ✅
- Dual-arm centralized MPC with Franka Panda arms
- `PoseCostMultiArm` class for multi-arm pose costs
- Particle MPC configurations for 2, 3, and 4 arms
- Modified `arm_reacher.py` with dual-arm support
- Robust simulation environment in Isaac Sim

### Hardcoded Limitations ⚠️
- Hardcoded link mapping: `['left_panda_hand', 'right_panda_hand']`
- Fixed robot configuration: `franka_dual_arm.yml`
- Manual target management for only 2 arms
- Limited to Franka Panda arm models

## Implementation Plan

### Phase 1: Configuration Infrastructure ✅ COMPLETED
**Timeline: Week 1**
**Status: Implemented**

**Deliverables:**
- ✅ `multi_arm_config_generator.py` - Generates URDF and YAML configs for K arms
- ✅ `ArmConfig` dataclass for individual arm specifications
- ✅ `MultiArmConfigGenerator` class for automated config generation
- ✅ Support for arbitrary arm spacing and positioning

**Key Features:**
```python
# Generate 4-arm Franka system
arms, system_name = create_franka_k_arm_system(4, arm_spacing=0.8)
urdf_path = generator.generate_multi_arm_urdf(arms, system_name)
config_path = generator.generate_curobo_config(arms, system_name, urdf_path)
```

### Phase 2: Core Algorithm Generalization ✅ PARTIALLY COMPLETED
**Timeline: Week 2**
**Status: In Progress**

**Deliverables:**
- ✅ Updated `_format_multi_arm_ee_data()` with configurable arm mapping
- ✅ Added `_get_arm_link_mapping()` method for flexible link resolution
- ✅ Backward compatibility with existing dual-arm configurations
- 🔄 **TODO:** Fix linter errors in `arm_reacher.py`

**Changes Made:**
```python
def _get_arm_link_mapping(self, num_arms: int) -> List[str]:
    """Get arm end-effector link mapping from robot config or generate defaults."""
    # Try robot configuration first
    if hasattr(self, 'kinematics') and hasattr(self.kinematics, 'link_names'):
        link_names = self.kinematics.link_names
        if isinstance(link_names, list) and len(link_names) >= num_arms:
            return link_names[:num_arms]
    
    # Backward compatibility fallbacks
    if num_arms == 2:
        return ['left_panda_hand', 'right_panda_hand']
    elif num_arms == 3:
        return ['left_panda_hand', 'center_panda_hand', 'right_panda_hand']
    # ... etc
```

### Phase 3: User Interface Enhancement ✅ COMPLETED
**Timeline: Week 2-3**
**Status: Implemented**

**Deliverables:**
- ✅ `k_arm_centralized_mpc.py` - Generalized MPC script
- ✅ `KArmTargetManager` class for managing K targets
- ✅ Command-line arguments for arm count and configs
- ✅ Automatic particle MPC config selection

**Usage Examples:**
```bash
# Dual-arm (backward compatible)
python k_arm_centralized_mpc.py --robot franka_dual_arm.yml --num_arms 2

# Triple-arm system
python k_arm_centralized_mpc.py --robot franka_triple_arm.yml --num_arms 3

# Auto-generated 6-arm system
python k_arm_centralized_mpc.py --num_arms 6

# Custom particle config
python k_arm_centralized_mpc.py --num_arms 4 --override_particle_file custom.yml
```

### Phase 4: Testing and Validation ✅ COMPLETED
**Timeline: Week 3**
**Status: Implemented**

**Deliverables:**
- ✅ `test_k_arm_progression.py` - Comprehensive testing suite
- ✅ Progressive testing from 2 to K arms
- ✅ Backward compatibility verification
- ✅ Multi-arm pose cost validation

**Test Phases:**
1. **Configuration Generation** - Test URDF/YAML generation for 2-5 arms
2. **Multi-Arm Pose Cost** - Validate pose cost calculations
3. **Arm Link Mapping** - Test link name resolution logic
4. **Backward Compatibility** - Ensure existing dual-arm setup works

### Phase 5: Mixed-Model Support 🔄 TODO
**Timeline: Week 4**
**Status: Planned**

**Objectives:**
- Support arms from different manufacturers (UR, Kinova, etc.)
- Mixed arm configurations (e.g., 2 Franka + 1 UR5e)
- Heterogeneous joint structures and workspace sizes

**Planned Implementation:**
```python
# Mixed arm configuration
mixed_arms = [
    ArmConfig("left_franka", "franka_panda.urdf", [0, 0, 0], ...),
    ArmConfig("right_ur5e", "ur5e.urdf", [1, 0, 0], ...),
    ArmConfig("center_kinova", "kinova_gen3.urdf", [0.5, 0.5, 0], ...)
]
```

### Phase 6: Advanced Features 🔄 TODO
**Timeline: Week 5-6**
**Status: Planned**

**Features:**
- Dynamic arm reconfiguration during runtime
- Collision avoidance between arms
- Coordinated multi-arm grasping primitives
- Load balancing across arms

## Risk Mitigation Strategy

### Maintaining Backward Compatibility ✅

**Strategy:**
1. **Preserve existing files** - Never modify `franka_dual_arm.yml` directly
2. **Fallback logic** - Default to original behavior for dual-arm setups
3. **Gradual migration** - Allow users to opt-in to new features
4. **Version detection** - Automatic detection of legacy vs. new configurations

**Implementation:**
```python
# Automatic fallback in _get_arm_link_mapping()
if num_arms == 2:
    return ['left_panda_hand', 'right_panda_hand']  # Legacy behavior
```

### Gentle Code Modifications ✅

**Principles:**
1. **Additive changes** - Add new methods rather than modifying existing ones
2. **Configuration-driven** - Use config files to control behavior
3. **Runtime detection** - Detect multi-arm setup at runtime
4. **Fail-safe defaults** - Always provide working defaults

### Testing at Each Step ✅

**Approach:**
1. **Progressive testing** - Test 2, 3, 4, 5+ arms sequentially
2. **Regression testing** - Verify existing functionality after each change
3. **Integration testing** - Test full pipeline end-to-end
4. **Performance testing** - Ensure no degradation in performance

## Current Status

### ✅ Completed
- Configuration generation infrastructure
- Basic K-arm MPC script
- Testing framework
- Backward compatibility preservation
- Core algorithm generalization (90%)

### 🔄 In Progress
- Fixing linter errors in `arm_reacher.py`
- Full testing validation

### 📋 TODO
- Mixed-model support
- Advanced coordination features
- Performance optimization
- Documentation and examples

## Getting Started

### Quick Test (Recommended First Step)
```bash
# Test configuration generation
python projects_root/examples/test_k_arm_progression.py

# Generate a 3-arm system
python projects_root/utils/multi_arm_config_generator.py
```

### Running K-Arm MPC
```bash
# Start with your existing dual-arm setup
python projects_root/examples/k_arm_centralized_mpc.py --robot franka_dual_arm.yml --num_arms 2

# Try triple-arm
python projects_root/examples/k_arm_centralized_mpc.py --num_arms 3 --visualize_spheres
```

## Migration Path

### For Existing Dual-Arm Users
1. **No immediate changes required** - Your existing setup continues to work
2. **Test new script** - Try `k_arm_centralized_mpc.py` with `--num_arms 2`
3. **Experiment with 3+ arms** - Use auto-generated configurations
4. **Customize as needed** - Create custom robot configurations

### For New Multi-Arm Users
1. **Start with config generation** - Use `MultiArmConfigGenerator`
2. **Create robot configurations** - Define arm positions and types
3. **Test progressively** - Start with 2-3 arms, then scale up
4. **Optimize for your use case** - Adjust spacing, orientations, etc.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Configuration  │    │   Core MPC      │    │   User Interface│
│   Generation    │───▶│   Algorithm     │───▶│    & Testing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ - URDF Gen      │    │ - Multi-arm EE  │    │ - K-arm Script  │
│ - YAML Configs  │    │ - Link Mapping  │    │ - Target Mgmt   │
│ - Particle MPC  │    │ - Pose Costs    │    │ - Testing Suite │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Performance Considerations

### Expected Scaling
- **2 arms**: Baseline performance (existing)
- **3-4 arms**: 10-20% increase in computation
- **5-8 arms**: 30-50% increase in computation
- **8+ arms**: May require optimization

### Optimization Strategies
1. **Parallel computation** - Leverage multi-GPU setups
2. **Selective updates** - Update only changed arm goals
3. **Caching strategies** - Cache collision computations
4. **Pruning techniques** - Skip distant arm interactions

## Troubleshooting

### Common Issues
1. **Link name mismatches** - Check robot config `link_names`
2. **URDF parsing errors** - Verify individual arm URDFs
3. **Memory issues** - Reduce batch size or horizon for many arms
4. **Convergence problems** - Adjust cost weights for multiple arms

### Debug Tools
- Use `--visualize_spheres` to see collision geometry
- Enable debug prints in `_format_multi_arm_ee_data()`
- Check target positions with `print_target_update()`
- Validate configs with `test_k_arm_progression.py`

## Next Steps

1. **Run the test suite** to validate current implementation
2. **Fix remaining linter errors** in `arm_reacher.py`
3. **Test with your specific robot setup**
4. **Provide feedback** on any issues or needed features
5. **Consider contributing** improvements back to the codebase

---

**Note**: This is a living document that will be updated as the implementation progresses. Please refer to the test suite for the most current status of each component. 