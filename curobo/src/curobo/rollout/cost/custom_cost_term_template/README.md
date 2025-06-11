# Custom Cost Term Template

This directory provides a template and guide for creating custom cost terms in CuRobo's MPC solver.

## Overview

CuRobo's cost framework allows you to add custom cost terms that integrate seamlessly with the existing optimization pipeline. Custom cost terms can be categorized into two types:

1. **arm_base costs**: General costs not tied to specific tasks (e.g., energy minimization, smoothness)
2. **arm_reacher costs**: Task-specific costs (e.g., custom pose objectives, task constraints)

## Quick Start

1. Copy `custom_cost_template.py` and rename it (e.g., `my_custom_cost.py`)
2. Implement your cost logic in the `forward` method
3. Add your cost term to the config file under `cost/custom/arm_base/` or `cost/custom/arm_reacher/`
4. Enable/disable your cost via the `weight` parameter (weight=0 disables the cost)

## File Structure

```
custom_cost_term_template/
├── README.md                    # This file - usage guide
├── custom_cost_template.py      # Template to copy and modify
├── example_custom_cost.py       # Working examples
└── example_particle_mpc.yml     # Example config file
```

## Creating a Custom Cost Term

### Step 1: Create Your Cost Class

Copy `custom_cost_template.py` and modify:

```python
from dataclasses import dataclass
from typing import Optional
import torch
from curobo.rollout.cost.cost_base import CostBase, CostConfig

@dataclass
class MyCustomCostConfig(CostConfig):
    """Configuration for my custom cost."""
    my_parameter: float = 1.0
    my_threshold: float = 0.5

class MyCustomCost(CostBase):
    """My custom cost implementation."""
    
    def __init__(self, config: MyCustomCostConfig):
        super().__init__(config)
    
    def forward(self, state):
        # Your cost computation here
        # state is of type KinematicModelState
        # Return tensor of shape [batch, horizon] or [batch]
        cost = torch.zeros(
            (state.state_seq.position.shape[0], state.state_seq.position.shape[1]),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype
        )
        # ... implement your cost logic ...
        return self.weight * cost
```

### Step 2: Configure in YAML

Add to your `particle_mpc.yml` config:

```yaml
cost:
  # ... other standard cost terms ...
  
  custom:
    arm_base:  # For general robot costs
      my_energy_cost:
        module_path: "path.to.my_module"
        class_name: "MyCustomCost"
        config_class_name: "MyCustomCostConfig"  # Optional
        weight: 10.0
        my_parameter: 2.0
        my_threshold: 0.3
        terminal: false
        
    arm_reacher:  # For task-specific costs
      my_task_cost:
        module_path: "path.to.my_module"
        class_name: "MyTaskCost"
        weight: 50.0
        task_specific_param: 1.5
        terminal: true
```

### Step 3: Cost Type Guidelines

**arm_base costs** should be used for:
- Energy minimization
- Smoothness constraints
- General robot health metrics
- Hardware limits
- Collision avoidance (if not task-specific)

**arm_reacher costs** should be used for:
- Task-specific objectives
- End-effector constraints
- Manipulation objectives
- Goal-dependent costs

## Configuration Parameters

### Required Parameters
- `module_path`: Python import path to your module
- `class_name`: Name of your cost class
- `weight`: Cost weight (0 = disabled, >0 = enabled)

### Optional Parameters
- `config_class_name`: Name of your config class (defaults to CostConfig)
- `terminal`: Apply cost only to terminal state (default: false)
- Any parameters specific to your cost function

## Working Examples

See `example_custom_cost.py` for working implementations:

1. **RandomCost**: Simple example for testing
2. **VelocitySmoothnessCost**: Penalizes velocity changes  
3. **EndEffectorRegionCost**: Keeps end-effector in/out of regions

## Integration Details

The custom cost framework automatically:

1. **Loads your classes** dynamically at runtime
2. **Initializes costs** during solver setup
3. **Executes costs** during optimization with proper profiling
4. **Handles errors** gracefully with logging
5. **Supports live plotting** if enabled

## Cost Function Requirements

Your cost function's `forward` method must:

1. **Accept a state parameter** of type `KinematicModelState`
2. **Return a tensor** of shape `[batch, horizon]` or `[batch]`
3. **Use proper device/dtype** from `self.tensor_args`
4. **Apply weight** using `self.weight`
5. **Handle batch dimensions** correctly

## Available State Information

The `state` parameter provides access to:

```python
state.state_seq.position     # Joint positions [batch, horizon, dof]
state.state_seq.velocity     # Joint velocities [batch, horizon, dof]  
state.state_seq.acceleration # Joint accelerations [batch, horizon, dof]
state.state_seq.jerk         # Joint jerks [batch, horizon, dof]
state.ee_pos_seq            # End-effector positions [batch, horizon, 3]
state.ee_quat_seq           # End-effector quaternions [batch, horizon, 4]
state.robot_spheres         # Robot collision spheres
state.link_pose             # All link poses (dict)
```

## Error Handling

The framework handles errors gracefully:
- Failed imports log errors but don't crash
- Runtime cost errors are logged and skipped
- Missing parameters use defaults where possible

## Performance Tips

1. **Vectorize operations** across batch and horizon dimensions
2. **Minimize Python loops** in the forward method
3. **Reuse tensors** when possible
4. **Use proper tensor device/dtype** from `self.tensor_args`
5. **Profile your costs** using the built-in profiler

## Debugging

Enable debug information by:
1. Setting higher log levels
2. Using `print()` statements in your cost function
3. Examining cost values in live plotting
4. Checking tensor shapes and devices

## Examples in Practice

### Energy Minimization Cost
```python
def forward(self, state):
    velocities = state.state_seq.velocity  # [batch, horizon, dof]
    energy = torch.sum(velocities**2, dim=-1)  # [batch, horizon]
    return self.weight * energy
```

### End-Effector Boundary Cost
```python
def forward(self, state):
    ee_pos = state.ee_pos_seq  # [batch, horizon, 3]
    distance_from_center = torch.norm(ee_pos - self.center, dim=-1)
    violation = torch.relu(distance_from_center - self.radius)
    return self.weight * violation
```

### Smoothness Cost
```python  
def forward(self, state):
    velocities = state.state_seq.velocity  # [batch, horizon, dof]
    vel_diff = velocities[:, 1:] - velocities[:, :-1]  # [batch, horizon-1, dof]
    smoothness = torch.sum(vel_diff**2, dim=-1)  # [batch, horizon-1]
    # Pad to match horizon dimension
    padded = torch.cat([smoothness, smoothness[:, -1:]], dim=1)
    return self.weight * padded
```

This framework provides a powerful way to extend CuRobo's optimization with custom objectives while maintaining performance and integration with existing tools.

## Live Plotting Integration

Your custom cost terms will automatically appear in CuRobo's live plotting feature with **clear, descriptive labels** based on their class names from the configuration.

### Automatic Plot Labels

When you enable live plotting (`live_plotting: true` in your config), custom costs are automatically labeled using their `class_name` field:

**Configuration:**
```yaml
cost:
  custom:
    arm_base:
      my_velocity_cost:
        module_path: "curobo.rollout.cost.custom_cost_term_template.example_custom_cost"
        class_name: "VelocitySmoothnessCost"  # This becomes the plot label
        weight: 20.0
        
    arm_reacher:
      my_region_cost:
        module_path: "curobo.rollout.cost.custom_cost_term_template.example_custom_cost"
        class_name: "EndEffectorRegionCost"  # This becomes the plot label
        weight: 30.0
```

**Plot Labels Generated:**
- `Custom Base: VelocitySmoothnessCost` (for arm_base costs)
- `Custom Reacher: EndEffectorRegionCost` (for arm_reacher costs)

### Visual Styling

Custom costs get special visual treatment in the plots:
- **Square markers** (instead of circles) to distinguish them from built-in costs
- **Slightly thicker lines** for better visibility
- **Unique colors** assigned automatically
- **Clear prefixes**: "Custom Base:" or "Custom Reacher:"

### Example Plot Output

When running with custom costs enabled, you'll see output like:
```
Active cost components: [('Bound Cost', '0.000123'), ('Custom Base: VelocitySmoothnessCost', '0.045678'), ...]
Custom costs detected: ['Custom Base: VelocitySmoothnessCost', 'Custom Reacher: EndEffectorRegionCost']
```

### Benefits

1. **No Manual Configuration**: Cost labels are automatically extracted from your `class_name`
2. **Clear Identification**: Easy to distinguish custom costs from built-in ones
3. **Debug-Friendly**: Immediate visual feedback on which custom costs are active
4. **Performance Monitoring**: Track how your custom costs behave during optimization

### Usage Tips

- Use descriptive `class_name` values since they become plot labels
- Custom costs appear in the same plot as built-in costs for easy comparison
- Plotting frequency can be adjusted with `set_plot_frequency(k)` method
- All enabled custom costs are automatically included - no manual plotting code needed 