# Custom Cost Terms

This directory contains template and example files for creating custom cost terms in CuRobo. Custom cost terms allow you to add domain-specific penalties and rewards to the optimization process.

## Quick Start (New Auto-Discovery Method)

### Option 1: Auto-Discovery (Recommended)
Simply drop your custom cost files into the appropriate directory and they will be automatically discovered and loaded:

- **For general robot costs** (energy, smoothness, etc.): 
  - Place files in: `curobo/src/curobo/rollout/cost/custom/arm_base/`
  - Example: `energy_minimization.py`, `smoothness_cost.py`

- **For task-specific costs** (end-effector constraints, workspace regions, etc.):
  - Place files in: `curobo/src/curobo/rollout/cost/custom/arm_reacher/`
  - Example: `workspace_constraint.py`, `target_tracking.py`

**Requirements for auto-discovery:**
1. Your file must contain a cost class that inherits from `CostBase`
2. Your file must contain a corresponding config class that inherits from `CostConfig`
3. The config class name must be: `{CostClassName}Config`

**Example:**
```python
# File: curobo/src/curobo/rollout/cost/custom/arm_base/my_custom_cost.py

from dataclasses import dataclass
import torch
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState

@dataclass
class MyCustomCostConfig(CostConfig):
    my_parameter: float = 1.0

class MyCustomCost(CostBase, MyCustomCostConfig):
    def __init__(self, config: MyCustomCostConfig):
        MyCustomCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
    
    def forward(self, state: KinematicModelState, **kwargs) -> torch.Tensor:
        # Your cost computation here
        return self._weight * torch.zeros((1, 1))  # Replace with actual cost
```

That's it! The cost will be automatically discovered and loaded with default parameters.

### Option 2: Explicit Configuration (Legacy Method)
You can still explicitly configure custom costs in your YAML configuration:

```yaml
custom:
  arm_base:
    my_custom_cost:
      module_path: "path.to.your.module"
      class_name: "MyCustomCost"
      config_class_name: "MyCustomCostConfig"  # Optional
      weight: 50.0
      my_parameter: 2.0
      terminal: False
```

## Directory Structure

```
curobo/src/curobo/rollout/cost/
├── custom_cost_term_template/          # Templates and examples
│   ├── README.md                       # This file
│   ├── custom_cost_template.py         # Template for creating costs
│   └── example_custom_cost.py          # Working examples
└── custom/                             # Auto-discovery directories
    ├── arm_base/                       # General robot costs
    │   ├── __init__.py
    │   └── energy_cost.py              # Example energy cost
    └── arm_reacher/                    # Task-specific costs
        ├── __init__.py
        └── task_cost.py                # Example task cost
```

## When to Use arm_base vs arm_reacher

### arm_base (General Robot Costs)
Use for costs that apply to general robot behavior, independent of specific tasks:
- Energy minimization
- Smoothness constraints  
- Joint velocity/acceleration limits
- Robot-specific safety constraints
- Dynamic constraints

### arm_reacher (Task-Specific Costs)
Use for costs that are specific to manipulation tasks:
- End-effector position/orientation costs
- Workspace constraints
- Task-specific obstacle avoidance
- Tool orientation constraints
- Task-dependent rewards

## Creating Custom Cost Terms

### 1. Copy the Template
Start with `custom_cost_template.py` as your base:

```bash
# Copy to arm_base for general costs
cp custom_cost_template.py ../custom/arm_base/my_cost.py

# Copy to arm_reacher for task-specific costs  
cp custom_cost_template.py ../custom/arm_reacher/my_task_cost.py
```

### 2. Implement Your Cost Function
The key method to implement is `forward()`:

```python
def forward(self, state: KinematicModelState, **kwargs) -> torch.Tensor:
    """
    Compute your custom cost.
    
    Args:
        state: Contains robot state (joint positions, velocities, etc.)
        **kwargs: Additional arguments (important for arm_reacher compatibility)
        
    Returns:
        torch.Tensor: Cost values of shape [batch_size, horizon]
    """
    # Access robot state
    positions = state.state_seq.position      # [batch, horizon, n_dofs]
    velocities = state.state_seq.velocity     # [batch, horizon, n_dofs] 
    accelerations = state.state_seq.acceleration  # [batch, horizon, n_dofs]
    
    # For arm_reacher, you can also access:
    ee_positions = state.ee_pos_seq          # [batch, horizon, 3]
    ee_quaternions = state.ee_quat_seq       # [batch, horizon, 4]
    
    # Compute your cost
    cost = torch.zeros_like(positions[:, :, 0])  # [batch, horizon]
    
    return self._weight * cost
```

### 3. Configuration Class
Define parameters for your cost:

```python
@dataclass 
class MyCostConfig(CostConfig):
    """Configuration for my custom cost."""
    
    # Add your parameters here
    scale_factor: float = 1.0
    enable_feature: bool = True
    target_value: float = 0.0
    
    def __post_init__(self):
        return super().__post_init__()
```

## Advanced Features

### Disabling Auto-Discovery
If you want to disable auto-discovery (e.g., for performance in production):

```python
# In your configuration loading code
cost_cfg = ArmCostConfig.from_dict(
    cost_dict, 
    robot_config, 
    enable_auto_discovery=False
)
```

### Mixing Auto-Discovery with Explicit Configuration
- Auto-discovered costs get default parameters
- Explicitly configured costs override auto-discovered ones
- You can mix both approaches in the same configuration

### Override Auto-Discovered Parameters
Create a custom configuration to override default parameters of auto-discovered costs:

```yaml
custom:
  arm_base:
    auto_energy_cost_EnergyCost:  # Auto-discovered cost name format
      weight: 100.0  # Override default weight
      energy_scale: 2.0  # Override parameter
```

## Examples

See the example files:
- `example_custom_cost.py` - Complete working examples
- `custom/arm_base/energy_cost.py` - Energy minimization cost
- `custom/arm_reacher/task_cost.py` - Task-specific pose cost

## Best Practices

1. **Tensor Operations**: Always ensure your operations work with batched tensors
2. **Device Handling**: Use `self.tensor_args.device` and `self.tensor_args.dtype`
3. **Error Handling**: Add type checks for tensor inputs
4. **Documentation**: Document your cost function's purpose and parameters
5. **Testing**: Test with different batch sizes and horizons
6. **Naming**: Use descriptive names for your cost files and classes

## Troubleshooting

### Cost Not Loading
- Check that your file is in the correct directory (`arm_base` or `arm_reacher`)
- Ensure your class inherits from `CostBase`
- Verify the config class name follows the pattern: `{CostClass}Config`
- Check the CuRobo logs for auto-discovery messages

### Runtime Errors
- Verify tensor shapes match expected dimensions
- Check that all operations are differentiable (PyTorch compatible)
- Ensure proper device placement of tensors

### Performance Issues
- Minimize expensive operations in the forward pass
- Use efficient tensor operations
- Consider disabling auto-discovery in production for faster startup 