# Custom Cost Terms for CuRobo

This directory provides templates and examples for creating custom cost terms in CuRobo's rollout optimization system.

## Quick Start

The CuRobo custom cost system supports two approaches:

### 1. **Explicit Configuration (Recommended)**

This is the **default behavior** - only custom costs explicitly listed in your YAML configuration will be loaded. This gives you full control over which costs are used and their parameters.

**Steps:**
1. Create your custom cost file in the appropriate directory:
   - `curobo/src/curobo/rollout/cost/custom/arm_base/` for general robot costs
   - `curobo/src/curobo/rollout/cost/custom/arm_reacher/` for task-specific costs

2. Add explicit configuration to your YAML file:

```yaml
cost:
  # ... other cost configurations ...
  
  custom:
    arm_base:
      my_energy_cost:
        module_path: "curobo.rollout.cost.custom.arm_base.energy_cost"
        class_name: "EnergyCost"
        config_class_name: "EnergyCostConfig"
        weight: 10.0
        terminal: false
        
    arm_reacher:
      my_task_cost:
        module_path: "curobo.rollout.cost.custom.arm_reacher.task_cost"
        class_name: "TaskCost"
        config_class_name: "TaskCostConfig" 
        weight: 25.0
        terminal: false
```

3. **Only the costs you explicitly configure will be loaded** - this gives you precise control.

### 2. **Auto-Discovery (Advanced)**

Auto-discovery automatically finds and loads all custom cost files in the directories. This is disabled by default.

**To enable auto-discovery:**
1. Set `enable_auto_discovery: true` in your solver configuration
2. Don't include a `custom:` section in your YAML (or leave it empty)
3. All `.py` files in the custom directories will be automatically loaded with default parameters

**Note:** Auto-discovery uses default weights and parameters. Use explicit configuration for fine-tuned control.

## Directory Structure

```
curobo/src/curobo/rollout/cost/custom/
├── arm_base/                    # General robot costs (velocity, energy, etc.)
│   ├── __init__.py
│   └── energy_cost.py          # Example: Energy minimization cost
├── arm_reacher/                 # Task-specific costs (pose tracking, etc.)
│   ├── __init__.py
│   └── task_cost.py            # Example: Task-specific pose cost
└── custom_cost_term_template/   # Templates and examples
    ├── README.md               # This file
    ├── custom_cost_template.py # Template for creating new costs
    └── example_particle_mpc.yml # Example configuration
```

## When to Use Each Directory

### `arm_base/` - General Robot Costs
Use for costs that apply to general robot behavior:
- Energy minimization
- Smoothness penalties
- Joint limit avoidance
- Velocity/acceleration constraints
- Self-collision avoidance enhancements

### `arm_reacher/` - Task-Specific Costs  
Use for costs specific to reaching/manipulation tasks:
- End-effector pose tracking
- Workspace constraints
- Orientation preferences
- Task-specific collision avoidance
- Multi-arm coordination

## Creating Custom Costs

### Step 1: Create Your Cost Class

Use `custom_cost_template.py` as a starting point. Your cost must:

1. **Inherit from `CostBase`**
2. **Have a corresponding config class inheriting from `CostConfig`**
3. **Implement the `forward()` method**

```python
from curobo.rollout.cost.cost_base import CostBase, CostConfig

class MyCostConfig(CostConfig):
    def __init__(self, my_parameter: float = 1.0, **kwargs):
        self.my_parameter = my_parameter
        super().__init__(**kwargs)

class MyCost(CostBase):
    def __init__(self, config: MyCostConfig):
        super().__init__(config)
        self._my_parameter = config.my_parameter
    
    def forward(self, state):
        # Implement your cost computation
        cost = compute_my_cost(state, self._my_parameter)
        return self._weight * cost
```

### Step 2: Choose Configuration Method

**For precise control (recommended):**
Add explicit configuration to your YAML as shown above.

**For quick testing:**
Enable auto-discovery in your solver config and place your file in the appropriate directory.

## Examples

The system includes example custom costs:

- **`arm_base/energy_cost.py`**: Energy minimization cost with velocity/acceleration penalties
- **`arm_reacher/task_cost.py`**: Task-specific pose cost for enhanced end-effector tracking

## Configuration Parameters

When using explicit configuration, you can specify:

- `module_path`: Path to your Python module
- `class_name`: Name of your cost class
- `config_class_name`: Name of your config class (optional)
- `weight`: Cost weight (required)
- `terminal`: Whether this is a terminal cost (optional, default: false)
- Any custom parameters your cost needs

## Best Practices

1. **Use explicit configuration for production code** - it's more predictable and maintainable
2. **Start with low weights** and tune gradually
3. **Test your costs in isolation** before combining with others
4. **Follow the naming convention**: `YourCost` and `YourCostConfig`
5. **Document your custom parameters** in the config class
6. **Use appropriate tensor operations** for GPU compatibility

## Troubleshooting

**"Custom cost not loaded":**
- Check that the module path is correct
- Verify class names match exactly (case-sensitive)
- Ensure your config class inherits from `CostConfig`
- Check that the cost is explicitly configured in your YAML

**"Import errors":**
- Verify the module path starts with `curobo.rollout.cost.custom`
- Check that `__init__.py` files exist in the directories
- Ensure all dependencies are imported correctly

**"Cost has no effect":**
- Check the weight value - it might be too small
- Verify the cost is enabled (`cost_instance.enabled`)
- Use live plotting to monitor cost values during optimization

## Integration with Existing Systems

Custom costs integrate seamlessly with CuRobo's existing cost system:
- They work with MPPI, iLQR, and other optimizers
- Compatible with CUDA graphs for performance
- Support batched operations for multiple trajectories
- Work with live plotting and debugging tools

For more advanced usage and examples, see the template files in this directory. 