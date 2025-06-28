# Isaac Sim Testing Scripts

This directory contains two scripts to test Isaac Sim installation and startup:

## Scripts

### 1. `test_isaacsim_simple.py`
A minimal script that just starts Isaac Sim and keeps it running, similar to `./isaac-sim.sh`.

### 2. `test_isaacsim.py` 
A more complete script that starts Isaac Sim with a basic scene (ground plane and red cube).

## How to Run

### Prerequisites
- Isaac Sim must be installed
- You need to run these scripts from within the Isaac Sim Python environment

### Method 1: Using Isaac Sim's Python
```bash
# Navigate to your Isaac Sim installation
cd /path/to/isaac-sim

# Run the script using Isaac Sim's Python interpreter
./python.sh /path/to/your/project/projects_root/examples/test_isaacsim_simple.py
```

### Method 2: Using Isaac Sim Script Runner
```bash
# If you have Isaac Sim in your PATH
isaac-sim --exec /path/to/your/project/projects_root/examples/test_isaacsim_simple.py
```

### Method 3: From Isaac Sim Source Directory
```bash
# If running from Isaac Sim source
cd /path/to/isaac-sim
./isaac-sim.sh --exec /path/to/your/project/projects_root/examples/test_isaacsim_simple.py
```

## Expected Behavior

1. **Simple version**: Opens Isaac Sim with an empty scene and keeps it running
2. **Full version**: Opens Isaac Sim with a ground plane and red cube

Both scripts will:
- Print confirmation messages to the console
- Keep Isaac Sim running until you close the window
- Handle Ctrl+C interruption gracefully
- Clean up properly on exit

## Troubleshooting

- **Import errors**: Normal when not running in Isaac Sim environment - the scripts need to be run with Isaac Sim's Python interpreter
- **Window doesn't appear**: Try setting `headless: False` explicitly
- **Script exits immediately**: Check that Isaac Sim is properly installed and the environment is set up correctly

## Configuration

You can modify these settings in the scripts:
- `headless`: Set to `True` for no GUI (headless mode)
- `width`/`height`: Adjust window size
- Add more objects to the scene in the full version 