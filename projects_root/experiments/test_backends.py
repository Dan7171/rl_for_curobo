#!/usr/bin/env python3
"""
Test different matplotlib backends to find one that works.
"""

import matplotlib
import os

# List of backends to try
backends = ['Qt5Agg', 'TkAgg', 'GTK3Agg', 'Qt4Agg', 'GTKAgg']

print("Testing different matplotlib backends...")
print(f"Current backend: {matplotlib.get_backend()}")
print(f"DISPLAY environment: {os.environ.get('DISPLAY', 'Not set')}")

for backend in backends:
    try:
        print(f"\n--- Testing {backend} ---")
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title(f'Test with {backend}')
        plt.show(block=False)
        
        print(f"✓ {backend} works! Creating window...")
        plt.pause(2)  # Show for 2 seconds
        plt.close('all')
        
        print(f"🎯 SUCCESS: {backend} backend works!")
        print(f"To use this backend, run: export MPLBACKEND={backend}")
        break
        
    except Exception as e:
        print(f"❌ {backend} failed: {e}")
        plt.close('all')  # Clean up
        continue
else:
    print("\n❌ No working backend found!")
    print("You may need to install GUI libraries:")
    print("  sudo apt-get install python3-tk python3-pyqt5") 