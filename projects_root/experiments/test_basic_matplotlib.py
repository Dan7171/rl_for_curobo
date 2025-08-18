#!/usr/bin/env python3
"""
Basic matplotlib test to check if plot windows appear.
"""

import matplotlib
print(f"Matplotlib backend: {matplotlib.get_backend()}")

import matplotlib.pyplot as plt
import numpy as np
import time

# Force interactive mode
plt.ion()
print(f"Interactive mode: {plt.isinteractive()}")

# Create a simple test plot
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
ax.set_title('Basic Matplotlib Test - Do you see this window?')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show(block=False)

print("=" * 60)
print("🎯 CRITICAL QUESTION: Do you see a matplotlib window with a sine wave?")
print("   - If YES: The plotting system should work, issue is elsewhere")
print("   - If NO: We have a display/backend issue to fix")
print("=" * 60)

# Keep updating the plot for 10 seconds
for i in range(50):
    # Update the plot
    y_new = np.sin(x + i * 0.1)
    ax.clear()
    ax.plot(x, y_new, 'b-', linewidth=2, label=f'sin(x + {i*0.1:.1f})')
    ax.set_title(f'Animated Test - Frame {i}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    
    plt.draw()
    plt.pause(0.1)
    
    if i % 10 == 0:
        print(f"Frame {i}/50 - Still updating...")

print("\n✓ Test completed. Did you see an animated sine wave?")
plt.show()  # Keep window open
input("Press Enter to close...")
plt.close('all') 