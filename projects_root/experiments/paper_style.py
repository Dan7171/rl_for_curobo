import matplotlib.pyplot as plt

# Apply "paper-ready" style
plt.rcParams.update({
    # Figure size tuned for IEEE conference one-column width (~3.5in)
    "figure.figsize": (3.5, 2.5),  
    
    # Fonts
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    
    # Lines & markers
    "lines.linewidth": 1,
    "lines.markersize": 4,
    
    # Grid & axes
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    
    # Save as tight PDF by default
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.format": "pdf"
})
