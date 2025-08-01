

# Count cores and memory:
scontrol show job $JOBID | grep -E 'NumCPUs|Memory'
