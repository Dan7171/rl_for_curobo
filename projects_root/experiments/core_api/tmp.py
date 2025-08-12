from rich.progress import Progress
import time
import numpy as np

with Progress() as progress:
    task1 = progress.add_task("Discrete Range", total=400)
    task2 = progress.add_task("Continuous Range", total=10)
    task3 = progress.add_task("Another Range", total=50)

    for _ in range(400):
        time.sleep(0.05)
        progress.update(task1, advance=1)

    for _ in np.arange(0, 1.0, 0.1):
        time.sleep(0.05)
        progress.update(task2, advance=1)

    for _ in range(50):
        time.sleep(0.01)
        progress.update(task3, advance=1)