from rich.progress import Progress
import time
import numpy as np

# with Progress() as progress:
#     task1 = progress.add_task("Discrete Range", total=400)
#     task2 = progress.add_task("Continuous Range", total=10)
#     task3 = progress.add_task("Another Range", total=50)

#     for _ in range(400):
#         time.sleep(0.05)
#         progress.update(task1, advance=1)

#     for _ in np.arange(0, 1.0, 0.1):
#         time.sleep(0.05)
#         progress.update(task2, advance=1)

#     for _ in range(50):
#         time.sleep(0.01)
#         progress.update(task3, advance=1)


def recursive_fill_from_default(a_cfg, default_cfg):
    for key in default_cfg:
        if key not in a_cfg:
            a_cfg[key] = default_cfg[key]
        elif isinstance(a_cfg[key], dict):
            recursive_fill_from_default(a_cfg[key], default_cfg[key])
    return a_cfg

a_cfg = {
    'a': 1,
    'b': {
        'c': 0,
        'd': 0
    },  
    'h': 4
}

default_cfg = {
    'a': 1,
    'b': {
        'c': 2,
        'd': 3,
        'e': {
            'f': 4
        },
        'g': 5
    },
    'h': 6,
    'i': 7
}

recursive_fill_from_default(a_cfg, default_cfg)
print(a_cfg)