import numpy as np
def make_3d_grid(center_position, num_points_per_axis, spacing) -> list[np.ndarray]:
    """
    Returns a list of positions (np.ndarray) of the form (x,y,z) for a 3D grid of targets.
    - center_position: Base position for grid (3D)
    - num_points_per_axis: List of [num_x, num_y, num_z] points per axis
    - spacing: List of [step_x, step_y, step_z] distances between points
    """
    targets = []
    half_x = (num_points_per_axis[0] - 1) * spacing[0] / 2
    half_y = (num_points_per_axis[1] - 1) * spacing[1] / 2
    half_z = (num_points_per_axis[2] - 1) * spacing[2] / 2
    
    for i in range(num_points_per_axis[0]):
        for j in range(num_points_per_axis[1]):
            for k in range(num_points_per_axis[2]):
                x = center_position[0] + i * spacing[0] - half_x
                y = center_position[1] + j * spacing[1] - half_y
                z = center_position[2] + k * spacing[2] - half_z
                position = np.array([x, y, z], dtype=np.float32)
                # orientation = np.array([1, 0, 0, 0], dtype=np.float32)  # default orientation
                targets.append(position)
    return targets


arrays = make_3d_grid(np.array([0,0,0]), [3,3,1], [1,1,1])
for a in arrays:
    print(a.tolist())