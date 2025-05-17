import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import Gf


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def integrate_quat(q, omega, dt):
    """
    Compute the next quaternion from the current quaternion and the angular velocity.
    i.e computes q' from q, omega and dt.
    This is a simple integration of the angular velocity.

    Args:
        q (np.ndarray):current quaternion. [w,x,y,z] and not [x,y,z,w] ("scalar-first" format).
        omega (np.ndarray): angular velocity vector in rad/s.
        dt (float): step duration (time in which the angular velocity is integrated).

    Returns:
        q' (np.ndarray): quaternion after integration.
    """
    omega_quat = np.array([0.0, *omega])
    dq = 0.5 * quat_mul(q, omega_quat)
    q_new = q + dq * dt
    return q_new / np.linalg.norm(q_new)




def isaacsim_euler2quat(rx, ry, rz, degrees=True, order="XYZ") -> np.ndarray:
    """
    Convert Euler angles to quaternion.
    Args:
        rx (float): Rotation around X axis in degs/radians.
        ry (float): Rotation around Y axis in degs/radians.
        rz (float): Rotation around Z axis in degs/radians.
        degrees (bool): If True, the Euler angles are in degrees (and not radians).
        order (str): Rotation order. Defaults to "XYZ".
        
    Returns:
        np.ndarray: Quaternion as [w,x,y,z]
    """
    
    
    return euler_angles_to_quat(np.array([rx, ry, rz]), degrees=degrees, extrinsic=(order == "ZYX"))

def euler_to_quaternion(rx, ry, rz, order="XYZ"):
    """
    Convert Euler angles to quaternion.

    Args:
        rx (float): Rotation around X axis in radians.
        ry (float): Rotation around Y axis in radians.
        rz (float): Rotation around Z axis in radians.
        order (str): Rotation order. Defaults to "XYZ".

    Returns:
        tuple: (w, x, y, z)
    """
    r = R.from_euler(order, [rx, ry, rz])
    q = r.as_quat()  # Returns in (x, y, z, w) order
    return (q[3], q[0], q[1], q[2])  # Convert to (w, x, y, z)

# # Example usage (Euler angles in radians)
# rx = np.deg2rad(45)
# ry = np.deg2rad(30)
# rz = np.deg2rad(60)

# quaternion = euler_to_quaternion(rx, ry, rz)
# print("Quaternion (w, x, y, z):", quaternion)

# Example usage:
if __name__ == "__main__":
    q = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
    omega = np.array([0.0, 1.0, 0.0])   # rad/s, around y-axis
    dt = 0.02
    q_next = integrate_quat(q, omega, dt)
    print(q_next)