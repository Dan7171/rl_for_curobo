import numpy as np
class FrameUtils:
    @staticmethod
    def quat_inv(q):
        # q = [w, x, y, z], unit quaternion
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    @staticmethod
    def quat_mul(q1, q2):
        # Hamilton product of two quaternions
        w1,x1,y1,z1 = q1
        w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def rotate_vec(q, v):
        # rotate 3-vector v by unit quaternion q
        qv = np.concatenate([[0.], v])
        return FrameUtils.quat_mul(FrameUtils.quat_mul(q, qv), FrameUtils.quat_inv(q))[1:]

    @staticmethod
    def world_to_F(p_WF, q_WF, p_WF2, q_WF2):
        """Takes 2 poses (frames) F and F2 in the world frame W, and returns the pose of F2 expressed in the frame F.
        In:
            Frame F's pose expressed in the world frame W:
                p_WF: position (x,y,z) of frame F expressed in the world frame W
                q_WF:quaternion (qw,qx,qy,qz) of frame F expressed in the world frame W
            Frame F2's pose in the world frame W:
                p_WF2: position (x,y,z) of frame F2 expressed in the world frame W
                q_WF2: quaternion (qw,qx,qy,qz) of frame F2 expressed in the world frame W
        Out:
            p_FF2: position (x,y,z) of frame F2 expressed in the frame F
            q_FF2: quaternion (qw,qx,qy,qz) of frame F2 expressed in the frame F
        """
        # returns (p_FF2, q_FF2)
        delta = p_WF2 - p_WF
        p_rel = FrameUtils.rotate_vec(FrameUtils.quat_inv(q_WF), delta)
        q_rel = FrameUtils.quat_mul(FrameUtils.quat_inv(q_WF), q_WF2)
        return p_rel, q_rel

    @staticmethod
    def F_to_world(p_WF, q_WF, p_FF2, q_FF2):
        """Takes a frame F's pose expressed in the world frame W and a frame F2's pose expressed in the frame F, and returns the pose of F2 expressed in the world frame W.
        In:
            Frame F's pose expressed in the world frame W:
                p_WF: position (x,y,z) of frame F expressed in the world frame W
                q_WF:quaternion (qw,qx,qy,qz) of frame F expressed in the world frame W
            Frame F2's pose in the frame F:
                p_FF2: position (x,y,z) of frame F2 expressed in the frame F
                q_FF2: quaternion (qw,qx,qy,qz) of frame F2 expressed in the frame F
        Out:
            p_WF2: position (x,y,z) of frame F2 expressed in the world frame W
            q_WF2: quaternion (qw,qx,qy,qz) of frame F2 expressed in the world frame W
        """
                    
        p_WF2 = FrameUtils.rotate_vec(q_WF, p_FF2) + p_WF
        q_WF2 = FrameUtils.quat_mul(q_WF, q_FF2)
        return p_WF2, q_WF2
