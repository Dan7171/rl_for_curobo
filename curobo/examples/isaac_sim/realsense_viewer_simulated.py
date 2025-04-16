# ... existing code ...
import cv2
import numpy as np
# Comment out the RealSense import
# from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader

class SimulatedRealsense:
    def __init__(self):
        self.frame_count = 0
    
    def get_raw_data(self):
        # Create simulated depth image (grayscale)
        depth = np.zeros((480, 640), dtype=np.uint16)
        # Add some simple animated pattern
        x, y = np.meshgrid(np.linspace(0, 639, 640), np.linspace(0, 479, 480))
        depth = ((np.sin(x/100 + self.frame_count/10) + np.cos(y/100)) * 1000).astype(np.uint16)
        
        # Create simulated color image
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        color[:, :, 0] = ((np.sin(x/100) + 1) * 127).astype(np.uint8)
        color[:, :, 1] = ((np.cos(y/100) + 1) * 127).astype(np.uint8)
        color[:, :, 2] = 128
        
        self.frame_count += 1
        return depth, color
    
    def stop_device(self):
        pass

def view_realsense():
    # Replace RealSense with simulation
    realsense_data = SimulatedRealsense()
    # ... rest of existing code ...
    # Streaming loop
    try:
        while True:
            data = realsense_data.get_raw_data()
            depth_image = data[0]
            color_image = data[1]
            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_JET
            )
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
            cv2.imshow("Align Example", images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        realsense_data.stop_device()


if __name__ == "__main__":
    view_realsense()
