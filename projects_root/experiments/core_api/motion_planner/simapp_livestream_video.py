


try:
    # Third Party
    import isaacsim
except ImportError:
    pass

import argparse
import time
import asyncio
import numpy as np
from datetime import datetime

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "hide_ui": False,  # Show the GUI
    "renderer": "RaytracedLighting",
    "display_options": 3286,  # Set display options to show default grid
}


# Start the omniverse application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp(launch_config=CONFIG)


import omni

from isaacsim.core.utils.extensions import enable_extension
# Default Livestream settings
simulation_app.set_setting("/app/window/drawMouse", True)
# Enable Livestream extension
enable_extension("omni.kit.livestream.webrtc")



async def record_video_async(num_frames, output_dir):
    """Async recording function using Replicator orchestrator"""
    import omni.replicator.core as rep
    from omni.replicator.core import BasicWriter
    
    # Create a camera for recording
    camera = rep.create.camera()
    
    # Position camera to see the scene
    with camera:
        rep.modify.pose(position=[0, -5, 3], look_at=[0, 0, 0])
    
    # Create render product
    render_product = rep.create.render_product(camera, (1280, 720))
    
    # Create writer for video output - RGB only
    writer = BasicWriter(
        output_dir=output_dir,
        frame_padding=4,
        rgb=True
    )
    
    # Attach writer to render product
    writer.attach([render_product])
    
    current_frame = 0
    print(f"Starting async recording for {num_frames} frames...")
    
    # Recording loop
    while current_frame < num_frames:
        # Get timeline interface
        timeline = omni.timeline.get_timeline_interface()
        
        # Start timeline if not playing
        if not timeline.is_playing():
            timeline.play()
            timeline.commit()
        
        # Step the orchestrator
        await rep.orchestrator.step_async(
            rt_subframes=1,
            delta_time=None,
            pause_timeline=False
        )
        
        current_frame += 1
        if current_frame % 50 == 0:
            print(f"Recorded {current_frame}/{num_frames} frames")
    
    # Stop timeline
    timeline.stop()
    print("Recording finished!")

def record_video_sync(num_frames, output_dir):
    """Synchronous recording function - replicates GUI Synthetic Data Recorder"""
    import omni.replicator.core as rep
    from omni.replicator.core import BasicWriter
    
    # Create a camera for recording
    camera = rep.create.camera()
    
    # Position camera to see the scene
    with camera:
        rep.modify.pose(position=[0, -5, 3], look_at=[0, 0, 0])
    
    # Create render product
    render_product = rep.create.render_product(camera, (1280, 720))
    
    # Create writer for video output - RGB only
    writer = BasicWriter(
        output_dir=output_dir,
        frame_padding=4,
        rgb=True
    )
    
    # Attach writer to render product
    writer.attach([render_product])
    
    print(f"🚀 Starting RGB recording for {num_frames} frames...")
    print(f"📁 Output: {output_dir}")
    print(f"📸 Recording: RGB PNG frames only")
    
    # Recording loop - replicates GUI behavior
    for frame in range(num_frames):
        # Get timeline interface
        timeline = omni.timeline.get_timeline_interface()
        
        # Start timeline if not playing (like GUI's "Control Timeline" checkbox)
        if not timeline.is_playing():
            timeline.play()
            timeline.commit()
            
        # Step the orchestrator synchronously (like GUI's step function)
        rep.orchestrator.step(
            rt_subframes=1,  # Like GUI's "RTSubframes" parameter
            delta_time=None,
            pause_timeline=False
        )
        
        if frame % 50 == 0:
            print(f"📸 Recorded {frame}/{num_frames} frames")
    
    # Wait for all data to be written (like GUI's wait functionality)
    rep.orchestrator.wait_until_complete()
    
    # Stop timeline
    timeline.stop()
    print("✅ RGB recording finished!")
    print(f"📁 Check output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true", help="Record video")
    parser.add_argument("--video_length", type=int, default=300, help="Frames to record")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for video frames (auto-generated if not specified)")
    parser.add_argument("--async_mode", action="store_true", help="Use async recording mode")
    args = parser.parse_args()
    
    # Generate automatic timestamp-based output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/cs_storage/evrond/{timestamp}"
    
    print(f"📁 Output directory: {args.output_dir}")
    
    # Ensure output directory exists
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✅ Created output directory: {args.output_dir}")

    # Enable required extensions
    enable_extension("omni.replicator.core")
    enable_extension("omni.replicator.isaac")
    
    # Wait for extensions to fully initialize
    time.sleep(3)
    
        # Initialize basic world for demonstration
    from omni.isaac.core import World
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    
    my_world.reset()

    if args.video:
        # Import replicator after extensions are enabled
        import omni.replicator.core as rep
        
        # Initialize replicator
        rep.orchestrator.run()
        
        if args.async_mode:
            # Use async recording
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.   (record_video_async(args.video_length, args.output_dir))
            loop.close()
        else:
            # Use sync recording
            record_video_sync(args.video_length, args.output_dir)
    else:
        # Just run simulation without recording
        print("Running simulation without recording...")
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        
        # Run simulation loop with livestream
        frame_count = 0
        try:
            while frame_count < args.video_length:
                my_world.step(render=True)
                
                if not my_world.is_playing():
                    if frame_count % 100 == 0:
                        print("**** Click Play to start simulation *****")
                    frame_count += 1
                    continue
                
                
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Simulation frame: {frame_count}/{args.video_length}")
                    
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        
        timeline.stop()

    print("Simulation completed")
    simulation_app.close()

if __name__ == "__main__":
    main()

    