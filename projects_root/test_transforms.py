import torch
import numpy as np
from utils.transforms import transform_pose_between_frames, transform_poses_batched


def test_single_vs_batched_consistency():
    """Test that batched function produces same results as single pose function"""
    
    # Test poses
    test_poses = [
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],  # Identity rotation
        [0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0],  # 90 deg rotation around X
        [5.0, -2.0, 1.5, 0.5, 0.5, 0.5, 0.5],  # Complex pose
        [-1.0, 0.5, -0.8, 0.0, 0.0, 0.707, 0.707],  # 90 deg rotation around Z
    ]
    
    # Test frames
    frame_expressed_in = [1.0, 1.0, 1.0, 0.707, 0.0, 0.707, 0.0]  # 90 deg Y rotation + translation
    frame_express_at = [2.0, 0.0, -1.0, 0.866, 0.0, 0.0, 0.5]  # 60 deg Z rotation + translation
    
    # Test single poses
    single_results = []
    for pose in test_poses:
        result = transform_pose_between_frames(pose, frame_expressed_in, frame_express_at)
        single_results.append(result)
    
    # Test batched poses
    poses_tensor = torch.tensor(test_poses, dtype=torch.float32)
    batched_result = transform_poses_batched(poses_tensor, frame_expressed_in, frame_express_at)
    
    # Compare results
    for i, single_result in enumerate(single_results):
        batched_single = batched_result[i].numpy()
        print(f"Pose {i}:")
        print(f"  Single:  {single_result}")
        print(f"  Batched: {batched_single}")
        print(f"  Diff:    {np.abs(np.array(single_result) - batched_single)}")
        
        # Check if results are close (within numerical precision)
        assert np.allclose(single_result, batched_single, atol=1e-6), f"Results differ for pose {i}"
    
    print("✓ Single vs batched consistency test passed!")


def test_identity_transform():
    """Test that identity transforms work correctly"""
    
    # Test poses
    poses = torch.tensor([
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0],
    ], dtype=torch.float32)
    
    # Identity transform (world to world)
    result = transform_poses_batched(poses, None, None)
    
    print("Identity transform test:")
    print(f"Input:  {poses}")
    print(f"Output: {result}")
    print(f"Diff:   {torch.abs(poses - result)}")
    
    assert torch.allclose(poses, result, atol=1e-6), "Identity transform failed"
    print("✓ Identity transform test passed!")


def test_different_shapes():
    """Test that different tensor shapes work correctly"""
    
    frame_expressed_in = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    frame_express_at = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    
    # Test 1D (single pose)
    pose_1d = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    result_1d = transform_poses_batched(pose_1d.unsqueeze(0), frame_expressed_in, frame_express_at)
    print(f"1D shape test: {pose_1d.shape} -> {result_1d.shape}")
    
    # Test 2D (batch of poses)
    poses_2d = torch.tensor([
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0],
    ], dtype=torch.float32)
    result_2d = transform_poses_batched(poses_2d, frame_expressed_in, frame_express_at)
    print(f"2D shape test: {poses_2d.shape} -> {result_2d.shape}")
    
    # Test 3D (batch of sequences)
    poses_3d = torch.tensor([
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0]],
        [[5.0, -2.0, 1.5, 0.5, 0.5, 0.5, 0.5],
         [-1.0, 0.5, -0.8, 0.0, 0.0, 0.707, 0.707]]
    ], dtype=torch.float32)
    result_3d = transform_poses_batched(poses_3d, frame_expressed_in, frame_express_at)
    print(f"3D shape test: {poses_3d.shape} -> {result_3d.shape}")
    
    # Verify shapes are preserved
    assert result_1d.shape == (1, 7), f"Expected (1, 7), got {result_1d.shape}"
    assert result_2d.shape == poses_2d.shape, f"Expected {poses_2d.shape}, got {result_2d.shape}"
    assert result_3d.shape == poses_3d.shape, f"Expected {poses_3d.shape}, got {result_3d.shape}"
    
    print("✓ Different shapes test passed!")


def test_gpu_compatibility():
    """Test that the function works on GPU if available"""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return
    
    device = torch.device('cuda')
    
    poses = torch.tensor([
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0],
    ], dtype=torch.float32, device=device)
    
    frame_expressed_in = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    frame_express_at = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    
    result = transform_poses_batched(poses, frame_expressed_in, frame_express_at)
    
    assert result.device == device, f"Expected result on {device}, got {result.device}"
    assert result.shape == poses.shape, f"Expected {poses.shape}, got {result.shape}"
    
    print("✓ GPU compatibility test passed!")


def test_error_handling():
    """Test that proper errors are raised for invalid inputs"""
    
    # Test wrong last dimension
    try:
        poses = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0]], dtype=torch.float32)  # Only 6 elements
        transform_poses_batched(poses)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("✓ Error handling test passed!")


if __name__ == "__main__":
    print("Running transform function tests...\n")
    
    test_single_vs_batched_consistency()
    print()
    
    test_identity_transform()
    print()
    
    test_different_shapes()
    print()
    
    test_gpu_compatibility()
    print()
    
    test_error_handling()
    print()
    
    print("All tests passed! 🎉") 