import torch


def mask_decreasing_values(tensor):
    """
    For each row, if a value is less than the previous value, set it to 0.
    Uses vectorized operations for better performance.
    """
    # Create shifted version of the tensor (each element is compared with its previous)
    shifted = torch.roll(tensor, shifts=1, dims=1)
    
    # Handle both integer and float tensors
    if tensor.dtype in [torch.int32, torch.int64]:
        # For integer tensors, use the minimum possible value
        shifted[:, 0] = torch.iinfo(tensor.dtype).min
    else:
        # For float tensors, use -inf
        shifted[:, 0] = float('-inf')
    
    # Create mask where current element is less than previous
    mask = tensor < shifted
    
    # Apply mask (set decreasing values to 0)
    return torch.where(mask, torch.zeros_like(tensor), tensor)

def shift_tensor_left(tensor):
    """
    Shifts each row of the tensor left by one column.
    The last column becomes 0 since there is no element to its right.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (n, h)
        
    Returns:
        torch.Tensor: Shifted tensor of same shape as input
    """
    # Create a new tensor with zeros
    shifted = torch.zeros_like(tensor)
    
    # Copy all columns except the last one, shifted left by one
    shifted[:, :-1] = tensor[:, 1:]
    
    return shifted

