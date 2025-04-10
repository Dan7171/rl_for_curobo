import torch

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

# Example usage:
if __name__ == "__main__":
    # Create example tensor
    example = torch.tensor([
        [1, 0, 4],
        [6, 1, 2],
        [0, 3, 1]
    ])
    
    # Shift the tensor
    result = shift_tensor_left(example)
    
    print("Original tensor:")
    print(example)
    print("\nShifted tensor:")
    print(result) 