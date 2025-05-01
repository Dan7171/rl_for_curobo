import torch
t = torch.tensor(False)
t.copy_(torch.tensor(True))
print(t)