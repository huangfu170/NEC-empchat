import torch
from transformers import AutoTokenizer



a=torch.tensor([0, 1, 2, 3])
b=torch.tensor([[1], [2], [3], [4]])
print(a<b)