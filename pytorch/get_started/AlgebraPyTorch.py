# Make use of a GPU or MPS (Apple) if one is available.
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
   
device=get_device()
# declare pytorch tensor for a matrix
matrix1 = torch.tensor([[1, 1, 1], [1, 1, 2]], device=device, dtype=torch.float16)
print(matrix1)
matrix2 = torch.tensor([[1, 2], [1,1],[1, 1]], device=device, dtype=torch.float16)
product = torch.mm(matrix1, matrix2)
print(product)
    
x = torch.tensor([1.0, 2.0], device=device)
a = torch.tensor([3.0, 3.0], device=device)

sub = torch.subtract(x, a)
print(sub)
    
