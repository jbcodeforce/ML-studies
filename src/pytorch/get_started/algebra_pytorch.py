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
print(f"Device is {device}")
# declare pytorch tensor for a matrix
matrix1 = torch.tensor([[1, 1, 1], [1, 1, 2]], device=device, dtype=torch.float16)
print(f"\n First Matrix {matrix1}")
matrix2 = torch.tensor([[1, 2], [1,1],[1, 1]], device=device, dtype=torch.float16)
product = torch.mm(matrix1, matrix2)
print(f"\n Product of matrix1 with {matrix2}")
print(f"is the matrix: {product}")
    
x = torch.tensor([1.0, 2.0], device=device)
a = torch.tensor([3.0, 3.0], device=device)

sub = torch.subtract(x, a)
print(f" {x} - {a} is {sub}")
    
