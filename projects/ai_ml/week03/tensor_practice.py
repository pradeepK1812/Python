import torch
torch.__version__

scalar = torch.tensor(7)
print(scalar)

vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)

print("MATRIX data")
# Matrix
MATRIX = torch.tensor([[7, 8],
                       [9, 10],
                       [11,12]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)

# Create a random tensor of size (3, 4)
print("Random tensor")
random_tensor = torch.rand(size=(3, 4))
print(random_tensor)
print(random_tensor.ndim)
print(random_tensor.shape)
print( random_tensor.dtype)

# Create a tensor of all zeros
print("Tensor of all zeros")
zeros = torch.zeros(size=(3, 4))
print(zeros)
print( zeros.dtype)



# Create a range of values 0 to 10
print("Range tensor")
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

# Default datatype for tensors is float32
print("tensor with float data type")
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

print("Float 16 tensor")
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

print(float_16_tensor.dtype)

#creating random tensor and extracting info from it
# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU
print(f"Device tensor dimension: {some_tensor.ndim}") # will default to CPU


#operations on tensors

# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
print(tensor * 10)
# Can also use torch functions
print(torch.multiply(tensor, 10))
# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)

#Matrix multiplication 
import torch
tensor = torch.tensor([1, 2, 3])
print(torch.matmul(tensor, tensor))
