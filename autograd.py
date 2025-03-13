#%% Basic autograd example
import torch
x = torch.tensor([2.0, 3.0])  
x.requires_grad = True # Enable gradient tracking
y = 2 * torch.dot(x, x)

y.backward()
x.grad == 4*x  

#%% Non smooth functions L1 norm
x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
y = torch.sum(torch.abs(x))
y.backward()
print(x.grad)  

#%% Function derivatives
def f(x):
    y = x * 2
    while y.norm() < 1000:
        y = y * 2
    return y

x = torch.tensor([0.5], requires_grad=True)
y = f(x)
y.backward()
print(x.grad)  # Gradient depends on control flow path taken


#%% Partial derivatives
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
z = x + y
z.backward()
print(x.grad)  # Should be tensor([1., 1., 1.])

#%%
# By default, PyTorch accumulates gradients, we need to clear previous values
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.sum(x**2)

y.backward()
x.grad == 2*x

#%% 
# Now compute a new function gradient 
y = torch.sum(x**3)
y.backward()
x.grad == 3*x**2 

x.grad.zero_()
y = torch.sum(x**3)
y.backward()
x.grad == 3*x**2


#%%
# Create a parameter with gradients enabled
x = torch.tensor([2.0], requires_grad=True)
y = x**2
y.backward()
# BAD: Update without torch.no_grad()
x = x - 0.1 * x.grad  # This creates a new tensor with requires_grad=True
# Now x is part of a computation graph and disconnected from its previous grad
print(x.grad)  # None, because this is a new tensor

# Good
x = torch.tensor([2.0], requires_grad=True)
y = x**2
y.backward()
with torch.no_grad():
    x = x - 0.1 * x.grad

# %% Complete Gradient Descent
x = torch.tensor([2.0], requires_grad=True)

def f(x):
    return x**2

lr = 0.1
for t in range(100):
    y = f(x)
    y.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()

