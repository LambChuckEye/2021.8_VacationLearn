import torch

x = torch.arange(4.0)
print(x)

# 设置梯度存储
x.requires_grad_(True)
# 通过 x.grad调用
print(x.grad)

# 给定函数y=2 * <x,x>
y = 2 * torch.dot(x, x)

# 求导
y.backward()
print(x.grad)
# 验证结果
print(x.grad == 4 * x)

# =======================================
# 默认情况，pytorch会累计梯度，在计算新函数前要清除之前的值
x.grad.zero_()
y = x.sum()
y.sum().backward()
print(x.grad)

# =========================================
# 将计算移动到计算图之外
x.grad.zero_()
y = x * x
# 将y作为常数，而不是x的函数
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad)
# 对于y仍可以求导
x.grad.zero_()
y.sum().backward()
print(x.grad)
print(len(torch.tensor([2, -3.4])))
print(torch.tensor([2, -3.4]).reshape((-1, 1)).shape)
