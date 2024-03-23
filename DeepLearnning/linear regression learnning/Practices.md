# 问题1

* 如果我们用nn.MSELoss(reduction=''sum')替换nn.MSELoss()——即将小批量总损失替换平均值，则我们应该让学习率除以batch_size 
* 因为需要除以批量数
# 问题2
提供了nn.HuberLoss
'''python
def huber_loss(y_hat,y_neta = 0.005)
error = torch.abs(y_hat-y.detach)
return torch.where(error<beta,0.5*error**2/beta,error-0.5*beta)
'''

# 问题3
一般来说先call.backward()再.grad就可以拿到梯度了