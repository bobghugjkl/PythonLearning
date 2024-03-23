'''
1.Pytorch的API可以简洁实现
2.data模块提供了数据处理工具，nn模块定义了大量神经网络和常见的损失函数
3.我们可以通过 _ 结尾的方法将参数替换。初始化参数
'''
import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000) # 生成数据集


def load_array(data_arrays,batch_size,is_train = True):
    # 构造pytorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
batch_size = 10
data_iter = load_array({features,labels},batch_size)
next(iter(data_iter)) #next:获取第一项,读取并打印样本,iter构建迭代器

# net：模型变量，是一个Sequential实例，Sequential类将多个层串联在一起-->第一层输出作为第二层输入，注意，这里只有一层，称为全连接层
net = nn.Sequential(nn.Linear(2,1));#输入特征形状为1，输出特征形状为2
#初始化模型参数-》权重与偏置-》预定义：权重参数从均值0，标准差0.01中随机采样;偏置参数：0
net[0].weight.data.normal_(0,0.01)#net[0]选择第一个图层，weight.data，bias.data-》访问参数
net[0].bias.data.fill_(0) #normal_/fill_重写参数值
#定义损失函数->MSELOSS
loss = nn.MSELoss()
#小批量随机梯度下降（优化算法）->optim model ->制定优化的参数->net.parameters(),只需要设置lr
trainer = torch.optim.SGD(net.parameters(),lr=0.3)
'''
迭代周期-》完整遍历一次数据集-》获取小批量输入和相应的标签
                                            -》调用net(x)生成预测并计算损失（前向传播）
                                            -》进行反向传播算梯度
                                            -》调用优化器更新模型参数
'''
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()#执行迭代
    l = loss(net(features),labels)#计算损失net(...)预测值,label真实值
    print(f'epoch {epoch+1}, loss {1:f}')#格式化输出
'''比较真实参数与训练得到的模型参数
                    访问参数：1.net访问所需的层
                            2.读取该层权重和偏置
'''
w = net[0].weight.data
'''
true_w - w.reshape(true_w.shape) 这行代码表示两个张量（或数组）之间的元素级减法。
这里，true_w 和 w 很可能是权重向量或参数向量，而 w.reshape(true_w.shape) 确保了 w 具有与 true_w 相同的形状，
以便它们可以进行元素级的减法。
'''
print('w的预测差值:',true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print("b预测误差:",true_b - b)