import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

'''
通过内置函数将Fashion-MNIST数据集下载并读取到内存中
'''
trans = transforms.ToTensor()  # 将图像从PIL类型换成32位浮点数类型，并除以255使得所有像素的数值均在0-1之间
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, download=True, transform=trans
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True
)
'''
Fashion-MNIST->10个类别图像
'''
len(mnist_test), len(mnist_train)
'''
输入的图像的高度和宽度均为28像素，数据集由灰度图像构成，通道数为1   h高度*w宽度
'''
mnist_train[0][0].shape

'''
数字标签和文本名称之间转换的函数
'''


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'Coat', 'Sandal', 'shirt', 'Sneaker', 'bag', 'Ankle boot']
    return [text_labels[int(i)] for i in labels]


'''
可视化
'''


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

'''
读取小批量
'''
batch_size = 256


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

'''
看时间
'''
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f}sec'

'''
整合所有组件
获取和读取数据集，返回训练集和验证集的数据迭代器，此外还接受一个resize，将图像大小调整为另一种形状
'''


def load_data_fashion_mnist(batch_size, resize=True):
    """下载数据集并将他们放在内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True
    )

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_train, batch_size, shuffle=False, num_workers=get_dataloader_workers()
                            ))


'''
resize测试load_data_fashion_mnist()调整图像大小功能
'''
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)

for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
