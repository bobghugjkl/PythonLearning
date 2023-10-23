#!/usr/bin/python3
import sys


class parent:
    def myMethod(self):
        print('调用父类方法')


class Child(parent):
    def myMethod(self):
        print("调用子类方法")


class people:
    # 基本类型
    name = ''
    age = 0
    # 私有类型
    __weight = 0

    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w

    def speak(self):
        print("%s 说：我%d 岁。" % (self.name, self.age))


# 单继承
class student(people):
    grade = ''

    def __init__(self, n, a, w, g):
        # 调用父类的构函
        people.__init__(self, n, a, w)
        self.grate = g

    # 覆写父类的方法
    def speak(self):
        print("%s 说：我 %d 岁了，我在读 %d 年级" % (self.name, self.age, self.grate))


class Test:
    def prt(self):
        print(self)
        print(self.__class__)


class MyClass:
    """一个简单的类实例"""
    i = 12345

    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

    def f(self):
        return 'hellow world'


# 实例化类
x = MyClass(3.0, -4.5)
# 访问类的属性和方法
print("MyClass类的属性i为：", x.f())
print("MyClass类的方法f输出为：", x.f())
print(x.r, x.i)

t = Test()
t.prt()

# 实例化类
p = people('runoob', 10, 30)
p.speak()

s = student('ken', 10, 60, 3)
s.speak()

c = Child()  # 子类实例
c.myMethod()  # 子类调用重写方法
super(Child, c).myMethod()  # 用子类对象调用父类已被覆盖的方法

list = [1, 2, 3, 4]
it = iter(list)

while True:
    try:
        print(next(it))
    except StopIteration:
        sys.exit()
