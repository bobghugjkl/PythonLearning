class Student:  # 要求首字母大写，其余的小写
    # pass   用来占位，保证程序不报错
    native_place = "吉林"  # 类属性(直接写在类里面的属性)

    def __init__(self, name, age):  # name,age为实例属性，是初始化方法，里面是赋值操作
        self.name = name
        self.__age = age

    # 实例方法,类之外定义的称为函数，类之内定义的称为方法
    def info(self):
        print("我的名字叫", self.name, "年龄是", self.__age)

    # 类方法
    @classmethod
    def cm(cls):
        print("类方法")

    # 静态方法
    @staticmethod
    def sm():
        print("静态方法")


# 判断Student是不是对象
print(id(Student))
print(type(Student))
print(Student)
# 创建一个对象，存在类指针指向Student类对象
stu1 = Student("张三", 20)
print(id(Student))
print(type(Student))
print(Student)  # 会发现与上面的不一样，说明新开了一个空间
Student.info(stu1)  # 此时调用类内部的实例方法，self指的是一个Student的对象，所以要传入一个对象
stu1.info()  # 当然也可以这样
print(Student.native_place)  # 访问类属性
Student.cm()  # 调用类方法
Student.sm()  # 调用静态方法


# 下面演示动态绑定属性和方法
def show():
    # stu = Student("Jack", 20)
    # stu.gender = "男"  # 动态绑定姓名,只属于stu
    # print(stu.name, stu.age, stu.gender)
    print("定义在类之外的方法")
stu1.show = show  # 动态绑定方法
stu1.show()
# 如果属性不希望在类外面被访问，前面使用两个'_',可以用dir()访问
# print(dir(stu1)) # 查看有多少属性和方法
print(stu1._Student__age)  # 这样就可以访问



