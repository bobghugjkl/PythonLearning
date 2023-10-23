class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def info(self):
        print(self.name, self.age)


class Student(Person):
    def __init__(self, name, age, score):
        super().__init__(name, age)
        self.score = score


stu = Student("jack", 20, "1001")
stu.info()

a = 20
b = 100
c = a + b
d = a.__add__(b)


class Learner:
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        return self.name + other.name


stu1 = Learner("张三")
stu2 = Learner("李四")

s = stu1 + stu2
print(a)
