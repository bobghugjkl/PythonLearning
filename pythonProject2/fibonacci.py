import traceback

# 上面的包用来输出bug（如果有的话）
r = range(10)
print(r)
print(list(r))
print(5 in r)
print(11 not in r)
s = [i for i in range(1, 10)]
print(s)
scores = {'张三': 100, '李四': 98, '王五': 45}
keys = scores.keys()
print(keys)
print(type(keys))
print(list(keys))
values = scores.values()
print(values)
print(type(values))
print(list(values))

items = ['Fruits', 'Books', 'Others']
prices = [96, 78, 85]

d = {item: price for item, price in zip(items, prices)}
print(d)

try:
    a = int(input('请输入第一个整数'))
    b = int(input('请输入第二个整数'))
    result = a / b
    print('结果为：', result)
except ZeroDivisionError:
    print("不准确")
except ValueError:
    print("只能输入数字！")
except BaseException as e:
    print("error", e)
else:
    print(result)
finally:
    print("无论是否产生异常都会执行")
print("程序结束")
try:
    print("=====================") # 由于线程问题一会在前面一会在后面（输出时）
    print(1 / 0)
except:
    traceback.print_exc()
# 左侧点一下是断点，然后可以左键点那个小虫子就可以进入调试页面，按F7的那个键是下一步