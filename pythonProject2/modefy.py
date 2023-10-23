#!/usr/bin/python3
# 文件名：using_sys.py

import sys


def fib(n):
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b
    print()


def fib2(n):
    result = []
    a, b = 0, 1
    while b < n:
        result.append(b)
        a, b = b, a + b
    return result


print('命令行参数如下：')
for i in sys.argv:
    print(i)
print('\n\nPython 路径为：', sys.path, '\n')
