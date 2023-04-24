# 本模块是全局变量管理模块，导入本模块就可以实现跨模块使用某个全局变量（用 global 关键字来定义） 'num_bands': 100
# !/usr/bin/env python3
# -*- coding:utf-8-*-
# 全局变量管理模块


def _init():
    """在主模块初始化"""
    global _global_dict  # 此处为什么要申明为global变量？ 如果不加global，则_global_dict只是一个_init()函数内部的局部变量
    _global_dict = {'image_file': 'D:\\Matlab练习\\duogun\\Pavia.mat', 'label_file': 'D:\\Matlab练习\\duogun\\Pavia_gt.mat', 'num_bands': 102}
    # 同理，set_value(),get_value()函数内部的_global_dict也都是局部变量，
    # 也就是说，尽管同名，但它们是3个不同的变量，相互之间是完全独立的。因此，申明为global变量的第一个作用是：
    # 在本篇代码文件内，使不同函数能够操作同一个变量，只要变量名是相同的，那么不管出现在本篇的任何地方，它都是同一个变量。


def set_value(key, value):
    """ 定义一个全局变量 """
    try:
        _global_dict[key] = value
        return True
    except KeyError:
        return False


def get_value(key, default_value=None):
    """ 获得一个全局变量,不存在则返回默认值 """
    try:
        return _global_dict[key]
    except KeyError:
        return default_value
