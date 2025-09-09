'''
@File: config.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 23, 2025
@HomePage: https://github.com/YanJieWen
'''

class Config(object):
    def __init__(self,diction):
        for key,value in diction.items():
            if isinstance(value,dict):
                setattr(self,key,Config(value))
            else:
                setattr(self,key,value)