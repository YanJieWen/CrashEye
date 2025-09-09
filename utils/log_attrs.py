'''
@File: log_attrs.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 23, 2025
@HomePage: https://github.com/YanJieWen
'''



from loguru import logger

def logger_attrs(obj, indent=0):
    for attr, value in obj.__dict__.items():
        if hasattr(value, "__dict__"):  # 说明是嵌套对象
            logger.info("  " * indent + f"{attr}:")
            logger_attrs(value, indent + 1)
        else:
            logger.info("  " * indent + f"{attr}: {value}")