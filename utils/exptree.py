'''
@File: exptree.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 23, 2025
@HomePage: https://github.com/YanJieWen
'''


import os
''''
导出层次化项目文件夹
'''

def export_tree(root_dir, output_file, ignore=None):
    if ignore is None:
        ignore = {'.git', '__pycache__'}

    def tree(dir_path, prefix=""):
        entries = [e for e in os.listdir(dir_path) if e not in ignore]
        entries.sort()
        entries_count = len(entries)
        for i, entry in enumerate(entries):
            path = os.path.join(dir_path, entry)
            connector = "└── " if i == entries_count - 1 else "├── "
            yield prefix + connector + entry
            if os.path.isdir(path):
                extension = "    " if i == entries_count - 1 else "│   "
                yield from tree(path, prefix + extension)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(os.path.basename(root_dir) + "/\n")
        for line in tree(root_dir):
            f.write(line + "\n")
