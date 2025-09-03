# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:01:56 2025

@author: jlmorgan
"""

def compare_files(file1_path, file2_path):
    import difflib
    from pathlib import Path


    py_path1 = str(Path(r"D:\PYTHON\jmPackages\bifo\src\bifo\diced\inferFromDiced.py"))
    py_path2 = str(Path(r"D:\PYTHON\jmPackages\bifo\src\bifo\diced\old\process_fetched_diced.py"))

    # py_path1 = str(Path(f'r{file1_path}'))
    # py_path2 = str(Path(f'r{file2_path}'))
    # py_path1 = file1_path.replace("\\","/")
    # py_path2 = file2_path.replace("\\","/")
    # py_path1 = py_path1.replace('\',"/")
    # py_path2 = py_path2.replace("\\","/")
        
    with open(py_path1, 'r') as f1, open(py_path2, 'r') as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
    
    d = difflib.Differ()
    diff = list(d.compare(file1_lines, file2_lines))
    
    for line in diff:
        print(line, end='')