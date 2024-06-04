import shutil
import os
import argparse

def del_files0(dir_path):
    shutil.rmtree(dir_path)


parser = argparse.ArgumentParser()
parser.add_argument('--deletetxt',default='./check/bad_demo.txt', type=str)
args = parser.parse_args()

del_all = []
with open(args.deletetxt,'r') as f:
    lines=f.readlines()
    for line in lines:
        line=line.strip('\n')
        del_all.append(line)
for i in range(len(del_all)):
    print(i)
    del_files0(del_all[i])
