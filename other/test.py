import pandas as pd
import numpy as np

t_seq = pd.read_csv('datasets/train_data_multiprocessing.csv')
t_func = pd.read_csv('datasets/train_data_multi.csv')

t_seq = t_seq.values
t_func = t_func.values

diff = list()

for index, row in enumerate(t_func):
    flag = False
    multi_flag = False
    for inner_index, inner_row in enumerate(t_seq):
        res = np.array_equal(row, inner_row)
        if res:
            if flag:
                multi_flag = True
                print('index = {0}, inner_index = {1}'.format(index, inner_row))
                diff.append(1)
            flag = True
    if not flag:
        print('row = {0}, index {1} not printed'.format(row, index))
        diff.append(1)

print(len(diff)) #26
