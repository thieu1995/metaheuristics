import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from utils.FunctionUtil import cal_mean, cal_std
from decimal import Decimal
"""
Generate unimodal, multimodal, hybrid, compostion result table 
Input: read input from history/overall/algo_dict_info.pkl which is generated after running 
get_experiment_infor.py
Output: tex().txt in history/generated_labtex_table/
"""


def load_data():
    with open('./history/overall/algo_dict_info.pkl', 'rb') as f:
        alf = pkl.load(f)
    flist = []
    for fi in range(30):
        std_rank = []
        mean_rank = []
        worst_rank = []
        best_rank = []
        names = set()

        for name, algo in alf.items():
            names.add(name)
            # std_rank = [['abc', 1.23], ['woa', 4.56]]
            std_rank.append([name, algo.std[fi]])
            mean_rank.append([name, algo.mean[fi]])
            worst_rank.append([name, algo.worst[fi]])
            best_rank.append([name, algo.best[fi]])

        std_rank = sorted(std_rank, key=lambda x: x[1])
        mean_rank = sorted(mean_rank, key=lambda x: x[1])
        worst_rank = sorted(worst_rank, key=lambda x: x[1])
        best_rank = sorted(best_rank, key=lambda x: x[1])

        std_infor = {}
        mean_infor = {}
        worst_infor = {}
        best_infor = {}

        for si in range(len(std_rank)):
            # std_infor = {'abc': {'value': 3, 'rank': 3}, 'bca': {..}}
            std_infor[std_rank[si][0]] = {'value': round(std_rank[si][1], 4),
                                          'rank': si + 1}
            mean_infor[mean_rank[si][0]] = {'value': round(mean_rank[si][1], 4),
                                            'rank': si + 1}
            worst_infor[worst_rank[si][0]] = {'value': round(worst_rank[si][1], 4),
                                              'rank': si + 1}
            best_infor[best_rank[si][0]] = {'value': round(best_rank[si][1], 4),
                                            'rank': si + 1}

        # flist = [{std: std_infor, 'mean': mean_infor, ...},{std: std_infor,}]
        flist.append({'std': std_infor,
                      'mean': mean_infor,
                      'worst': worst_infor,
                      'best': best_infor
                      })
    return flist, names


def gen_latex_table(flist, names, list_fun, font_size='footnotesize'):
    with open("tex " + str(list_fun) + '.txt', 'w', encoding='utf-8') as f:
        for fun_break in range(len(list_fun) - 1):
            f.write("{ \n")
            f.write('\\begin{table}[h!] \n')
            f.write('\\begin{' + font_size + '}\n')
            f.write('\\begin{center} \n')
            format_table = ''
            for i in range(len(names) + 2):
                format_table += 'c'
            # \begin{tablular}{|c|c|c|}
            f.write('   \\begin{tabular}{' + format_table + '}  \n')

            # ------------------------------
            f.write('       \hline \n')
            # Function & Criteria & ABC & PSO & WO
            f.write('       Function & Criteria ')
            for name in names:
                if name == 'ABFOLS':
                    name = 'ABFO'
                f.write(' & ' + name + ' ')
            f.write('\\\ \n')
            f.write('       \hline \n')

            # ---------------------------
            # F1  mean   0.3   0.4  0.5
            #     std    0.6   0.7  0.8
            #     worst  0.5   0.8  0.9
            #     fit    0.3   0.5  0.5
            #     rank    1     2    3
            for i in range(list_fun[fun_break], list_fun[fun_break + 1]):
                # F1 &
                f.write('       \multirow{5}{1em}{F' + str(i + 1) + '}')

                # line 0 = F1 mean 0.3  0.3  0.2
                # line 1 =    std  0.2  1    3
                line = 0

                # flist[i] = {'std':{'abc': {'value': 3, 'rank': 3}, 'bca': {..}}}
                # k = 'std', v = {'abc' : {'value': 3, 'rank': 3}}
                for k, v in flist[i].items():
                    # F & std & v['abc']['value'] & v['woa']['value'] \\
                    if line == 0:
                        f.write('     & ' + str(k) + ' ')
                    else:
                        f.write('                               & ' + str(k) + ' ')
                    line += 1
                    for al_name in names:
                        value = v[al_name]['value']
                        value = '%.2E' % Decimal(value)
                        if v[al_name]['rank'] == 1:
                            str_value = '\\textbf{' + str(value) + '}'
                        else:
                            str_value = str(value)
                        f.write('           & ' + str_value + ' ')
                    f.write(' \\' + "\\ ")  # new line in latex
                    f.write(' \n ')  # write new line in file
                f.write('                               & rank')
                for name in names:
                    rank = flist[i]['mean'][name]['rank']
                    if rank == 1:
                        str_rank = '\\textbf{' + str(rank) + '}'
                    else:
                        str_rank = str(rank)
                    f.write('           & ' + str_rank + '            ')
                f.write(' \\' + "\\ ")  # new line in latex
                f.write(' \n')  # write new line in file
                f.write('       \hline \n')
            f.write('   \end{tabular} \n')
            # f.write(' \\' + "\\ \n")  # new line in latex
            f.write('\end{center} \n')
            #  f.write(' \\' + "\\ \n")  # new line in latex
            f.write('\\end{' + font_size + '} \n')
            # f.write(' \\' + "\\ \n")  # new line in latex
            f.write('\\caption{Result and comparison of different algorithms based on several metrics for multimodal functions} \n')
            f.write('\\label{tb:} \n')
            f.write('\\end{table} \n')
            f.write("} \n")


if __name__ == "__main__":
    flist, names = load_data()
    list_fun = (0, 3, 16, 22, 30)
    names = ['GA', 'PSO', 'ABC', 'CRO', 'WOA', 'QSO', 'IQSO']
    gen_latex_table(flist, names, list_fun)
