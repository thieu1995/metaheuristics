import numpy as np
""""
Generate latex table of 30 benmark functions
"""


list_fun = ['Rotated High Conditioned Elliptic Function',
            'Rotated Bent Cigar Function',
            'Rotated Discus Function',
            'Shifted and Rotated Rosenbrock’s Function',
            'Shifted and Rotated Ackley’s Function',
            'Shifted and Rotated Weierstrass Function',
            'Shifted and Rotated Griewank’s Function',
            'Shifted Rastrigin’s Function',
            'Shifted and Rotated Rastrigin’s Function',
            'Shifted Schwefel’s Function',
            'Shifted and Rotated Schwefel’s Function',
            'Shifted and Rotated Katsuura Function',
            'Shifted and Rotated HappyCat Function',
            'Shifted and Rotated HGBat Function',
            'Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function',
            'Shifted and Rotated Expanded Scaffer’s F6 Function',
            'Hybrid Function 1 (N=3)',
            'Hybrid Function 2 (N=3)',
            'Hybrid Function 3 (N=4)',
            'Hybrid Function 4 (N=4)',
            'Hybrid Function 5 (N=5)',
            'Hybrid Function 6 (N=5)',
            'Composition Function 1 (N=5)',
            'Composition Function 2 (N=3)',
            'Composition Function 3 (N=3)',
            'Composition Function 4 (N=5)',
            'Composition Function 5 (N=5)',
            'Composition Function 6 (N=5)',
            'Composition Function 7 (N=3)',
            'Composition Function 8 (N=3)'
            ]

with open('/history/generated_latex_table/fun.txt', 'w') as f:
        for i in range(len(list_fun)):
            if i == 0:
                f.write('\multirow{3}{2em}{Unimodal} & ' + str(i+1) + ' & ' + str(list_fun[i]) + ' & '+ str((i+1)*100))
                f.write('\\\\ \n')
            elif i == 3:
                f.write('\hline \n')
                f.write('\multirow{13}{2em}{Unimodal} & ' + str(i+1) + ' & ' + str(list_fun[i]) + ' & '+ str((i+1)*100))
                f.write('\\\\ \n')
            elif i == 16:
                f.write('\hline \n')
                f.write('\multirow{6}{2em}{Unimodal} & ' + str(i+1) + ' & ' + str(list_fun[i]) + ' & '+ str((i+1)*100))
                f.write('\\\\ \n')
            elif i == 23:
                f.write('\hline \n')
                f.write('\multirow{8}{2em}{Unimodal} & ' + str(i+1) + ' & ' + str(list_fun[i]) + ' & '+ str((i+1)*100))
                f.write('\\\\ \n')
            else:
                f.write('& ' + str(i+1) + ' & ' + str(list_fun[i]) + ' & ' + str((i+1)*100))
                f.write('\\\\ \n')
        f.write('\hline \n')
