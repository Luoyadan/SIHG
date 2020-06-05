import numpy as np

file_list = ['soc-sign-Slashdot090221.txt', 'epinions.txt']
for file in file_list:
    x = np.loadtxt(file)
    print('| For {} Dataset |'.format(file[9:-4]))
    print('number of nodes: ' + str(x.max()))
    print('number of edges: ' + str(x.shape[0]))
    neg_edges = np.array([1 if ele == -1 else 0 for ele in x[:, 2]]).sum()
    neg_ratio = neg_edges / x.shape[0]
    pos_ratio = 1 - neg_ratio
    print('ratio of negative edges {}'.format(neg_ratio))
    print('ratio of positive edges {}'.format(pos_ratio))