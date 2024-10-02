import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import sys

def colorlabel(y,unique_arr):
    if y == unique_arr[0] :
        return "b"
    elif y == unique_arr[1] :return "g" 
    
m = 2
for i in range(40):
    print("If you want to quit, enter 'y'.")
    x = input('>> ')
    if x == "y" : break
    angle = 1
    #y = np.load(f'npz_retagged/sample_slcio_{sys.argv[1]}_retagged.npz')
    #y = np.load(f'npz_retagged_doublePG/sample_slcio_{i+1}_retagged.npz')
    #y = np.load(f'/gluster/maxi/ilc/tsumura/npz_retagged_doublePG_{angle}_com/npz_retagged_{i+1}_1_{angle}.npz')
    ##y = np.load(f'/gluster/maxi/ilc/tsumura/npz_retagged_doublePG_{angle}_com/npz_retagged_{i+1}_1_{angle}.npz')
    ###y = np.load(f'/home/suehara/mlm/pytorch_ILC_PFA/mydata/npz_retagged_doublePG_1_com/npz_retagged_0_20_1.npz')
    y = np.load(f'/gluster/maxi/ilc/tsumura/npz_retagged_doublePG_1_com/npz_retagged_0_20_1.npz')
    #y=np.load(f'/gluster/maxi/ilc/tsumura/npz_retagged_doublePG_0207_1/npz_retagged_doublePG_{angle}_com_rename/npz_retagged_{i+1}_1_{angle}.npz')
    #print(f"y : {y['recHitFeatures']}")

    x_point=y['recHitFeatures'][:,1]
    y_point=y['recHitFeatures'][:,2]
    z_point=y['recHitFeatures'][:,3]

    """
    print(f"x_point : {y['recHitFeatures'][1]}")
    print(f"y_point : {y['recHitFeatures'][2]}")
    print(f"z_point : {y['recHitFeatures'][3]}")
    """
    #print(f"plot points number : {y['recHitFeatures'][:,3].shape}")
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    label=y['recHitTruthClusterIdx']
    unique_label=np.unique(label)
    l = 0
    for x1,y1,z1,label1 in zip(x_point,y_point,z_point,label):
        #if colorlabel(label1) == "b":
        #print(f"label : {colorlabel(label1)}")
        ax.scatter(x1, y1, z1,c=colorlabel(label1,unique_label))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    
