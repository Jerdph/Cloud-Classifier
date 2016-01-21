import os
from algorithm import plot_cloud, extract_data, wing_detect, hood_detect, z_max_y, rotateXMatrix, rotateYMatrix, rotateZMatrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math
import operator
import pandas as pd
import os, random



def create_plot(filename, file, sampling = 'easy'):
    data= pd.read_csv(filename)
    print filename
    rel_col = ['Start X','Start Y','Start Z','End X','End Y','End Z']
    data = data[rel_col]
    data_1 = data[['Start X','Start Y','Start Z']]
    data_2 = data[['End X','End Y','End Z']]
    data_2.columns = ['Start X','Start Y','Start Z']
    dat_new = pd.concat((data_1,data_2),axis=0)
    dat_new = dat_new.dropna(axis=0)
    dat_new = (dat_new-dat_new.mean())/np.linalg.norm(dat_new.values)

    #set sampling rate
    sam_dict = {'easy':0.7,'medium':0.1,'hard':0.01}
    sam_rate = sam_dict[sampling]

    #sample using sampling rate
    dat_new.reset_index(drop=True,inplace=True)
    ind = np.random.choice(dat_new.index, size=dat_new.shape[0]*sam_rate, replace=False)
    dat_new = dat_new.iloc[ind,:]

    # find highest, lowest standard deviation vector and orient the model accordingly
    highest = np.where(dat_new.std()==dat_new.std().max())[0][0]
    lowest = np.where(dat_new.std()==dat_new.std().min())[0][0]
    if highest == 1:
        dat_new = np.dot(dat_new,rotateZMatrix(math.pi/2))
    elif highest == 2:
        dat_new = np.dot(dat_new.values,rotateYMatrix(math.pi/2))
    data_plot = pd.DataFrame(dat_new, columns = ['Start X','Start Y','Start Z'])

    #Plot
    fig = plt.figure()

    data_plot = pd.DataFrame(data_plot, columns = ['Start X','Start Y','Start Z'])
    ax = plt.axes(projection='3d')
    x = data_plot['Start X']
    y = data_plot['Start Y']
    z = data_plot['Start Z']
    #name the plot files
    name_front = 'temp/' + file[:-4] + '_front'
    name_side = 'temp/' + file[:-4] + '_side'
    name_top = 'temp/' + file[:-4] + '_top'
    ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=0)
    plt.savefig('../static/'+ name_front + sampling)
    #side view
    ax.view_init(azim=90)
    plt.savefig('../static/'+ name_side + sampling)
    #top view
    ax.view_init(elev=90,azim=0)
    plt.savefig('../static/'+ name_top + sampling)

difficulties = ['easy','medium','hard']
for file in os.listdir("../data/test set")[1:]:
    filename = "../data/test set/" + file
    for diff in difficulties:
        print diff
        create_plot(filename = filename, file=file, sampling =diff)
