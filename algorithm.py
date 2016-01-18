import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math
import operator
import os

#Extract Shape
def extract_data(filename, sampling = 'easy'):
    '''
    INPUT: String, String
    OUTPUT: Numpy Array
    Extract feature from a raw file. Then generate a numpy array of
    the features.
    '''
    #Create data dictionary to append features from each file
    data_dict = {'width': [], 'height': [],'length':[],
                 'ratio_xy':[],'ratio_midz':[], 'ratio_mid_pts':[],
                 'ratio_xz':[],'ratio_yz':[],'y_var_ratio':[], 'z_var_ratio':[]}
    #load the file
    load_dat = pd.read_csv(filename)
    #There are 6 relevant columns and 3 of them repeat so concat them on top of each other
    dat_1 = load_dat[['Start X','Start Y','Start Z']]
    dat_2 = load_dat[['End X','End Y','End Z']]
    dat_2.columns = ['Start X','Start Y','Start Z']
    dat_new = pd.concat((dat_1,dat_2),axis=0)
    dat_new = dat_new.dropna(axis=0)
    columns = dat_new.columns
    #normalize the data using the norm
    dat_new = (dat_new -dat_new.mean())/np.linalg.norm(dat_new.values)
    #find highest, lowest standard deviation vector and orient the model accordingly
    highest = np.where(dat_new.std()==dat_new.std().max())[0][0]
    lowest = np.where(dat_new.std()==dat_new.std().min())[0][0]
    if highest == 1:
        dat_new = np.dot(dat_new,rotateZMatrix(math.pi/2))
    elif highest == 2:
        dat_new = np.dot(dat_new.values,rotateYMatrix(math.pi/2))
    dat_new = pd.DataFrame(dat_new, columns = ['Start X','Start Y','Start Z'])
    #sample using sampling rate determined by level difficulty
    sam_dict = {'easy':0.7,'medium':0.1,'hard':0.01}
    sam_rate = sam_dict[level]
    dat_new.reset_index(drop=True,inplace=True)
    ind = np.random.choice(dat_new.index, size=dat_new.shape[0]*sam_rate, replace=False)
    dat_new = dat_new.iloc[ind,:]
    #max and min xyz after scale
    max_x = dat_new['Start X'].max()
    max_y = dat_new['Start Y'].max()
    max_z = dat_new['Start Z'].max()
    min_x = dat_new['Start X'].min()
    min_y = dat_new['Start Y'].min()
    min_z = dat_new['Start Z'].min()
    #calculate width height length
    width = dat_new['Start Y'].std()
    length = dat_new['Start X'].std()
    height = dat_new['Start Z'].std()
    #ratio of length and width
    ratio_xy = float(dat_new['Start Y'].std())/(
        dat_new['Start X'].std())
    ratio_xz = float(dat_new['Start X'].std())/(
        dat_new['Start Z'].std())
    ratio_yz = float(dat_new['Start Y'].std())/(
        dat_new['Start Z'].std())
    #create ratio of points above and below midpoint in the z axis
    midz = dat_new['Start Z'].mean()
    z_pts_above = dat_new[dat_new['Start Z']>midz]['Start Z'].count()
    z_pts_below = dat_new[dat_new['Start Z']<midz]['Start Z'].count()
    ratio_mid_pts = float(z_pts_above)/z_pts_below
    #create ratio of distance above and below the midpoint in z axis
    zabove = max_z - midz
    zbelow = midz - min_z
    ratio_midz = float(zabove)/zbelow
    #detect wings through diff of median and mean in the y axis
    y_var_ratio = wing_detect(dat_new)
    #hood detection
    z_var_ratio = hood_detect(dat_new)
    #append to dictionary
    data_dict['width'].append(width)
    data_dict['length'].append(length)
    data_dict['height'].append(height)
    data_dict['ratio_xy'].append(ratio_xy)
    data_dict['ratio_xz'].append(ratio_xz)
    data_dict['ratio_yz'].append(ratio_yz)
    data_dict['ratio_midz'].append(ratio_midz)
    data_dict['ratio_mid_pts'].append(ratio_mid_pts)
    data_dict['y_var_ratio'].append(y_var_ratio)
    data_dict['z_var_ratio'].append(z_var_ratio)
    data_feat = pd.DataFrame(data=data_dict, index= range(1))
    return data_feat.values

def plot_cloud(filename, file, sampling = 'easy'):
    '''
    INPUT: String, String, String
    OUTPUT: String, String, String
    Plots three 3D plots for each raw data file and return the name of the
    plotted figure.
    '''
    data= pd.read_csv(filename)
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
    sam_rate = sam_dict[level]

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
    #front view
    if os.path.isfile(name_front + sampling + '.png'):
        pass
    else:
        ax.scatter(x, y, z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(azim=0)
        plt.savefig('static/'+ name_front + sampling)
        #side view
        ax.view_init(azim=90)
        plt.savefig('static/'+ name_side + sampling)
        #top view
        ax.view_init(elev=90,azim=0)
        plt.savefig('static/'+ name_top + sampling)

    return name_front + sampling + '.png', name_side + sampling + '.png', name_top + sampling + '.png'

#function to detect z axis with max y axis
def z_max_y(df):
    '''
    INPUT: Pandas DataFrame
    OUTPUT: Float
    Determine the z value with the highest y variance.
    '''
    df = df.dropna(axis=0)
    unique_z = df['Start Z'].unique()

    std = [0] * len(unique_z)
    for i, z in enumerate(unique_z):
        std[i] = df[df['Start Z']==z]['Start Y'].std()
    std_arr = np.array(std)
    std_arr =np.nan_to_num(std_arr)
    max_ind = np.where(std_arr==std_arr.max())

    return unique_z[max_ind][0]

#wing detection
def wing_detect(df):
    '''
    INPUT: Pandas DataFrame
    OUTPUT: Float
    Compute the distance between max(x) and min(x) for each y values
    along the z value that has the greatest y value. Then compute difference
    between the median and the mean of the distances.
    '''
    range_x = []
    z_wing = z_max_y(df)
    df = df[(abs(df['Start Z']-z_wing)<.1)]
    for y_val in df['Start Y'].unique():
        data_fix = df[(df['Start Y']==y_val)]
        if data_fix.size>0:
            x_max =data_fix['Start X'].max()
            x_min = data_fix['Start X'].min()
            range_x.append(x_max-x_min)
    return np.mean(range_x)-np.median(range_x)

#generate rotational matrices
def rotateXMatrix(radians):
    """
    INPUT: Rotation angle(in radians)
    OUTPUT: Numpy array of rotation matrix
    Return matrix for rotating about the x-axis by 'radians' radians """

    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]])

def rotateYMatrix(radians):
    """
    INPUT: Rotation angle(in radians)
    OUTPUT: Numpy array of rotation matrix
    Return matrix for rotating about the y-axis by 'radians' radians """

    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotateZMatrix(radians):
    """
    INPUT: Rotation angle(in radians)
    OUTPUT: Numpy array of rotation matrix
    Return matrix for rotating about the z-axis by 'radians' radians """

    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

#hood detect
def hood_detect(df):
    '''
    INPUT: Pandas DataFrame
    OUTPUT: Float
    Compute the distance between max(x)- min(x) on the highest z values
    for each unique_y values. Then compute the mean of the distances to
    detect if an object has a hood.
    '''
    range_x = []
    df = df[(abs(df['Start Z']-np.percentile(df['Start Z'],100)<.001))]
    for y_val in df['Start Y'].unique():
        data_fix = df[(df['Start Y']==y_val)]
        if data_fix.size>0:
            x_max =data_fix['Start X'].max()
            x_min = data_fix['Start X'].min()
            range_x.append(x_max-x_min)
    return np.mean(range_x)
