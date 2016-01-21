import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math
import operator
import cPickle as pickle
import pandas as pd
import os, random
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from algorithm import wing_detect, hood_detect, z_max_y, rotateXMatrix, rotateYMatrix, rotateZMatrix

#Extract data for motorcycles
#Create dictionary to hold the data for each file
motor_dict = {'width': [], 'height': [],'length':[],'number':[],
             'ratio_xy':[],'ratio_midz':[], 'ratio_mid_pts':[],
             'ratio_xz':[],'ratio_yz':[],'y_var_ratio':[], 'z_var_ratio':[]}
#Extract data for all files
for i in xrange(1,len(os.listdir("../data/motorcycles"))):
    #load data from file
    filename = '../data/motorcycles/motor'+str(i)+'.csv'
    load_dat = pd.read_csv(filename)
    #There are 6 relevant columns and 3 of them repeat so concat them on top of each other
    dat_1 = load_dat[['Start X','Start Y','Start Z']]
    dat_2 = load_dat[['End X','End Y','End Z']]
    dat_2.columns = ['Start X','Start Y','Start Z']
    dat_new = pd.concat((dat_1,dat_2),axis=0)
    dat_new = dat_new.dropna(axis=0)
    columns = dat_new.columns
    #normalize using the norm
    dat_new = (dat_new -dat_new.mean())/np.linalg.norm(dat_new.values)
    # find highest, lowest standard deviation vector and orient the model accordingly
    highest = np.where(dat_new.std()==dat_new.std().max())[0][0]
    lowest = np.where(dat_new.std()==dat_new.std().min())[0][0]
    if highest == 1:
        dat_new = np.dot(dat_new,rotateZMatrix(math.pi/2))
    elif highest == 2:
        dat_new = np.dot(dat_new.values,rotateYMatrix(math.pi/2))
    dat_new = pd.DataFrame(dat_new, columns = ['Start X','Start Y','Start Z'])
    dat_new['number']= np.ones((dat_new.shape[0],1))*i
    #max and min xyz after scale
    max_x = dat_new['Start X'].max()
    max_y = dat_new['Start Y'].max()
    max_z = dat_new['Start Z'].max()
    min_x = dat_new['Start X'].min()
    min_y = dat_new['Start Y'].min()
    min_z = dat_new['Start Z'].min()
    #calculate width height length
    width = max_y-min_y
    length = max_x-min_x
    height = max_z-min_z
    #ratio of length and width
    ratio_xy = float(dat_new['Start Y'].std())/(
        dat_new['Start X'].std())
    ratio_xz = float(dat_new['Start X'].std())/(
        dat_new['Start Z'].std())
    ratio_yz = float(dat_new['Start Y'].std())/(
        dat_new['Start Z'].std())
    #create ratio of points above and below midpoint in the z direction
    midz = dat_new['Start Z'].mean()
    z_pts_above = dat_new[dat_new['Start Z']>midz]['Start Z'].count()
    z_pts_below = dat_new[dat_new['Start Z']<midz]['Start Z'].count()
    ratio_mid_pts = float(z_pts_above)/z_pts_below
    #create ratio of distance above and below midpoint in the z direction
    zabove = max_z - midz
    zbelow = midz - min_z
    ratio_midz = float(zabove)/zbelow
    #detect wings through diff of median and mean in the y direction
    y_var_ratio = wing_detect(dat_new)
    #detect hood
    z_var_ratio = hood_detect(dat_new)
    #append to dictionary
    motor_dict['width'].append(width)
    motor_dict['length'].append(length)
    motor_dict['height'].append(height)
    motor_dict['ratio_xy'].append(ratio_xy)
    motor_dict['ratio_xz'].append(ratio_xz)
    motor_dict['ratio_yz'].append(ratio_yz)
    motor_dict['ratio_midz'].append(ratio_midz)
    motor_dict['ratio_mid_pts'].append(ratio_mid_pts)
    motor_dict['y_var_ratio'].append(y_var_ratio)
    motor_dict['z_var_ratio'].append(z_var_ratio)
    motor_dict['number'].append(i)

data_feat = pd.DataFrame(data=motor_dict, index=range(len(motor_dict['width'])))
data_feat['y'] = np.array(['0']*data_feat.shape[0])

#Extract data for training from car dataset
#Create dictionary to hold the data for each file
car_dict = {'width': [], 'height': [],'length':[],'number':[],
             'ratio_xy':[],'ratio_midz':[],'ratio_mid_pts':[],
             'ratio_xz':[],'ratio_yz':[],'y_var_ratio':[], 'z_var_ratio':[]}
#Extract data for all files
for i in xrange(1,len(os.listdir("../data/cars"))):
    #load the data from file
    filename = '../data/cars/car'+str(i)+'.csv'
    load_dat = pd.read_csv(filename)
    #There are 6 relevant columns and 3 of them repeat so concat them on top of each other
    dat_1 = load_dat[['Start X','Start Y','Start Z']]
    dat_2 = load_dat[['End X','End Y','End Z']]
    dat_2.columns = ['Start X','Start Y','Start Z']
    dat_new = pd.concat((dat_1,dat_2),axis=0)
    dat_new = dat_new.dropna(axis=0)
    columns = dat_new.columns
    #normalize using the norm
    dat_new = (dat_new -dat_new.mean())/np.linalg.norm(dat_new.values)
    # find highest, lowest standard deviation vector and orient the model accordingly
    highest = np.where(dat_new.std()==dat_new.std().max())[0][0]
    lowest = np.where(dat_new.std()==dat_new.std().min())[0][0]
    if highest == 1:
        dat_new = np.dot(dat_new,rotateZMatrix(math.pi/2))
    elif highest == 2:
        dat_new = np.dot(dat_new.values,rotateYMatrix(math.pi/2))
    dat_new = pd.DataFrame(dat_new, columns = ['Start X','Start Y','Start Z'])
    dat_new['number']= np.ones((dat_new.shape[0],1))*i
    # dat_new.reset_index(drop=True,inplace=True)
    # ind = np.random.choice(dat_new.index, size=dat_new.shape[0]*0.1, replace=False)
    # dat_new = dat_new.iloc[ind,:]
    #max and min xyz after scale
    max_x = dat_new['Start X'].max()
    max_y = dat_new['Start Y'].max()
    max_z = dat_new['Start Z'].max()
    min_x = dat_new['Start X'].min()
    min_y = dat_new['Start Y'].min()
    min_z = dat_new['Start Z'].min()
    #calculate width height length
    width = max_y-min_y
    length = max_x-min_x
    height = max_z-min_z
    #ratio of length and width
    ratio_xy = float(dat_new['Start Y'].std())/(
        dat_new['Start X'].std())
    ratio_xz = float(dat_new['Start X'].std())/(
        dat_new['Start Z'].std())
    ratio_yz = float(dat_new['Start Y'].std())/(
        dat_new['Start Z'].std())
    #create ratio of points above and below midpoint in the z direction
    midz = dat_new['Start Z'].mean()
    z_pts_above = dat_new[dat_new['Start Z']>midz]['Start Z'].count()
    z_pts_below = dat_new[dat_new['Start Z']<midz]['Start Z'].count()
    ratio_mid_pts = float(z_pts_above)/z_pts_below
    #create ratio of distance above and below midpoint
    zabove = max_z - midz
    zbelow = midz - min_z
    ratio_midz = float(zabove)/zbelow
    #detect wings through diff of median and mean in the y direction
    y_var_ratio = wing_detect(dat_new)
    #detect hood
    z_var_ratio = hood_detect(dat_new)
    #append to dictionary
    car_dict['width'].append(width)
    car_dict['length'].append(length)
    car_dict['height'].append(height)
    car_dict['ratio_xy'].append(ratio_xy)
    car_dict['ratio_xz'].append(ratio_xz)
    car_dict['ratio_yz'].append(ratio_yz)
    car_dict['ratio_midz'].append(ratio_midz)
    car_dict['ratio_mid_pts'].append(ratio_mid_pts)
    car_dict['y_var_ratio'].append(y_var_ratio)
    car_dict['z_var_ratio'].append(z_var_ratio)
    car_dict['number'].append(i)
car_feat = pd.DataFrame(data=car_dict, index=range(len(car_dict['width'])))
car_feat['y'] = np.array(['1']*car_feat.shape[0])

#Extract data for training from planes dataset
#Create dictionary to hold the data for each file
pl_dict = {'width': [], 'height': [],'length':[],'number':[],
             'ratio_xy':[],'ratio_midz':[],'ratio_mid_pts':[],
             'ratio_xz':[],'ratio_yz':[],'y_var_ratio':[],'z_var_ratio':[]}
#Extract data for all files
for i in xrange(1,len(os.listdir("../data/planes"))):
    #load data from file
    filename = '../data/planes/plane'+str(i)+'.csv'
    load_dat = pd.read_csv(filename)
    #There are 6 relevant columns and 3 of them repeat so concat them on top of each other
    dat_1 = load_dat[['Start X','Start Y','Start Z']]
    dat_2 = load_dat[['End X','End Y','End Z']]
    dat_2.columns = ['Start X','Start Y','Start Z']
    dat_new = pd.concat((dat_1,dat_2),axis=0)
    dat_new = dat_new.dropna(axis=0)
    columns = dat_new.columns
    #normalize using the norm
    dat_new = (dat_new -dat_new.mean())/np.linalg.norm(dat_new.values)
    # find highest, lowest standard deviation vector and orient the model accordingly
    highest = np.where(dat_new.std()==dat_new.std().max())[0][0]
    lowest = np.where(dat_new.std()==dat_new.std().min())[0][0]
    if highest == 1:
        dat_new = np.dot(dat_new,rotateZMatrix(math.pi/2))
    elif highest == 2:
        dat_new = np.dot(dat_new.values,rotateYMatrix(math.pi/2))
    dat_new = pd.DataFrame(dat_new, columns = ['Start X','Start Y','Start Z'])
    dat_new['number']= np.ones((dat_new.shape[0],1))*i
    #sample with sampling rate
    # dat_new.reset_index(drop=True,inplace=True)
    # ind = np.random.choice(dat_new.index, size=dat_new.shape[0]*0.1, replace=False)
    # dat_new = dat_new.iloc[ind,:]
    #max and min xyz after scale
    max_x = dat_new['Start X'].max()
    max_y = dat_new['Start Y'].max()
    max_z = dat_new['Start Z'].max()
    min_x = dat_new['Start X'].min()
    min_y = dat_new['Start Y'].min()
    min_z = dat_new['Start Z'].min()
    #calculate width height length
    width = max_y-min_y
    length = max_x-min_x
    height = max_z-min_z
    #ratio of length and width
    ratio_xy = float(dat_new['Start Y'].std())/(
        dat_new['Start X'].std())
    ratio_xz = float(dat_new['Start X'].std())/(
        dat_new['Start Z'].std())
    ratio_yz = float(dat_new['Start Y'].std())/(
        dat_new['Start Z'].std())
    #create ratio of points above and below midpoint in the z direction
    midz = dat_new['Start Z'].mean()
    z_pts_above = dat_new[dat_new['Start Z']>midz]['Start Z'].count()
    z_pts_below = dat_new[dat_new['Start Z']<midz]['Start Z'].count()
    ratio_mid_pts = float(z_pts_above)/z_pts_below
    #create ratio of distance above and below midpoint in z direction
    zabove = max_z - midz
    zbelow = midz - min_z
    ratio_midz = float(zabove)/zbelow
    #detect wings through diff of median and mean in the y direction
    y_var_ratio = wing_detect(dat_new)
    #detect hood
    z_var_ratio = hood_detect(dat_new)
    #append to dictionary
    pl_dict['width'].append(width)
    pl_dict['length'].append(length)
    pl_dict['height'].append(height)
    pl_dict['ratio_xy'].append(ratio_xy)
    pl_dict['ratio_xz'].append(ratio_xz)
    pl_dict['ratio_yz'].append(ratio_yz)
    pl_dict['ratio_midz'].append(ratio_midz)
    pl_dict['ratio_mid_pts'].append(ratio_mid_pts)
    pl_dict['y_var_ratio'].append(y_var_ratio)
    pl_dict['z_var_ratio'].append(z_var_ratio)
    pl_dict['number'].append(i)
pl_feat = pd.DataFrame(data=pl_dict, index=range(len(pl_dict['width'])))
pl_feat['y'] = np.array(['2']*pl_feat.shape[0])

#Extract data for training from convertible cars dataset
#Create dictionary to hold the data for each file
con_dict = {'width': [], 'height': [],'length':[],'number':[],
             'ratio_xy':[],'ratio_midz':[],'ratio_mid_pts':[],
             'ratio_xz':[],'ratio_yz':[],'y_var_ratio':[],'z_var_ratio':[]}
#Extract data for all files
for i in xrange(1,len(os.listdir("../data/convertibles"))):
    #load data from file
    filename = '../data/convertibles/con'+str(i)+'.csv'
    load_dat = pd.read_csv(filename)
    #There are 6 relevant columns and 3 of them repeat so concat them on top of each other
    dat_1 = load_dat[['Start X','Start Y','Start Z']]
    dat_2 = load_dat[['End X','End Y','End Z']]
    dat_2.columns = ['Start X','Start Y','Start Z']
    dat_new = pd.concat((dat_1,dat_2),axis=0)
    dat_new = dat_new.dropna(axis=0)
    columns = dat_new.columns
    #normalize the data from norm
    dat_new = (dat_new -dat_new.mean())/np.linalg.norm(dat_new.values)#dat_new.std().mean()
    # find highest, lowest standard deviation vector and orient the model accordingly
    highest = np.where(dat_new.std()==dat_new.std().max())[0][0]
    lowest = np.where(dat_new.std()==dat_new.std().min())[0][0]
    if highest == 1:
        dat_new = np.dot(dat_new,rotateZMatrix(math.pi/2))
    elif highest == 2:
        dat_new = np.dot(dat_new.values,rotateYMatrix(math.pi/2))
    dat_new = pd.DataFrame(dat_new, columns = ['Start X','Start Y','Start Z'])
    dat_new['number']= np.ones((dat_new.shape[0],1))*i
    # #sampling with sampling rate
    # dat_new.reset_index(drop=True,inplace=True)
    # ind = np.random.choice(dat_new.index, size=dat_new.shape[0]*0.1, replace=False)
    # dat_new = dat_new.iloc[ind,:]
    #max and min xyz after scale
    max_x = dat_new['Start X'].max()
    max_y = dat_new['Start Y'].max()
    max_z = dat_new['Start Z'].max()
    min_x = dat_new['Start X'].min()
    min_y = dat_new['Start Y'].min()
    min_z = dat_new['Start Z'].min()
    #calculate width height length
    width = max_y-min_y
    length = max_x-min_x
    height = max_z-min_z
    #ratio of length and width
    ratio_xy = float(dat_new['Start Y'].std())/(
        dat_new['Start X'].std())
    ratio_xz = float(dat_new['Start X'].std())/(
        dat_new['Start Z'].std())
    ratio_yz = float(dat_new['Start Y'].std())/(
        dat_new['Start Z'].std())
    #ratio of points above and below midpoint in the z direction
    midz = dat_new['Start Z'].mean()
    z_pts_above = dat_new[dat_new['Start Z']>midz]['Start Z'].count()
    z_pts_below = dat_new[dat_new['Start Z']<midz]['Start Z'].count()
    ratio_mid_pts = float(z_pts_above)/z_pts_below
    #create ratio of distance above and below midpoint in z direction
    zabove = max_z - midz
    zbelow = midz - min_z
    ratio_midz = float(zabove)/zbelow
    #detect wings through diff of median and mean in the y direction
    y_var_ratio = wing_detect(dat_new)
    #detect hood
    z_var_ratio = hood_detect(dat_new)
    #append to dictionary
    con_dict['width'].append(width)
    con_dict['length'].append(length)
    con_dict['height'].append(height)
    con_dict['ratio_xy'].append(ratio_xy)
    con_dict['ratio_xz'].append(ratio_xz)
    con_dict['ratio_yz'].append(ratio_yz)
    con_dict['ratio_midz'].append(ratio_midz)
    con_dict['ratio_mid_pts'].append(ratio_mid_pts)
    con_dict['y_var_ratio'].append(y_var_ratio)
    con_dict['z_var_ratio'].append(z_var_ratio)
    con_dict['number'].append(i)
con_feat = pd.DataFrame(data=con_dict, index=range(len(con_dict['width'])))
con_feat['y'] = np.array(['3']*con_feat.shape[0])

#Extract data for training from trains dataset
#Create dictionary to hold the data for each file
train_dict = {'width': [], 'height': [],'length':[],'number':[],
             'ratio_xy':[],'ratio_midz':[],'ratio_mid_pts':[],
             'ratio_xz':[],'ratio_yz':[],'y_var_ratio':[],'z_var_ratio':[]}
#Extract data for all files
for i in xrange(1,len(os.listdir("../data/trains"))):
    #load data from file
    filename = '../data/trains/train'+str(i)+'.csv'
    load_dat = pd.read_csv(filename)
    #There are 6 relevant columns and 3 of them repeat so concat them on top of each other
    dat_1 = load_dat[['Start X','Start Y','Start Z']]
    dat_2 = load_dat[['End X','End Y','End Z']]
    dat_2.columns = ['Start X','Start Y','Start Z']
    dat_new = pd.concat((dat_1,dat_2),axis=0)
    dat_new = dat_new.dropna(axis=0)
    columns = dat_new.columns
    #normalize data using norm
    dat_new = (dat_new -dat_new.mean())/np.linalg.norm(dat_new.values)#dat_new.std().mean()
    # find highest, lowest standard deviation vector and orient the model accordingly
    highest = np.where(dat_new.std()==dat_new.std().max())[0][0]
    lowest = np.where(dat_new.std()==dat_new.std().min())[0][0]
    if highest == 1:
        dat_new = np.dot(dat_new,rotateZMatrix(math.pi/2))
    elif highest == 2:
        dat_new = np.dot(dat_new.values,rotateYMatrix(math.pi/2))
    dat_new = pd.DataFrame(dat_new, columns = ['Start X','Start Y','Start Z'])
    dat_new['number']= np.ones((dat_new.shape[0],1))*i
    # #sample with sampling rate
    # dat_new.reset_index(drop=True,inplace=True)
    # ind = np.random.choice(dat_new.index, size=dat_new.shape[0]*0.1, replace=False)
    # dat_new = dat_new.iloc[ind,:]
    #max and min xyz after scale
    max_x = dat_new['Start X'].max()
    max_y = dat_new['Start Y'].max()
    max_z = dat_new['Start Z'].max()
    min_x = dat_new['Start X'].min()
    min_y = dat_new['Start Y'].min()
    min_z = dat_new['Start Z'].min()
    #calculate width height length
    width = max_y-min_y
    length = max_x-min_x
    height = max_z-min_z
    #ratio of length and width
    ratio_xy = float(dat_new['Start Y'].std())/(
        dat_new['Start X'].std())
    ratio_xz = float(dat_new['Start X'].std())/(
        dat_new['Start Z'].std())
    ratio_yz = float(dat_new['Start Y'].std())/(
        dat_new['Start Z'].std())
    #ratio of points above and below midpoint in the z direction
    midz = dat_new['Start Z'].mean()
    z_pts_above = dat_new[dat_new['Start Z']>midz]['Start Z'].count()
    z_pts_below = dat_new[dat_new['Start Z']<midz]['Start Z'].count()
    ratio_mid_pts = float(z_pts_above)/z_pts_below
    #create ratio of distance above and below midpoint
    zabove = max_z - midz
    zbelow = midz - min_z
    ratio_midz = float(zabove)/zbelow
    #detect wings through diff of median and mean in the y direction
    y_var_ratio = wing_detect(dat_new)
    #detect hood
    z_var_ratio = hood_detect(dat_new)
    #append to dictionary
    train_dict['width'].append(width)
    train_dict['length'].append(length)
    train_dict['height'].append(height)
    train_dict['ratio_xy'].append(ratio_xy)
    train_dict['ratio_xz'].append(ratio_xz)
    train_dict['ratio_yz'].append(ratio_yz)
    train_dict['ratio_midz'].append(ratio_midz)
    train_dict['ratio_mid_pts'].append(ratio_mid_pts)
    train_dict['y_var_ratio'].append(y_var_ratio)
    train_dict['z_var_ratio'].append(z_var_ratio)
    train_dict['number'].append(i)
train_feat = pd.DataFrame(data=train_dict, index=range(len(train_dict['width'])))
train_feat['y'] = np.array(['4']*train_feat.shape[0])


#Extract data for training from convertible cars dataset
#Create dictionary to hold the data for each file
heli_dict = {'width': [], 'height': [],'length':[],'number':[],
             'ratio_xy':[],'ratio_midz':[],'ratio_mid_pts':[],
             'ratio_xz':[],'ratio_yz':[],'y_var_ratio':[],'z_var_ratio':[]}
#Extract data for all files
for i in xrange(1,len(os.listdir("../data/helicopter"))):
    #load data from file
    filename = '../data/helicopter/heli'+str(i)+'.csv'
    load_dat = pd.read_csv(filename)
    #There are 6 relevant columns and 3 of them repeat so concat them on top of each other
    dat_1 = load_dat[['Start X','Start Y','Start Z']]
    dat_2 = load_dat[['End X','End Y','End Z']]
    dat_2.columns = ['Start X','Start Y','Start Z']
    dat_new = pd.concat((dat_1,dat_2),axis=0)
    dat_new = dat_new.dropna(axis=0)
    columns = dat_new.columns
    #normalize data using the norm
    dat_new = (dat_new -dat_new.mean())/np.linalg.norm(dat_new.values)#dat_new.std().mean()
    # find highest, lowest standard deviation vector and orient the model accordingly
    highest = np.where(dat_new.std()==dat_new.std().max())[0][0]
    lowest = np.where(dat_new.std()==dat_new.std().min())[0][0]
    if highest == 1:
        dat_new = np.dot(dat_new,rotateZMatrix(math.pi/2))
    elif highest == 2:
        dat_new = np.dot(dat_new.values,rotateYMatrix(math.pi/2))
    dat_new = pd.DataFrame(dat_new, columns = ['Start X','Start Y','Start Z'])
    dat_new['number']= np.ones((dat_new.shape[0],1))*i
    #max and min xyz after scale
    max_x = dat_new['Start X'].max()
    max_y = dat_new['Start Y'].max()
    max_z = dat_new['Start Z'].max()
    min_x = dat_new['Start X'].min()
    min_y = dat_new['Start Y'].min()
    min_z = dat_new['Start Z'].min()
    #calculate width height length
    width = max_y-min_y
    length = max_x-min_x
    height = max_z-min_z
    #ratio of length and width
    ratio_xy = float(dat_new['Start Y'].std())/(
        dat_new['Start X'].std())
    ratio_xz = float(dat_new['Start X'].std())/(
        dat_new['Start Z'].std())
    ratio_yz = float(dat_new['Start Y'].std())/(
        dat_new['Start Z'].std())
    #ratio of points above and below midpoint in the z direction
    midz = dat_new['Start Z'].mean()
    z_pts_above = dat_new[dat_new['Start Z']>midz]['Start Z'].count()
    z_pts_below = dat_new[dat_new['Start Z']<midz]['Start Z'].count()
    ratio_mid_pts = float(z_pts_above)/z_pts_below
    #create ratio of distance above and below midpoint in z direction
    zabove = max_z - midz
    zbelow = midz - min_z
    ratio_midz = float(zabove)/zbelow
    #detect wings through diff of median and mean in the y direction
    y_var_ratio = wing_detect(dat_new)
    #detect hood
    z_var_ratio = hood_detect(dat_new)
    #append to dictionary
    heli_dict['width'].append(width)
    heli_dict['length'].append(length)
    heli_dict['height'].append(height)
    heli_dict['ratio_xy'].append(ratio_xy)
    heli_dict['ratio_xz'].append(ratio_xz)
    heli_dict['ratio_yz'].append(ratio_yz)
    heli_dict['ratio_midz'].append(ratio_midz)
    heli_dict['ratio_mid_pts'].append(ratio_mid_pts)
    heli_dict['y_var_ratio'].append(y_var_ratio)
    heli_dict['z_var_ratio'].append(z_var_ratio)
    heli_dict['number'].append(i)
heli_feat = pd.DataFrame(data=heli_dict, index=range(len(heli_dict['width'])))
heli_feat['y'] = np.array(['5']*heli_feat.shape[0])

#Clean up large outliers
data_out = data_feat
car_out = car_feat
pl_out = pl_feat
con_out= con_feat
train_out=train_feat
heli_out = heli_feat
columns = data_out.columns.drop(['number','y'])
for col in columns:
    data_out = data_out[(data_out[col]<np.percentile(data_out[col],100))]

for col in columns:
    car_out = car_out[(car_out[col]<np.percentile(car_out[col],100))]

for col in columns:
    pl_out = pl_out[(pl_out[col]<np.percentile(pl_out[col],100))]

for col in columns:
    con_out = con_out[(con_out[col]<np.percentile(con_out[col],100))]

for col in columns:
    train_out = train_out[(train_out[col]<np.percentile(train_out[col],100))]

for col in columns:
    heli_out = heli_out[(heli_out[col]<np.percentile(heli_out[col],100))]

#Combine data for each class
data_all= pd.concat((car_out, con_out, heli_out, pl_out,data_out, train_out),axis=0)
#Randomize the order and reset indices
data_all.reset_index(drop=True,inplace=True)
ind = np.random.choice(data_all.index, size=data_all.shape[0], replace=False)
data_all = data_all.iloc[ind,:]
data_all.reset_index(drop=True,inplace=True)
data_all = data_all.dropna(axis=0)

#Scale the data using Standard Scaler
scale = StandardScaler()
Y = data_all['y'].values
X = data_all.drop(['y','number'],axis=1).values
scaler = scale.fit(X)
X = scaler.transform(X)

#Train model
svm = SVC(class_weight='balanced',probability=True, C=100)
model = svm.fit(X, Y)

#Pickle Model and Scaler
with open('model.pkl', 'w') as f:
     model = svm.fit(X, Y)
     pickle.dump(model, f)

with open('scaler.pkl', 'w') as f:
    pickle.dump(scaler, f)
