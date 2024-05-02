# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:41:32 2023
Updated on Wed Feb  7 13:21:15 2024
@author: Liam Pledger - liam.pledger@pg.canterbury.ac.nz
"""
import sys

sys.path.append(r'C:\Users\dde62\Anaconda3\envs\dcc\Lib\site-packages')

import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from pickle import dump, load
from ipywidgets import*
from matplotlib import pyplot as plt

def pretrain_model(units):
    file = "RC column database.csv"

    # Load data
    columns_to_load = [3, 4, 9, 11, 12, 14, 15, 16, 17, 18, 21]
    data = pd.read_csv(file, skiprows=1, delimiter=",", usecols=columns_to_load)

    # Extract features and labels
    X = data.iloc[:, :-1].values  # Measured drift capacity
    Y = data.iloc[:, -1].values   # Column properties 


    if units != 1:  # converting units from the database of column to imperial based on input by user. 
        X[:, 0] = X[:, 0] * 25.4  # a, aspect ratio
        X[:, 2] = X[:, 2] / 145   # fyt, yield strength of transverse reinforcement
        X[:, 4] = X[:, 4] / 145   # fc, compressive strength of concrete
        X[:, 8] = X[:, 8] / 145   # ρl*fyl
        X[:, 9] = X[:, 9] / 145   # ρt*fyt
        

    # Create a LightGBM dataset for training
    train_data = lgb.Dataset(X, label=Y)

    # Define hyperparameters for the LightGBM model
    params = {
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'max_bin' : 50,
        'learning_rate': 0.0075,
        'feature_fraction': 0.4,
    }

    # Turn off warning messages (set logging level to 'silent')
    params['verbose'] = -1  # Set verbose to -1 to suppress warnings

    # Train the LightGBM model
    num_round = 5000
    bst = lgb.train(params, train_data, num_round)
    return bst
    

def load_pretrained_model(units=1):
    if units !=1:
        fl = 'pretrain_imperial.pkl'
    else:
        fl = 'pretrain.pkl'
    
    if os.path.isfile(fl):
        with open(fl, 'rb') as fp:
            bst=load(fp)
    else:
        bst = pretrain_model(units)
        with open(fl,'wb') as fp:
            dump(bst, fp)
    return bst

def _drift_calculator(var1, d, box, textbox, bst):
    
    a = 1500
    # d = 300
    s = 100
    fc = 30
    fyl = 500
    fyt = 500
    rhol = 0.02
    rhot = 0.005
    v = 0.3
    lbd = 25

    input_data = np.transpose(np.array([[a], [a/d], [fyt], [s/d], [fc], [v], [s/lbd], [a/s], [rhol*fyl], [rhot*fyt]], dtype=float))

    y_pred = bst.predict(input_data, num_iteration=bst.best_iteration)
    print(f'var1={var1}, y_pred={y_pred}, box={box}, textbox={textbox}')

    f,ax=plt.subplots(1,1)
    if box: c='k'
    else: c='r'
    ax.plot(d,y_pred[0],c+'o')
    ax.set_xlim([250,350])
    ax.set_ylim([2.0,3.0])
    ax.set_xlabel('d')
    ax.set_ylabel('ypred')
    ax.set_title('interactive updating plot')
    

def drift_calculator():
    slider1=IntSlider(value=150, min=20, max=240, step=10, description='$T_m$', continuous_update=False)
    slider2=FloatSlider(value=300, min=250, max=350, step=10, description='$d$', continuous_update=False)
    checkbox = Checkbox(value = False, description='Best (LSQ) Model')
    box2=BoundedFloatText(value=2.2, description='predict at')
    bst=load_pretrained_model()
    io=interactive_output(_drift_calculator, {'var1':slider1,'d':slider2,'box':checkbox,'textbox':box2,'bst':fixed(bst)})
    return VBox([HBox([slider1, VBox([slider2, checkbox]), box2]),io])

def main():
    # Open the user_inputs.txt file for reading
    with open('user_inputs.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Parse the input data from the file
    units = int(lines[0].strip().split(': ')[1])
    a = float(lines[1].strip().split(': ')[1])
    d = float(lines[2].strip().split(': ')[1])
    s = float(lines[3].strip().split(': ')[1])
    fc = float(lines[4].strip().split(': ')[1])
    fyl = float(lines[5].strip().split(': ')[1])
    fyt = float(lines[6].strip().split(': ')[1])
    rhol = float(lines[7].strip().split(': ')[1])
    rhot = float(lines[8].strip().split(': ')[1])
    v = float(lines[9].strip().split(': ')[1])
    lbd = float(lines[10].strip().split(': ')[1])

    input_data = np.transpose(np.array([[a], [a/d], [fyt], [s/d], [fc], [v], [s/lbd], [a/s], [rhol*fyl], [rhot*fyt]], dtype=float))

    # Make predictions on the test data
    bst=load_pretrained_model(units)
    y_pred = bst.predict(input_data, num_iteration=bst.best_iteration)


    print("""
                
        
        """)
    print("The estimated drift capacity of the column is = " + str(np.round(y_pred[0], 1)) + " %")

if __name__=="__main__":
    main()