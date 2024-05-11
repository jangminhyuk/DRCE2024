#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import re
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

def summarize_lambda(lqg_lambda_values, lqg_theta_v_values, lqg_cost_values ,wdrc_lambda_values, wdrc_theta_v_values, wdrc_cost_values , drce_lambda_values, drce_theta_v_values, drce_cost_values, drcmmse_lambda_values, drcmmse_theta_v_values, drcmmse_cost_values,  dist, noise_dist, infinite, use_lambda, path):
    
    surfaces = []
    labels = []
    # Create 3D plot
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    })
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # -------------------
    # LQG
    # Interpolate cost values for smooth surface - LQG
    lambda_grid_lqg, theta_v_grid_lqg = np.meshgrid(
        np.linspace(min(lqg_lambda_values), max(lqg_lambda_values), 100),
        np.linspace(min(lqg_theta_v_values), max(lqg_theta_v_values), 100)
    )
    cost_grid_lqg = griddata(
        (lqg_lambda_values, lqg_theta_v_values), lqg_cost_values,
        (lambda_grid_lqg, theta_v_grid_lqg), method='cubic'
    )

    # Plot data points - LQG
    #ax.scatter(lqg_lambda_values, lqg_theta_values, lqg_cost_values, label='LQG')

    # Plot smooth surface - LQG
    surface_lqg =ax.plot_surface(lambda_grid_lqg, theta_v_grid_lqg, cost_grid_lqg, alpha=0.4, color='red', label='LQG')
    surfaces.append(surface_lqg)
    labels.append('LQG')
    #-------------------------
    
    # Repeat the process for WDRC
    # Interpolate cost values for smooth surface - WDRC
    lambda_grid_wdrc, theta_v_grid_wdrc = np.meshgrid(
    np.linspace(min(wdrc_lambda_values), max(wdrc_lambda_values), 100),
    np.linspace(min(wdrc_theta_v_values), max(wdrc_theta_v_values), 100)
    )
    cost_grid_wdrc = griddata(
        (wdrc_lambda_values, wdrc_theta_v_values), wdrc_cost_values,
        (lambda_grid_wdrc, theta_v_grid_wdrc), method='linear'  # Use linear interpolation
    )

    # Plot data points - WDRC
    #ax.scatter(wdrc_lambda_values, wdrc_theta_values, wdrc_cost_values, label='WDRC')

    # Plot smooth surface - WDRC
    surface_wdrc =ax.plot_surface(lambda_grid_wdrc, theta_v_grid_wdrc, cost_grid_wdrc, alpha=0.5, color='blue', label='WDRC+MMSE')
    surfaces.append(surface_wdrc)
    labels.append('WDRC+MMSE')
    #--------------
    #ax.scatter(drkf_lambda_values, drkf_theta_values, drkf_cost_values, label='DRKF')

    # Interpolate cost values for smooth surface - DRKF
    lambda_grid_drcmmse, theta_v_grid_drcmmse = np.meshgrid(
        np.linspace(min(drcmmse_lambda_values), max(drcmmse_lambda_values), 100),
        np.linspace(min(drcmmse_theta_v_values), max(drcmmse_theta_v_values), 100)
    )
    cost_grid_drcmmse = griddata(
        (drcmmse_lambda_values, drcmmse_theta_v_values), drcmmse_cost_values,
        (lambda_grid_drcmmse, theta_v_grid_drcmmse), method='cubic'
    )
    
    # Plot smooth surface - DCE
    surface_drcmmse = ax.plot_surface(lambda_grid_drcmmse, theta_v_grid_drcmmse, cost_grid_drcmmse, alpha=0.6, color='yellow', label='WDRC+DRMMSE ambiguity w/ x and v')
    surfaces.append(surface_drcmmse)
    labels.append('WDRC+DRMMSE ambiguity w/ x and v')
    
    #---------------------------
    # Interpolate cost values for smooth surface - DRKF
    lambda_grid_drce, theta_v_grid_drce = np.meshgrid(
        np.linspace(min(drce_lambda_values), max(drce_lambda_values), 100),
        np.linspace(min(drce_theta_v_values), max(drce_theta_v_values), 100)
    )
    cost_grid_drce = griddata(
        (drce_lambda_values, drce_theta_v_values), drce_cost_values,
        (lambda_grid_drce, theta_v_grid_drce), method='cubic'
    )
    
    # Plot smooth surface - DCE
    surface_drce = ax.plot_surface(lambda_grid_drce, theta_v_grid_drce, cost_grid_drce, alpha=0.6, color='green', label='WDR-CE')
    surfaces.append(surface_drce)
    labels.append('WDR-CE')
    
    
    ax.legend(handles=surfaces, labels=labels)
    
    # Set labels
    ax.set_xlabel(r'$\lambda$', fontsize=16)
    ax.set_ylabel(r'$\theta_v$', fontsize=16)
    ax.set_zlabel(r'Average MSE', fontsize=16, rotation=90, labelpad=3)
    
    ax.view_init(elev=20, azim=-65)
    
    plt.show()
    fig.savefig(path + 'params_mse_{}_{}.pdf'.format(dist, noise_dist), dpi=300, bbox_inches="tight", pad_inches=0.3)
    #plt.clf()
    
def summarize_theta_w(lqg_theta_w_values, lqg_theta_v_values, lqg_cost_values ,wdrc_theta_w_values, wdrc_theta_v_values, wdrc_cost_values , drce_theta_w_values, drce_theta_v_values, drce_cost_values, drcmmse_theta_w_values, drcmmse_theta_v_values, drcmmse_cost_values, dist, noise_dist, infinite, use_lambda, path):
    
    surfaces = []
    labels = []
    # Create 3D plot
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    })
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # -------------------
    # LQG
    # Interpolate cost values for smooth surface - LQG
    theta_w_grid_lqg, theta_v_grid_lqg = np.meshgrid(
        np.linspace(min(lqg_theta_w_values), max(lqg_theta_w_values), 100),
        np.linspace(min(lqg_theta_v_values), max(lqg_theta_v_values), 100)
    )
    cost_grid_lqg = griddata(
        (lqg_theta_w_values, lqg_theta_v_values), lqg_cost_values,
        (theta_w_grid_lqg, theta_v_grid_lqg), method='cubic'
    )

    # Plot data points - LQG
    #ax.scatter(lqg_lambda_values, lqg_theta_values, lqg_cost_values, label='LQG')

    # Plot smooth surface - LQG
    surface_lqg =ax.plot_surface(theta_w_grid_lqg, theta_v_grid_lqg, cost_grid_lqg, alpha=0.4, color='red', label='LQG')
    surfaces.append(surface_lqg)
    labels.append('LQG')
    #-------------------------
    
    # Repeat the process for WDRC
    # Interpolate cost values for smooth surface - WDRC
    theta_w_grid_wdrc, theta_v_grid_wdrc = np.meshgrid(
    np.linspace(min(wdrc_theta_w_values), max(wdrc_theta_w_values), 100),
    np.linspace(min(wdrc_theta_v_values), max(wdrc_theta_v_values), 100)
    )
    cost_grid_wdrc = griddata(
        (wdrc_theta_w_values, wdrc_theta_v_values), wdrc_cost_values,
        (theta_w_grid_wdrc, theta_v_grid_wdrc), method='linear'  # Use linear interpolation
    )

    # Plot data points - WDRC
    #ax.scatter(wdrc_lambda_values, wdrc_theta_values, wdrc_cost_values, label='WDRC')

    # Plot smooth surface - WDRC
    surface_wdrc =ax.plot_surface(theta_w_grid_wdrc, theta_v_grid_wdrc, cost_grid_wdrc,  color='blue', label='WDRC+MMSE')
    surfaces.append(surface_wdrc)
    labels.append('WDRC')
    #--------------

    # Interpolate cost values for smooth surface - WDRCMMSE
    theta_w_grid_drcmmse, theta_v_grid_drcmmse = np.meshgrid(
        np.linspace(min(drcmmse_theta_w_values), max(drcmmse_theta_w_values), 100),
        np.linspace(min(drcmmse_theta_v_values), max(drcmmse_theta_v_values), 100)
    )
    cost_grid_drcmmse = griddata(
        (drcmmse_theta_w_values, drcmmse_theta_v_values), drcmmse_cost_values,
        (theta_w_grid_drcmmse, theta_v_grid_drcmmse), method='cubic'
    )
    
    # Plot smooth surface - DCE
    surface_drcmmse = ax.plot_surface(theta_w_grid_drcmmse, theta_v_grid_drcmmse, cost_grid_drcmmse, alpha=0.9, color='aqua', label='WDRC+DRMMSE')
    surfaces.append(surface_drcmmse)
    labels.append('WDRC+DRMMSE')
    
    #--------------
    # Plot DRKF data points
    #ax.scatter(drkf_lambda_values, drkf_theta_values, drkf_cost_values, label='DRKF')

    # Interpolate cost values for smooth surface - DRKF
    theta_w_grid_drce, theta_v_grid_drce = np.meshgrid(
        np.linspace(min(drce_theta_w_values), max(drce_theta_w_values), 100),
        np.linspace(min(drce_theta_v_values), max(drce_theta_v_values), 100)
    )
    cost_grid_drce = griddata(
        (drce_theta_w_values, drce_theta_v_values), drce_cost_values,
        (theta_w_grid_drce, theta_v_grid_drce), method='cubic'
    )
    
    # Plot smooth surface - DCE
    surface_drce = ax.plot_surface(theta_w_grid_drce, theta_v_grid_drce, cost_grid_drce, alpha=0.6, color='green', label='WDR-CE')
    surfaces.append(surface_drce)
    labels.append('WDR-CE')
    
    #---------------
    ax.legend(handles=surfaces, labels=labels)
    
    # Set labels
    ax.set_xlabel(r'$\theta_w$', fontsize=16)
    ax.set_ylabel(r'$\theta_v$', fontsize=16)
    ax.set_zlabel(r'MSE', fontsize=16, rotation=90, labelpad=3)
    
    ax.view_init(elev=20, azim=-65)
    
    plt.show()
    fig.savefig(path + 'params_mse_{}_{}.pdf'.format(dist, noise_dist), dpi=300, bbox_inches="tight", pad_inches=0.3)
    #plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    parser.add_argument('--use_lambda', required=False, action="store_true") #use lambda results if flagged
    args = parser.parse_args()
    
    if args.infinite:
        if args.use_lambda:
            path = "./results/{}_{}/infinite/multiple/params_lambda/".format(args.dist, args.noise_dist)
        else:
            path = "./results/{}_{}/infinite/multiple/params_thetas/".format(args.dist, args.noise_dist)
    else:
        if args.use_lambda:
            path = "./results/{}_{}/finite/multiple/params_lambda/".format(args.dist, args.noise_dist)
        else:
            path = "./results/{}_{}/finite/multiple/params_thetas/".format(args.dist, args.noise_dist)

    #Load data
    drcmmse_theta_w_values =[]
    drcmmse_lambda_values = []
    drcmmse_theta_v_values = []
    drcmmse_cost_values = []
    drcmmse_mse_values = []
    
    drce_theta_w_values =[]
    drce_lambda_values = []
    drce_theta_v_values = []
    drce_cost_values = []
    drce_mse_values = []
    
    wdrc_theta_w_values = []
    wdrc_lambda_values = []
    wdrc_theta_v_values = []
    wdrc_cost_values = []
    wdrc_mse_values = []
    
    lqg_theta_w_values =[]
    lqg_lambda_values = []
    lqg_theta_v_values = []
    lqg_cost_values = []
    lqg_mse_values = []
    # theta_v_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    # lambda_list = [ 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    # TODO : Modify the theta_v_list and lambda_list below to match your experiments!!! 
    # theta_v_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    # lambda_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    theta_w_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if args.dist=='normal':
        lambda_list = [12, 15, 20, 25, 30, 35, 40, 45, 50] # disturbance distribution penalty parameter
    else:
        lambda_list = [15, 20, 25, 30, 35, 40, 45, 50] # disturbance distribution penalty parameter

    # Regular expression pattern to extract numbers from file names
    
    #---------
    # MSE
    if args.use_lambda:
        pattern_drce = r"drce_mse_(\d+)and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_drcmmse = r"drcmmse_mse_(\d+)and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_wdrc = r"wdrc_mse_(\d+)"
    else:
        pattern_drcmmse = r"drcmmse_mse_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_drce = r"drce_mse_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_wdrc = r"wdrc_mse_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
    pattern_lqg = r"lqg_mse"
    # Iterate over each file in the directory
    for filename in os.listdir(path):
        match = re.search(pattern_drce, filename)
        if match:
            if args.use_lambda:
                lambda_value = float(match.group(1))  # Extract lambda
                theta_v_value = float(match.group(2))   # Extract theta_v value
                theta_v_str = match.group(3)
                theta_v_value += float(theta_v_str)/10
                #changed _1_5_ to 1.5!
                # Store theta_w and theta values
                drce_lambda_values.append(lambda_value)
                drce_theta_v_values.append(theta_v_value)
            else:
                theta_w_value = float(match.group(1))  # Extract theta_w value
                theta_w_str = match.group(2)
                theta_w_value += float(theta_w_str)/10
                theta_v_value = float(match.group(3))   # Extract theta_v value
                theta_v_str = match.group(4)
                theta_v_value += float(theta_v_str)/10
                #changed _1_5_ to 1.5!
                # Store theta_w and theta values
                drce_theta_w_values.append(theta_w_value)
                drce_theta_v_values.append(theta_v_value)
            
            drce_file = open(path + filename, 'rb')
            drce_mse = pickle.load(drce_file)
            drce_file.close()
            drce_mse_values.append(drce_mse)  # Store cost value
        else:
            match_drcmmse = re.search(pattern_drcmmse, filename)
            if match_drcmmse:
                if args.use_lambda:
                    lambda_value = float(match_drcmmse.group(1))  # Extract lambda
                    theta_v_value = float(match_drcmmse.group(2))   # Extract theta_v value
                    theta_v_str = match_drcmmse.group(3)
                    theta_v_value += float(theta_v_str)/10
                    #changed _1_5_ to 1.5!
                    # Store theta_w and theta values
                    drcmmse_lambda_values.append(lambda_value)
                    drcmmse_theta_v_values.append(theta_v_value)
                else:
                    theta_w_value = float(match_drcmmse.group(1))  # Extract theta_w value
                    theta_w_str = match_drcmmse.group(2)
                    theta_w_value += float(theta_w_str)/10
                    theta_v_value = float(match_drcmmse.group(3))   # Extract theta_v value
                    theta_v_str = match_drcmmse.group(4)
                    theta_v_value += float(theta_v_str)/10
                    #changed _1_5_ to 1.5!
                    # Store theta_w and theta values
                    drcmmse_theta_w_values.append(theta_w_value)
                    drcmmse_theta_v_values.append(theta_v_value)
                
                drcmmse_file = open(path + filename, 'rb')
                drcmmse_mse = pickle.load(drcmmse_file)
                drcmmse_file.close()
                drcmmse_mse_values.append(drcmmse_mse)  # Store cost value
            else:
                match_wdrc = re.search(pattern_wdrc, filename)
                if match_wdrc: # wdrc
                    if args.use_lambda:
                        lambda_value = float(match_wdrc.group(1))  # Extract lambda
                    else:
                        theta_w_value = float(match_wdrc.group(1))  # Extract theta_w value
                        theta_w_str = match_wdrc.group(2)
                        theta_w_value += float(theta_w_str)/10
                    #print('theta w : ', theta_w_value)
                    wdrc_file = open(path + filename, 'rb')
                    wdrc_mse = pickle.load(wdrc_file)
                    wdrc_file.close()
                    #print(wdrc_cost[0])
                    for aux_theta_v in theta_v_list:
                        if args.use_lambda:
                            wdrc_lambda_values.append(lambda_value)
                        else:
                            wdrc_theta_w_values.append(theta_w_value)
                        #print(wdrc_cost[0])
                        wdrc_theta_v_values.append(aux_theta_v) # since wdrc not affected by theta v, just add auxilary theta for plot
                        wdrc_mse_values.append(wdrc_mse)
                        #print(wdrc_cost[0])
                else:
                    match_lqg = re.search(pattern_lqg, filename)
                    if match_lqg:
                        lqg_file = open(path + filename, 'rb')
                        lqg_mse = pickle.load(lqg_file)
                        lqg_file.close()
                        if args.use_lambda:
                            for aux_lambda in lambda_list:
                                for aux_theta_v in theta_v_list:
                                    lqg_lambda_values.append(aux_lambda)
                                    lqg_theta_v_values.append(aux_theta_v)
                                    lqg_mse_values.append(lqg_mse)
                                    #print(lqg_cost[0])
                        else:
                            for aux_theta_w in theta_w_list:
                                for aux_theta_v in theta_v_list:
                                    lqg_theta_w_values.append(aux_theta_w)
                                    lqg_theta_v_values.append(aux_theta_v)
                                    lqg_mse_values.append(lqg_mse)    

                    

    # Convert lists to numpy arrays
    if args.use_lambda:
        drcmmse_lambda_values = np.array(drcmmse_lambda_values)
        drce_lambda_values = np.array(drce_lambda_values)
        wdrc_lambda_values = np.array(wdrc_lambda_values)
        lqg_lambda_values = np.array(lqg_lambda_values)
    else:
        drcmmse_theta_w_values = np.array(drcmmse_theta_w_values)
        drce_theta_w_values = np.array(drce_theta_w_values)
        wdrc_theta_w_values = np.array(wdrc_theta_w_values)
        lqg_theta_w_values = np.array(lqg_theta_w_values)
    
    drcmmse_theta_v_values = np.array(drcmmse_theta_v_values)
    drcmmse_mse_values = np.array(drcmmse_mse_values)
    
    drce_theta_v_values = np.array(drce_theta_v_values)
    drce_mse_values = np.array(drce_mse_values)

    wdrc_theta_v_values = np.array(wdrc_theta_v_values)
    wdrc_mse_values = np.array(wdrc_mse_values)
    
    lqg_theta_v_values = np.array(lqg_theta_v_values)
    lqg_mse_values = np.array(lqg_mse_values)
    
    if args.use_lambda:
        summarize_lambda(lqg_lambda_values, lqg_theta_v_values, lqg_mse_values ,wdrc_lambda_values, wdrc_theta_v_values, wdrc_mse_values , drce_lambda_values, drce_theta_v_values, drce_mse_values, drcmmse_lambda_values, drcmmse_theta_v_values, drcmmse_mse_values, args.dist, args.noise_dist, args.infinite, args.use_lambda, path)
    else:
        summarize_theta_w(lqg_theta_w_values, lqg_theta_v_values, lqg_mse_values ,wdrc_theta_w_values, wdrc_theta_v_values, wdrc_mse_values , drce_theta_w_values, drce_theta_v_values, drce_mse_values, drcmmse_theta_w_values, drcmmse_theta_v_values, drcmmse_mse_values, args.dist, args.noise_dist, args.infinite, args.use_lambda, path)

