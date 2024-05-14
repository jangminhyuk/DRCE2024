#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize_noise(avg_time_drlqc_0_0001, std_time_drlqc_0_0001,avg_time_drlqc_0_001, std_time_drlqc_0_001,avg_time_drlqc_0_01, std_time_drlqc_0_01, avg_time_drce, std_time_drce, dist, noise_dist, infinite, path):

    #dimension_list
    d = [2,3,4,5,6,7,8,9,10,20,30,40,50]
    
    drlqc_0_0001_avgT = np.array(avg_time_drlqc_0_0001[0:])
    drlqc_0_0001_stdT = np.array(std_time_drlqc_0_0001[0:])
    
    drlqc_0_001_avgT = np.array(avg_time_drlqc_0_001[0:])
    drlqc_0_001_stdT = np.array(std_time_drlqc_0_001[0:])
    
    drlqc_0_01_avgT = np.array(avg_time_drlqc_0_01[0:])
    drlqc_0_01_stdT = np.array(std_time_drlqc_0_01[0:])
    
    drce_avgT = np.array(avg_time_drce[0:])
    drce_stdT = np.array(std_time_drce[0:])
    
    
    fig= plt.figure(figsize=(6,4), dpi=300)
    
    #----------------------------------------------
    plt.plot(d[:10], drlqc_0_0001_avgT, color='dimgrey', label='DRLQC (1e-4)',linestyle=':')
    plt.fill_between(d[:10], drlqc_0_0001_avgT + 2*drlqc_0_0001_stdT, drlqc_0_0001_avgT - 2*drlqc_0_0001_stdT, facecolor='grey', alpha=0.3)
    
    #----------------------------------------------
    plt.plot(d[:11], drlqc_0_001_avgT, color='dimgrey', label='DRLQC (1e-3)')
    plt.fill_between(d[:11], drlqc_0_001_avgT + 2*drlqc_0_001_stdT, drlqc_0_001_avgT - 2*drlqc_0_001_stdT, facecolor='grey', alpha=0.3)
    
    #----------------------------------------------
    plt.plot(d[:12], drlqc_0_01_avgT, color='dimgrey', label='DRLQC (1e-2)',linestyle='--')
    plt.fill_between(d[:12], drlqc_0_01_avgT + 2*drlqc_0_01_stdT, drlqc_0_01_avgT - 2*drlqc_0_01_stdT, facecolor='grey', alpha=0.3)

    
    plt.plot(d, drce_avgT,  color='tab:green', label='WDR-CE')
    plt.fill_between(d, drce_avgT + 2*drce_stdT, drce_avgT - 2*drce_stdT, facecolor='tab:green', alpha=0.3)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Dimension', fontsize=16)
    plt.ylabel(r'Computation Time (s)', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, which="both",ls="-", alpha=0.3)
    plt.xlim(left=2,right=d[-1])
    #plt.ylim(bottom=2) # For visibility
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(path +'/Computation_Space_Time_{}_{}.pdf'.format(dist, noise_dist), dpi=300, bbox_inches="tight")
    plt.clf()
    print("Time plot generated!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform)
    parser.add_argument('--theta', required=False, default="0.1")
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    args = parser.parse_args()
    
    
    print('\n-------Summary-------')
    if args.infinite:
        path = "./results/{}_{}/infinite/multiple/DRLQC/".format(args.dist, args.noise_dist)
    else:
        path = "./results/{}_{}/finite/multiple/DRLQC".format(args.dist, args.noise_dist)
    
    
    
    avg_time_drce_file = open(path + '/drce_space_avgT.pkl', 'rb' )
    avg_time_drce = pickle.load(avg_time_drce_file)
    avg_time_drce_file.close()
    std_time_drce_file = open(path + '/drce_space_stdT.pkl', 'rb' )
    std_time_drce = pickle.load(std_time_drce_file)
    std_time_drce_file.close()
    
    avg_time_drlqc_0_0001_file = open(path + '/drlqc_space_0_0001_avgT.pkl', 'rb' )
    avg_time_drlqc_0_0001 = pickle.load(avg_time_drlqc_0_0001_file)
    avg_time_drlqc_0_0001_file.close()
    std_time_drlqc_0_0001_file = open(path + '/drlqc_space_0_0001_stdT.pkl', 'rb' )
    std_time_drlqc_0_0001 = pickle.load(std_time_drlqc_0_0001_file)
    std_time_drlqc_0_0001_file.close()
    
    avg_time_drlqc_0_001_file = open(path + '/drlqc_space_0_001_avgT.pkl', 'rb' )
    avg_time_drlqc_0_001 = pickle.load(avg_time_drlqc_0_001_file)
    avg_time_drlqc_0_001_file.close()
    std_time_drlqc_0_001_file = open(path + '/drlqc_space_0_001_stdT.pkl', 'rb' )
    std_time_drlqc_0_001 = pickle.load(std_time_drlqc_0_001_file)
    std_time_drlqc_0_001_file.close()
    
    avg_time_drlqc_0_01_file = open(path + '/drlqc_space_0_01_avgT.pkl', 'rb' )
    avg_time_drlqc_0_01 = pickle.load(avg_time_drlqc_0_01_file)
    avg_time_drlqc_0_01_file.close()
    std_time_drlqc_0_01_file = open(path + '/drlqc_space_0_01_stdT.pkl', 'rb' )
    std_time_drlqc_0_01 = pickle.load(std_time_drlqc_0_01_file)
    std_time_drlqc_0_01_file.close()
    
    summarize_noise(avg_time_drlqc_0_0001, std_time_drlqc_0_0001,avg_time_drlqc_0_001, std_time_drlqc_0_001,avg_time_drlqc_0_01, std_time_drlqc_0_01,  avg_time_drce, std_time_drce, args.dist, args.noise_dist, args.infinite ,path)
