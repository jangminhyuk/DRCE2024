#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize_noise( avg_time_drlqc, std_time_drlqc, avg_time_wdrc, std_time_wdrc, avg_time_drce, std_time_drce, dist, noise_dist, infinite, path):

    #horizon_list
    t = np.array([10,20,30,40,50,60,70,80,90,100,200,300,400])
    
    drlqc_avgT = np.array(avg_time_drlqc[0:])
    wdrc_avgT = np.array(avg_time_wdrc[0:])
    drce_avgT = np.array(avg_time_drce[0:])
    
    drlqc_stdT = np.array(std_time_drlqc[0:])
    wdrc_stdT = np.array(std_time_wdrc[0:])
    drce_stdT = np.array(std_time_drce[0:])
    
    
    fig= plt.figure(figsize=(6,4), dpi=300)
    
    #----------------------------------------------
    plt.plot(t[:5], drlqc_avgT, color='tab:purple', label='DRLQC')
    plt.fill_between(t[:5], drlqc_avgT + 2*drlqc_stdT, drlqc_avgT - 2*drlqc_stdT, facecolor='tab:purple', alpha=0.3)
    
    plt.plot(t, wdrc_avgT, color='tab:blue', label='WDRC')
    plt.fill_between(t, wdrc_avgT + 2*wdrc_stdT, wdrc_avgT - 2*wdrc_stdT, facecolor='tab:blue', alpha=0.3)
    
    plt.plot(t, drce_avgT,  color='tab:green', label='WDR-CE')
    plt.fill_between(t, drce_avgT + 2*drce_stdT, drce_avgT - 2*drce_stdT, facecolor='tab:green', alpha=0.3)
    
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'Time horizon', fontsize=16)
    plt.ylabel(r'Computation Time (s)', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, which="both",ls="-", alpha=0.3)
    plt.xlim(left=9,right=400)
    plt.ylim(bottom=1, top=drlqc_avgT[-1]*1.05) # For visibility
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(path +'/Computation_Time_{}_{}.pdf'.format(dist, noise_dist), dpi=300, bbox_inches="tight")
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
    
    
    
    avg_time_wdrc_file = open(path + '/wdrc_avgT.pkl', 'rb' )
    avg_time_wdrc = pickle.load(avg_time_wdrc_file)
    #print(avg_time_wdrc)
    avg_time_wdrc_file.close()
    std_time_wdrc_file = open(path + '/wdrc_stdT.pkl', 'rb' )
    std_time_wdrc = pickle.load(std_time_wdrc_file)
    std_time_wdrc_file.close()
    
    avg_time_drce_file = open(path + '/drce_avgT.pkl', 'rb' )
    avg_time_drce = pickle.load(avg_time_drce_file)
    #print(avg_time_drkf_wdrc)
    avg_time_drce_file.close()
    std_time_drce_file = open(path + '/drce_stdT.pkl', 'rb' )
    std_time_drce = pickle.load(std_time_drce_file)
    std_time_drce_file.close()
    
    avg_time_drlqc_file = open(path + '/drlqc_avgT.pkl', 'rb' )
    avg_time_drlqc = pickle.load(avg_time_drlqc_file)
    #print(avg_time_drkf_wdrc)
    avg_time_drlqc_file.close()
    std_time_drlqc_file = open(path + '/drlqc_stdT.pkl', 'rb' )
    std_time_drlqc = pickle.load(std_time_drlqc_file)
    std_time_drlqc_file.close()
    
    summarize_noise(avg_time_drlqc, std_time_drlqc, avg_time_wdrc, std_time_wdrc, avg_time_drce, std_time_drce, args.dist, args.noise_dist, args.infinite ,path)
