#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize(out_lq_list, out_wdrc_list, out_drce_list, out_drcmmse_list, dist, noise_dist, path, num, infinite, plot_results=True):
    x_lqr_list, J_lqr_list, y_lqr_list, u_lqr_list = [], [], [], []
    x_wdrc_list, J_wdrc_list, y_wdrc_list, u_wdrc_list = [], [], [], [] # original wdrc with ordinary Kalman Filter
    x_drce_list, J_drce_list, y_drce_list, u_drce_list = [], [], [], [] # drce
    x_drcmmse_list, J_drcmmse_list, y_drcmmse_list, u_drcmmse_list = [], [], [], [] # drce
    time_wdrc_list, time_lqr_list, time_drce_list, time_drcmmse_list = [], [], [], []


    for out in out_lq_list:
         x_lqr_list.append(out['state_traj'])
         J_lqr_list.append(out['cost'])
         y_lqr_list.append(out['output_traj'])
         u_lqr_list.append(out['control_traj'])
         time_lqr_list.append(out['comp_time'])
         
    x_lqr_mean, J_lqr_mean, y_lqr_mean, u_lqr_mean = np.mean(x_lqr_list, axis=0), np.mean(J_lqr_list, axis=0), np.mean(y_lqr_list, axis=0), np.mean(u_lqr_list, axis=0)
    x_lqr_std, J_lqr_std, y_lqr_std, u_lqr_std = np.std(x_lqr_list, axis=0), np.std(J_lqr_list, axis=0), np.std(y_lqr_list, axis=0), np.std(u_lqr_list, axis=0)
    time_lqr_ar = np.array(time_lqr_list)
    print("LQG cost : ", J_lqr_mean[0])
    print("LQG cost std : ", J_lqr_std[0])
    J_lqr_ar = np.array(J_lqr_list)
    
    
    for out in out_wdrc_list:
        x_wdrc_list.append(out['state_traj'])
        J_wdrc_list.append(out['cost'])
        y_wdrc_list.append(out['output_traj'])
        u_wdrc_list.append(out['control_traj'])
        time_wdrc_list.append(out['comp_time'])
    x_wdrc_mean, J_wdrc_mean, y_wdrc_mean, u_wdrc_mean = np.mean(x_wdrc_list, axis=0), np.mean(J_wdrc_list, axis=0), np.mean(y_wdrc_list, axis=0), np.mean(u_wdrc_list, axis=0)
    x_wdrc_std, J_wdrc_std, y_wdrc_std, u_wdrc_std = np.std(x_wdrc_list, axis=0), np.std(J_wdrc_list, axis=0), np.std(y_wdrc_list, axis=0), np.std(u_wdrc_list, axis=0)
    time_wdrc_ar = np.array(time_wdrc_list)
    print("WDRC cost : ", J_wdrc_mean[0])
    print("WDRC cost std : ", J_wdrc_std[0])
    J_wdrc_ar = np.array(J_wdrc_list)



    for out in out_drce_list:
            x_drce_list.append(out['state_traj'])
            J_drce_list.append(out['cost'])
            y_drce_list.append(out['output_traj'])
            u_drce_list.append(out['control_traj'])
            time_drce_list.append(out['comp_time'])
    x_drce_mean, J_drce_mean, y_drce_mean, u_drce_mean = np.mean(x_drce_list, axis=0), np.mean(J_drce_list, axis=0), np.mean(y_drce_list, axis=0), np.mean(u_drce_list, axis=0)
    x_drce_std, J_drce_std, y_drce_std, u_drce_std = np.std(x_drce_list, axis=0), np.std(J_drce_list, axis=0), np.std(y_drce_list, axis=0), np.std(u_drce_list, axis=0)
    time_drce_ar = np.array(time_drce_list)
    print("DRCE cost : ", J_drce_mean[0])
    print("DRCE cost std : ", J_drce_std[0])
    J_drce_ar = np.array(J_drce_list)   
    nx = x_drce_mean.shape[1]
    T = u_drce_mean.shape[0]
    
    for out in out_drcmmse_list:
            x_drcmmse_list.append(out['state_traj'])
            J_drcmmse_list.append(out['cost'])
            y_drcmmse_list.append(out['output_traj'])
            u_drcmmse_list.append(out['control_traj'])
            time_drcmmse_list.append(out['comp_time'])
    x_drcmmse_mean, J_drcmmse_mean, y_drcmmse_mean, u_drcmmse_mean = np.mean(x_drcmmse_list, axis=0), np.mean(J_drcmmse_list, axis=0), np.mean(y_drcmmse_list, axis=0), np.mean(u_drcmmse_list, axis=0)
    x_drcmmse_std, J_drcmmse_std, y_drcmmse_std, u_drcmmse_std = np.std(x_drcmmse_list, axis=0), np.std(J_drcmmse_list, axis=0), np.std(y_drcmmse_list, axis=0), np.std(u_drcmmse_list, axis=0)
    time_drcmmse_ar = np.array(time_drcmmse_list)
    print("DRCMMSE cost : ", J_drcmmse_mean[0])
    print("DRCMMSE cost std : ", J_drcmmse_std[0])
    J_drcmmse_ar = np.array(J_drcmmse_list)   
    nx = x_drcmmse_mean.shape[1]
    T = u_drcmmse_mean.shape[0]
    
    
    
    # ------------------------------------------------------------
    if plot_results:
        nx = x_drce_mean.shape[1]
        T = u_drce_mean.shape[0]
        nu = u_drce_mean.shape[1]
        ny= y_drce_mean.shape[1]

        fig = plt.figure(figsize=(6,4), dpi=300)

        t = np.arange(T+1)
        for i in range(nx):

            if x_lqr_list != []:
                plt.plot(t, x_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, x_lqr_mean[:,i, 0] + 0.3*x_lqr_std[:,i,0],
                               x_lqr_mean[:,i,0] - 0.3*x_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            if x_wdrc_list != []:
                plt.plot(t, x_wdrc_mean[:,i,0], 'tab:blue', label='WDRC')
                plt.fill_between(t, x_wdrc_mean[:,i,0] + 0.3*x_wdrc_std[:,i,0],
                                x_wdrc_mean[:,i,0] - 0.3*x_wdrc_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            if x_drcmmse_list != []:
                plt.plot(t, x_drcmmse_mean[:,i,0], 'tab:purple', label='WDRC-MMSE')
                plt.fill_between(t, x_drcmmse_mean[:,i, 0] + 0.3*x_drcmmse_std[:,i,0],
                               x_drcmmse_mean[:,i,0] - 0.3*x_drcmmse_std[:,i,0], facecolor='tab:purple', alpha=0.3)
            if x_drce_list != []:
                plt.plot(t, x_drce_mean[:,i,0], 'tab:green', label='WDR-CE')
                plt.fill_between(t, x_drce_mean[:,i, 0] + 0.3*x_drce_std[:,i,0],
                               x_drce_mean[:,i,0] - 0.3*x_drce_std[:,i,0], facecolor='tab:green', alpha=0.3)
            
                
            plt.xlabel(r'$t$', fontsize=22)
            plt.ylabel(r'$x_{{{}}}$'.format(i+1), fontsize=22)
            plt.legend(fontsize=20)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'states_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist), dpi=300, bbox_inches="tight")
            plt.clf()

        t = np.arange(T)
        for i in range(nu):

            if u_lqr_list != []:
                plt.plot(t, u_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, u_lqr_mean[:,i,0] + 0.25*u_lqr_std[:,i,0],
                             u_lqr_mean[:,i,0] - 0.25*u_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            if u_wdrc_list != []:
                plt.plot(t, u_wdrc_mean[:,i,0], 'tab:blue', label='WDRC')
                plt.fill_between(t, u_wdrc_mean[:,i,0] + 0.25*u_wdrc_std[:,i,0],
                                u_wdrc_mean[:,i,0] - 0.25*u_wdrc_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            if u_drcmmse_list != []:
                plt.plot(t, u_drcmmse_mean[:,i,0], 'tab:purple', label='WDRC-MMSE')
                plt.fill_between(t, u_drcmmse_mean[:,i,0] + 0.25*u_drcmmse_std[:,i,0],
                             u_drcmmse_mean[:,i,0] - 0.25*u_drcmmse_std[:,i,0], facecolor='tab:purple', alpha=0.3) 
            if u_drce_list != []:
                plt.plot(t, u_drce_mean[:,i,0], 'tab:green', label='WDR-CE')
                plt.fill_between(t, u_drce_mean[:,i,0] + 0.25*u_drce_std[:,i,0],
                             u_drce_mean[:,i,0] - 0.25*u_drce_std[:,i,0], facecolor='tab:green', alpha=0.3)       
            
            plt.xlabel(r'$t$', fontsize=16)
            plt.ylabel(r'$u_{{{}}}$'.format(i+1), fontsize=16)
            plt.legend(fontsize=16)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'controls_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist), dpi=300, bbox_inches="tight")
            plt.clf()

        t = np.arange(T+1)
        for i in range(ny):
            if y_lqr_list != []:
                plt.plot(t, y_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, y_lqr_mean[:,i,0] + 0.25*y_lqr_std[:,i,0],
                             y_lqr_mean[:,i, 0] - 0.25*y_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            if y_wdrc_list != []:
                plt.plot(t, y_wdrc_mean[:,i,0], 'tab:blue', label='WDRC')
                plt.fill_between(t, y_wdrc_mean[:,i,0] + 0.25*y_wdrc_std[:,i,0],
                                y_wdrc_mean[:,i, 0] - 0.25*y_wdrc_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            if y_drcmmse_list != []:
                plt.plot(t, y_drcmmse_mean[:,i,0], 'tab:purple', label='WDRC-MMSE')
                plt.fill_between(t, y_drcmmse_mean[:,i,0] + 0.25*y_drcmmse_std[:,i,0],
                             y_drcmmse_mean[:,i, 0] - 0.25*y_drcmmse_std[:,i,0], facecolor='tab:purple', alpha=0.3)
            if y_drce_list != []:
                plt.plot(t, y_drce_mean[:,i,0], 'tab:green', label='WDR-CE')
                plt.fill_between(t, y_drce_mean[:,i,0] + 0.25*y_drce_std[:,i,0],
                             y_drce_mean[:,i, 0] - 0.25*y_drce_std[:,i,0], facecolor='tab:green', alpha=0.3)
            
            plt.xlabel(r'$t$', fontsize=16)
            plt.ylabel(r'$y_{{{}}}$'.format(i+1), fontsize=16)
            plt.legend(fontsize=16)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'outputs_{}_{}_{}_{}.pdf'.format(i+1,num, dist, noise_dist), dpi=300, bbox_inches="tight")
            plt.clf()


        plt.title('Optimal Value')
        t = np.arange(T+1)

        if J_lqr_list != []:
            plt.plot(t, J_lqr_mean, 'tab:red', label='LQG')
            plt.fill_between(t, J_lqr_mean + 0.25*J_lqr_std, J_lqr_mean - 0.25*J_lqr_std, facecolor='tab:red', alpha=0.3)
        if J_wdrc_list != []:
            plt.plot(t, J_wdrc_mean, 'tab:blue', label='WDRC')
            plt.fill_between(t, J_wdrc_mean + 0.25*J_wdrc_std, J_wdrc_mean - 0.25*J_wdrc_std, facecolor='tab:blue', alpha=0.3)
        if J_drcmmse_list != []:
            plt.plot(t, J_drcmmse_mean, 'tab:purple', label='WDRC-MMSE')
            plt.fill_between(t, J_drcmmse_mean + 0.25*J_drcmmse_std, J_drcmmse_mean - 0.25*J_drcmmse_std, facecolor='tab:purple', alpha=0.3)
        if J_drce_list != []:
            plt.plot(t, J_drce_mean, 'tab:green', label='WDR-CE')
            plt.fill_between(t, J_drce_mean + 0.25*J_drce_std, J_drce_mean - 0.25*J_drce_std, facecolor='tab:green', alpha=0.3)
        
        plt.xlabel(r'$t$', fontsize=16)
        plt.ylabel(r'$V_t(x_t)$', fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.xlim([t[0], t[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(path +'J_{}_{}_{}.pdf'.format(num, dist, noise_dist), dpi=300, bbox_inches="tight")
        plt.clf()


        ax = fig.gca()
        t = np.arange(T+1)
        
        max_bin = np.max([J_wdrc_ar[:,0], J_lqr_ar[:,0], J_drce_ar[:,0], J_drcmmse_ar[:,0]])
        min_bin = np.min([J_wdrc_ar[:,0], J_lqr_ar[:,0], J_drce_ar[:,0], J_drcmmse_ar[:,0]])


        
        ax.hist(J_lqr_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:red', label='LQG', alpha=0.5, linewidth=0.5, edgecolor='tab:red')
        ax.hist(J_wdrc_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:blue', label='WDRC', alpha=0.5, linewidth=0.5, edgecolor='tab:blue')
        ax.hist(J_drcmmse_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:purple', label='WDRC-MMSE', alpha=0.5, linewidth=0.5, edgecolor='tab:purple')
        ax.hist(J_drce_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:green', label='WDR-CE', alpha=0.5, linewidth=0.5, edgecolor='tab:green')
        
        ax.axvline(J_wdrc_ar[:,0].mean(), color='navy', linestyle='dashed', linewidth=1.5)
        ax.axvline(J_lqr_ar[:,0].mean(), color='maroon', linestyle='dashed', linewidth=1.5)
        ax.axvline(J_drcmmse_ar[:,0].mean(), color='purple', linestyle='dashed', linewidth=1.5)
        ax.axvline(J_drce_ar[:,0].mean(), color='green', linestyle='dashed', linewidth=1.5)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        handles, labels = plt.gca().get_legend_handles_labels()
        
        order = [0, 1, 2, 3]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)

        ax.grid()
        ax.set_axisbelow(True)
        #plt.title('{} system disturbance, {} observation noise'.format(dist, noise_dist))
        plt.xlabel(r'Total Cost', fontsize=16)
        plt.ylabel(r'Frequency', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(path +'J_hist_{}_{}_{}.pdf'.format(num, dist, noise_dist), dpi=300, bbox_inches="tight")
        plt.clf()


        plt.close('all')
        
        
    print( 'cost_lqr:{} ({})'.format(J_lqr_mean[0],J_lqr_std[0]),'cost_WDRC: {} ({})'.format(J_wdrc_mean[0], J_wdrc_std[0]) , 'cost_wdrce:{} ({})'.format(J_drce_mean[0],J_drce_std[0]), 'cost_wdrcmmse:{} ({})'.format(J_drcmmse_mean[0],J_drcmmse_std[0]))
    print( 'time_lqr: {} ({})'.format(time_lqr_ar.mean(), time_lqr_ar.std()),'time_WDRC: {} ({})'.format(time_wdrc_ar.mean(), time_wdrc_ar.std()), 'time_wdrce: {} ({})'.format(time_drce_ar.mean(), time_drce_ar.std()), 'time_wdrcmmse: {} ({})'.format(time_drcmmse_ar.mean(), time_drcmmse_ar.std()))
    #print( 'Settling time_lqr: {}'.format(SettlingTime_lqr),'Settling time_WDRC: {} '.format(SettlingTime), 'Settling time_drce: {}'.format(SettlingTime_drce))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quad)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quad)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs to plot
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged

    args = parser.parse_args()

    horizon = "finite"
        
    print('\n-------Summary-------')
    if args.infinite:
        path = "./results/{}_{}/infinite/multiple/".format(args.dist, args.noise_dist)
    else:
        path = "./results/{}_{}/finite/multiple/".format(args.dist, args.noise_dist)
    
    #Load data
    lqg_file = open(path + 'lqg.pkl', 'rb')
    wdrc_file = open(path + 'wdrc.pkl', 'rb')
    drce_file = open(path + 'drce.pkl', 'rb')
    drcmmse_file = open(path + 'drcmmse.pkl', 'rb')
    
    lqg_data = pickle.load(lqg_file)
    wdrc_data = pickle.load(wdrc_file)
    drce_data = pickle.load(drce_file)
    drcmmse_data = pickle.load(drcmmse_file)
    
    lqg_file.close()
    wdrc_file.close()
    drce_file.close()
    drcmmse_file.close()
    
    summarize(lqg_data, wdrc_data, drce_data, drcmmse_data, args.dist, args.noise_dist, path, args.num_sim, args.infinite , plot_results=True)
    

