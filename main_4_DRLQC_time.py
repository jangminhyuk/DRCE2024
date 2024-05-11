#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRCE import DRCE
from controllers.DRLQC import DRLQC
from controllers.DRLQC import DRLQC
from controllers.inf_LQG import inf_LQG
from controllers.inf_WDRC import inf_WDRC
from controllers.inf_DRCE import inf_DRCE

import os
import pickle
import control
import time

def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b-a)*np.random.rand(N,n)
    return x.T

def normal(mu, Sigma, N=1):
    x = np.random.multivariate_normal(mu[:,0], Sigma, size=N).T
    return x
def quad_inverse(x, b, a):
    row = x.shape[0]
    col = x.shape[1]
    for i in range(row):
        for j in range(col):
            beta = (a[j]+b[j])/2.0
            alpha = 12.0/((b[j]-a[j])**3)
            tmp = 3*x[i][j]/alpha - (beta - a[j])**3
            if 0<=tmp:
                x[i][j] = beta + ( tmp)**(1./3.)
            else:
                x[i][j] = beta -(-tmp)**(1./3.)
    return x

# quadratic U-shape distrubituon in [wmin , wmax]
def quadratic(wmax, wmin, N=1):
    n = wmin.shape[0]
    x = np.random.rand(N, n)
    #print("wmax : " , wmax)
    x = quad_inverse(x, wmax, wmin)
    return x.T

def multimodal(mu, Sigma, N=1):
    modes = 2
    n = mu[0].shape[0]
    x = np.zeros((n,N,modes))
    for i in range(modes):
        w = np.random.normal(size=(N,n))
        if (Sigma[i] == 0).all():
            x[:,:,i] = mu[i]
        else:
            x[:,:,i] = mu[i] + np.linalg.cholesky(Sigma[i]) @ w.T

    #w = np.random.choice([0, 1], size=(n,N))
    w = 0.5
    y = x[:,:,0]*w + x[:,:,1]*(1-w)
    return y

def gen_sample_dist(dist, T, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist=="normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist=="uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist=="multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)
    elif dist=="quadratic":
        w = quadratic(w_max, w_min, N=N_sample)

    mean_ = np.average(w, axis = 1)
    diff = (w.T - mean_)[...,np.newaxis]
    var_ = np.average( (diff @ np.transpose(diff, (0,2,1))) , axis = 0)
    return np.tile(mean_[...,np.newaxis], (T, 1, 1)), np.tile(var_, (T, 1, 1))

def gen_sample_dist_inf(dist, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist=="normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist=="uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist=="multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)
    elif dist=="quadratic":
        w = quadratic(w_max, w_min, N=N_sample)
        
    mean_ = np.average(w, axis = 1)[...,np.newaxis]
    var_ = np.cov(w)
    return mean_, var_


def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()

def main(dist, noise_dist1, num_sim, num_samples, num_noise_samples, T, plot_results, noise_plot_results, infinite):
    #noise_plot_results = True
    seed = 2024 # Random seed !  any value
    
    # --- for DRLQC --- #
    tol = 1e-3
    # --- ----- --------#
    
    if noise_plot_results:
        num_noise_list = [5, 10, 15, 20, 25, 30, 35, 40]
    else:
        num_noise_list = [num_noise_samples]
    num_x0_samples = 15 # num x0 samples 
    # for the noise_plot_results!!
    output_J_LQG_mean, output_J_WDRC_mean, output_J_DRCE_mean, output_J_DRLQC_mean=[], [], [], []
    output_J_LQG_std, output_J_WDRC_std, output_J_DRCE_std, output_J_DRLQC_std=[], [], [], []
    #-------Initialization-------
    nx = 10 #state dimension
    nu = 10 #control input dimension
    ny = 10#output dimension
    temp = np.ones((nx, nx))
    A = 0.1*(np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2))
    B = Q = R = Qf = np.eye(10)
    C = np.eye(10)
    #C = np.hstack([np.eye(9), np.zeros((9,1))])
    #----------------------------
    # change True to False if you don't want to use given lambda
    use_lambda = True
    lambda_ = 15 # will not be used if the parameter "use_lambda = False"
    noisedist = [noise_dist1]
    #noisedist = ["normal", "uniform", "quadratic"]
    #theta_v_list  # radius of noise ambiguity set
    #theta_w_list  # theta_w have no effect if the parameter "use_lambda = True"
    if dist == "normal":
        theta_w_list = [2.0]
        theta_v_list = [2.0]
        theta_x0 = 2.0 # radius of initial state ambiguity set
    elif dist == "quadratic":
        theta_w_list = [1.0]
        theta_v_list = [2.0]
        theta_x0 = 2.0
    else:
        theta_w_list = [2.0]
        theta_v_list = [5.0]
        theta_x0 = 5.0
    
    horizon_list = [10,20,30,40,50,60,70,80,90,100,200,300,400]
    
    # Save offline computation time for each method
    lqg_time_avg = []
    wdrc_time_avg = []
    drce_time_avg = []
    drlqc_time_avg = []
    lqg_time_std = []
    wdrc_time_std = []
    drce_time_std = []
    drlqc_time_std = []
    for noise_dist in noisedist:
        for T in horizon_list:
            lqg_time_list, wdrc_time_list, drce_time_list, drlqc_time_list=[], [], [], []
            for idx in range(10):
                num_noise = num_noise_list[0]
                theta_w = theta_w_list[0]
                theta = theta_v_list[0]
                print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise, "/ theta_w : ", theta_w, "/ theta_v : ", theta, "/ Horizon : ", T)
                np.random.seed(seed) # fix Random seed!
                print("--------------------------------------------")
                print("number of noise sample : ", num_noise)
                print("number of disturbance sample : ", num_samples)
                if infinite:
                    path = "./results/{}_{}/infinite/multiple/DRLQC".format(dist, noise_dist)
                else:
                    path = "./results/{}_{}/finite/multiple/DRLQC".format(dist, noise_dist)    
                if not os.path.exists(path):
                    os.makedirs(path)
            
                #-------Disturbance Distribution-------
                if dist == "normal":
                    #disturbance distribution parameters
                    w_max = None
                    w_min = None
                    mu_w = 0.0*np.ones((nx, 1))
                    Sigma_w= 0.1*np.eye(nx)
                    #initial state distribution parameters
                    x0_max = None
                    x0_min = None
                    x0_mean = 0.0*np.ones((nx,1))
                    x0_cov = 0.1*np.eye(nx)
                elif dist == "quadratic":
                    #disturbance distribution parameters
                    w_max = 1.0*np.ones(nx)
                    w_min = -1.0*np.ones(nx)
                    mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                    Sigma_w = 3.0/20.0*np.diag((w_max - w_min)**2)
                    #initial state distribution parameters
                    x0_max = 0.5*np.ones(nx)
                    x0_min = -0.5*np.ones(nx)
                    x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                    x0_cov = 3.0/20.0 *np.diag((x0_max - x0_min)**2)
                elif dist =="uniform":
                    #disturbance distribution parameters
                    w_max = 0.3*np.ones(nx)
                    w_min = -0.2*np.ones(nx)
                    mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                    Sigma_w = 1/12*np.diag((w_max - w_min)**2)
                    #initial state distribution parameters
                    x0_max = 0.1*np.ones(nx)
                    x0_min = -0.1*np.ones(nx)
                    x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                    x0_cov = 1/12*np.diag((x0_max - x0_min)**2)
                    
                #-------Noise distribution ---------#
                if noise_dist =="normal":
                    v_max = None
                    v_min = None
                    M = 1.5*np.eye(ny) #observation noise covariance
                    mu_v = 0.0*np.ones((ny, 1))
                elif noise_dist =="quadratic":
                    v_min = -1.0*np.ones(ny)
                    v_max = 1.0*np.ones(ny)
                    mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                    M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
                elif noise_dist == "uniform":
                    v_min = -1.0*np.ones(ny)
                    v_max = 2.0*np.ones(ny)
                    mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                    M = 1/12*np.diag((v_max - v_min)**2) #observation noise covariance
                    
                    
                #-------Estimate the nominal distribution-------
                if infinite:
                    # Nominal initial state distribution
                    x0_mean_hat, x0_cov_hat = gen_sample_dist_inf(dist, num_x0_samples, mu_w=x0_mean, Sigma_w=x0_cov, w_max=x0_max, w_min=x0_min)
                    # Nominal Disturbance distribution
                    mu_hat, Sigma_hat = gen_sample_dist_inf(dist, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                    # Nominal Noise distribution
                    v_mean_hat, M_hat = gen_sample_dist_inf(noise_dist, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)
                else:
                    # Nominal initial state distribution
                    x0_mean_hat, x0_cov_hat = gen_sample_dist(dist, 1, num_x0_samples, mu_w=x0_mean, Sigma_w=x0_cov, w_max=x0_max, w_min=x0_min)
                    # Nominal Disturbance distribution
                    mu_hat, Sigma_hat = gen_sample_dist(dist, T+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                    # Nominal Noise distribution
                    v_mean_hat, M_hat = gen_sample_dist(noise_dist, T+1, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)
                
                M_hat = M_hat + 1e-8*np.eye(ny) # to prevent numerical error from inverse in standard KF at small sample size
                Sigma_hat = Sigma_hat + 1e-8*np.eye(nx)
                x0_cov_hat = x0_cov_hat + 1e-8*np.eye(nx) 
                # for DRLQC-------------------
                W_hat = np.zeros((nx, nx, T+1))
                V_hat = np.zeros((ny, ny, T+1))
                for i in range(T):
                    W_hat[:,:,i] = Sigma_hat[i]
                    V_hat[:,:,i] = M_hat[i]
                # ----------------------------
                print(V_hat.shape)
                
                #-------Create a random system-------
                system_data = (A, B, C, Q, Qf, R, M)
                
                #-------Perform n  independent simulations and summarize the results-------
                
                
                #Initialize controllers
                if infinite:
                    drce = inf_DRCE(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, x0_mean_hat, x0_cov_hat, use_lambda)
                    lqg = inf_LQG(T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat , x0_mean_hat, x0_cov_hat)
                    wdrc = inf_WDRC(lambda_, theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat, use_lambda)
                    
                else:
                    wdrc = WDRC(lambda_, theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda)
                    drce = DRCE(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda)
                    drlqc = DRLQC(theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, W_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, V_hat, x0_mean_hat[0], x0_cov_hat[0], tol)
                    lqg = LQG(T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat , x0_mean_hat[0], x0_cov_hat[0])
                
                # --- DRLQC --- #
                
                if T<=50: # Since DRLQC took over 100s for this system when horizon T > 50, we will not plot it.
                    start = time.time()
                    drlqc.solve_sdp()
                    drlqc.backward()
                    end = time.time()
                    print("DRLQC Offline Computation time : ", end-start)
                    drlqc_time_list.append(end-start)
                
                # --- DRCE --- #
                start = time.time()
                drce.backward()
                end = time.time()
                print("DRCE Offline Computation time : ", end-start)
                drce_time_list.append(end-start)
                
                # --- WDRC --- #
                start = time.time()
                wdrc.backward()
                end = time.time()
                print("WDRC Offline Computation time : ", end-start)
                wdrc_time_list.append(end-start)
                    
                
            if infinite:
                path = "./results/{}_{}/infinite/multiple/DRLQC/".format(dist, noise_dist)
            else:
                path = "./results/{}_{}/finite/multiple/DRLQC/".format(dist, noise_dist)
            if not os.path.exists(path):
                os.makedirs(path)
            
            wdrc_time_avg.append(np.mean(wdrc_time_list))
            drce_time_avg.append(np.mean(drce_time_list))
            
            wdrc_time_std.append(np.std(wdrc_time_list))
            drce_time_std.append(np.std(drce_time_list))
            if T<=50:
                drlqc_time_std.append(np.std(drlqc_time_list))
                drlqc_time_avg.append(np.mean(drlqc_time_list))
            
            print("Average offline computation time: For Time horizon 10 ~ ",T)
            print("WDRC (avg): ", wdrc_time_avg)
            print("DRCE (avg): ", drce_time_avg)
            print("DRLQC (avg): ", drlqc_time_avg)
            print("Standard Deviation")
            print("WDRC (std): ", wdrc_time_std)
            print("DRCE (std): ", drce_time_std)
            print("DRLQC (std): ", drlqc_time_std)
            
        if infinite:
            path = "./results/{}_{}/infinite/multiple/DRLQC/".format(dist, noise_dist)
        else:
            path = "./results/{}_{}/finite/multiple/DRLQC/".format(dist, noise_dist)
        if not os.path.exists(path):
            os.makedirs(path)    
        save_data(path + 'drlqc_avgT.pkl', drlqc_time_avg)
        save_data(path + 'drce_avgT.pkl', drce_time_avg)
        save_data(path + 'wdrc_avgT.pkl', wdrc_time_avg)
        
        save_data(path + 'drlqc_stdT.pkl', drlqc_time_std)
        save_data(path + 'drce_stdT.pkl', drce_time_std)
        save_data(path + 'wdrc_stdT.pkl', wdrc_time_std)
            
        print('\n-------Summary-------')
        print("dist : ", dist,"/ noise dist : ", noise_dist, "/ num_samples : ", num_samples, "/ num_noise_samples : ", num_noise, "/seed : ", seed)
                    
                    
            # after running 10 repeated calcuatoin for this time T
            
                    
    print("Data generation Completed!!")
    
    if noise_plot_results:
        if infinite:
            print("For noise sample size effect plot : Use python plot_J4_dlrqc.py --infinite --dist "+ dist + " --noise_dist " + noise_dist)
        else:
            print("For noise sample size effect plot : Use python plot_J4_drlqc.py --dist "+ dist + " --noise_dist " + noise_dist)
    else:
        if infinite:
            print("For plot : Use python plot4_drlqc.py --infinite --dist "+ dist + " --noise_dist " + noise_dist)
        else:
            print("For plot : Use python plot4_drlqc.py --dist "+ dist + " --noise_dist " + noise_dist)
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--num_sim', required=False, default=5, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=15, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=15, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=20, type=int) #horizon length
    parser.add_argument('--plot', required=False, action="store_true") #plot results+
    parser.add_argument('--noise_plot', required=False, action="store_true") # noise sample size plot
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon, args.plot, args.noise_plot, args.infinite)
