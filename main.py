#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRCE import DRCE

from plot_params import summarize
from plot_J import summarize_noise

import os
import pickle

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
#    var_ = np.diag(np.diag(var_))
    return mean_, var_

# def create_matrices(nx, ny, nu):
#     A = np.load("./inputs/A.npy") # (n x n) matrix
#     B = np.load("./inputs/B.npy")
#     C = np.hstack([np.eye(ny, int(ny/2)), np.zeros((ny, int((nx-ny)/2))), np.eye(ny, int(ny/2), k=-int(ny/2)), np.zeros((ny, int((nx-ny)/2)))])
# #    C = np.hstack([np.zeros((ny, nx-ny)), np.eye(ny, ny)])
# #    C = np.eye(ny)

#     return A, B, C

def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()

def main(dist, noise_dist1, num_sim, num_samples, num_noise_samples, T):
    noise_plot_results = True
    lambda_ = 10
    seed = 2024 # any value
    if noise_plot_results: # if you need to draw plot_J
        num_noise_list = [5, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40]
    else:
        num_noise_list = [num_noise_samples]
    
    # for the noise_plot_results!!
    output_J_LQG_mean, output_J_WDRC_mean, output_J_DRCE_mean=[], [], []
    output_J_LQG_std, output_J_WDRC_std, output_J_DRCE_std=[], [], []
    #-------Initialization-------
    nx = 10 #state dimension
    nu = 10 #control input dimension
    ny = 10#output dimension
    temp = np.ones((nx, nx))
    A = np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2)
    B = C = Q = R = np.eye(10) 
    Qf = np.zeros((10,10))
    #----------------------------
    # HERE!! change 1 to 0 if you don't want to use given lambda
    use_lambda = 0
    
    #theta_v_list = [1.0, 1.2, 1.4, 1.6, 2.0]
    theta_v_list = [2.0]
    #theta_w_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    theta_w_list = [2.0] # theta_w have no effect if use set use_lambda = 1
    noisedist = [noise_dist1]
    theta_x0 = 0.5 # radius of initial state ambiguity set
    
    for noise_dist in noisedist:
        for theta_w in theta_w_list:
            for theta in theta_v_list:
                for num_noise in num_noise_list:
                    print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise, "/ theta_w : ", theta_w, "/ theta_v : ", theta)
                    np.random.seed(seed) # fix Random seed!
                    print("--------------------------------------------")
                    print("number of noise sample : ", num_noise)
                    print("number of disturbance sample : ", num_samples)
                    path = "./results/{}_{}/finite/multiple/".format(dist, noise_dist)
                    if not os.path.exists(path):
                        os.makedirs(path)
                
                    #-------Initialization-------
                    if dist =="uniform":
                        #disturbance distribution parameters
                        w_max = 0.3*np.ones(nx)
                        w_min = -0.3*np.ones(nx)
                        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                        Sigma_w = 1/12*np.diag((w_max - w_min)**2)
                        #initial state distribution parameters
                        x0_max = 0.05*np.ones(nx)
                        x0_min = -0.05*np.ones(nx)
                        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                        x0_cov = 1/12*np.diag((x0_max - x0_min)**2)
                        
                    elif dist == "normal":
                        
                        w_max = None
                        w_min = None

                        mu_w = 0.1*np.ones((nx, 1))
                        Sigma_w= 0.1*np.eye(nx)
                        #initial state distribution parameters
                        x0_max = None
                        x0_min = None
                        x0_mean = np.zeros((nx,1))
                        x0_cov = 0.01*np.eye(nx)
                    elif dist == "quadratic":
                        w_max = 0.2*np.ones(nx)
                        w_min = -0.1*np.ones(nx)
                        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                        Sigma_w = 3.0/20.0*np.diag((w_max - w_min)**2)
                        #initial state distribution parameters
                        x0_max = 0.1*np.ones(nx)
                        x0_min = -0.1*np.ones(nx)
                        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                        x0_cov = 3.0/20.0 *np.diag((x0_max - x0_min)**2)
                        
                    #-------Noise distribution ---------#
                    if noise_dist == "uniform":
                        v_min = -0.5*np.ones(ny)
                        v_max = 0.5*np.ones(ny)
                        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                        M = 1/12*np.diag((v_max - v_min)**2)
                    elif noise_dist =="normal":
                        v_max = None
                        v_min = None
                        M = 0.3*np.eye(ny) #observation noise covariance
                        mu_v = 0.3*np.zeros((ny, 1))
                    elif noise_dist =="quadratic":
                        v_min = 0.0*np.ones(ny)
                        v_max = 1.0*np.ones(ny)
                        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                        M = 3.0/20.0 *np.diag((v_max-v_min)**2)
                        
                        
                    #-------Estimate the nominal distribution-------
                    # Nominal Disturbance distribution
                    mu_hat, Sigma_hat = gen_sample_dist(dist, T+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                    
                    # Nominal Noise distribution
                    v_mean_hat, M_hat = gen_sample_dist(noise_dist, T+1, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)
                    
                    M_hat = M_hat + 1e-6*np.eye(ny) # to prevent numerical error
                    
                    #-------Create a random system-------
                    system_data = (A, B, C, Q, Qf, R, M)
                    
                    #-------Perform n  independent simulations and summarize the results-------
                    output_lqg_list = []
                    output_wdrc_list = []
                    output_drce_list = []
                    #Initialize controllers
                    
                    
                    drce = DRCE(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, use_lambda)
                    wdrc = WDRC(lambda_, theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, use_lambda)
                    lqg = LQG(T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat)
                
                    wdrc.backward()
                    drce.backward()
                    lqg.backward()
                        
                    print('---------------------')
                    
                    #----------------------------
                    print("Running DRCE Forward step ...")
                    for i in range(num_sim):
                        
                        #Perform state estimation and apply the controller
                        output_drce = drce.forward()
                        output_drce_list.append(output_drce)
                    
                        print('cost (DRCE):', output_drce['cost'][0], 'time (DRCE):', output_drce['comp_time'])
                    
                    J_DRCE_list = []
                    for out in output_drce_list:
                        J_DRCE_list.append(out['cost'])
                    J_DRCE_mean= np.mean(J_DRCE_list, axis=0)
                    J_DRCE_std = np.std(J_DRCE_list, axis=0)
                    output_J_DRCE_mean.append(J_DRCE_mean[0])
                    output_J_DRCE_std.append(J_DRCE_std[0])
                    print(" Average cost (DRCE) : ", J_DRCE_mean[0])
                    print(" std (DRCE) : ", J_DRCE_std[0])
                    
                    #----------------------------             
                    np.random.seed(seed) # fix Random seed!
                    print("Running WDRC Forward step ...")  
                    for i in range(num_sim):
                
                        #Perform state estimation and apply the controller
                        output_wdrc = wdrc.forward()
                        output_wdrc_list.append(output_wdrc)
                        print('cost (WDRC):', output_wdrc['cost'][0], 'time (WDRC):', output_wdrc['comp_time'])
                    
                    J_WDRC_list = []
                    for out in output_wdrc_list:
                        J_WDRC_list.append(out['cost'])
                    J_WDRC_mean= np.mean(J_WDRC_list, axis=0)
                    J_WDRC_std = np.std(J_WDRC_list, axis=0)
                    output_J_WDRC_mean.append(J_WDRC_mean[0])
                    output_J_WDRC_std.append(J_WDRC_std[0])
                    print(" Average cost (WDRC) : ", J_WDRC_mean[0])
                    print(" std (WDRC) : ", J_WDRC_std[0])
                    #----------------------------
                    np.random.seed(seed) # fix Random seed!
                    print("Running LQG Forward step ...")
                    for i in range(num_sim):
                        output_lqg = lqg.forward()
                        output_lqg_list.append(output_lqg)
                
                        print('cost (LQG):', output_lqg['cost'][0], 'time (LQG):', output_lqg['comp_time'])
                        
                    J_LQG_list = []
                    for out in output_lqg_list:
                        J_LQG_list.append(out['cost'])
                    J_LQG_mean= np.mean(J_LQG_list, axis=0)
                    J_LQG_std = np.std(J_LQG_list, axis=0)
                    output_J_LQG_mean.append(J_LQG_mean[0])
                    output_J_LQG_std.append(J_LQG_std[0])
                    print(" Average cost (LQG) : ", J_LQG_mean[0])
                    print(" std (LQG) : ", J_LQG_std[0])
                    
                
                    if noise_plot_results:
                        J_LQG_list, J_WDRC_list, J_DRCE_list= [], [], []
                        
                        #lqg
                        for out in output_lqg_list:
                            J_LQG_list.append(out['cost'])
                            
                        J_LQG_mean= np.mean(J_LQG_list, axis=0)
                        J_LQG_std = np.std(J_LQG_list, axis=0)
                        
                        #wdrc
                        for out in output_wdrc_list:
                            J_WDRC_list.append(out['cost'])
                            
                        J_WDRC_mean= np.mean(J_WDRC_list, axis=0)
                        J_WDRC_std = np.std(J_WDRC_list, axis=0)

                        #drce
                        for out in output_drce_list:
                            J_DRCE_list.append(out['cost'])
                            
                        J_DRCE_mean= np.mean(J_DRCE_list, axis=0)
                        J_DRCE_std = np.std(J_DRCE_list, axis=0)
                        
                        print("num_noise_sample : ", num_noise, " / finished with dist : ", dist, "/ noise_dist : ", noise_dist, "/ seed : ", seed)
                    else:
                        
                        save_data(path + 'drce.pkl', J_DRCE_mean)
                        save_data(path + 'wdrc.pkl', J_WDRC_mean)
                        save_data(path + 'lqg.pkl', J_LQG_mean)
                
                        #Summarize and plot the results
                        print('\n-------Summary-------')
                        print("dist : ", dist,"/ noise dist : ", noise_dist, "/ num_samples : ", num_samples, "/ num_noise_samples : ", num_noise, "/seed : ", seed)
                        
                        
                
                # after running noise_samples lists!
                if noise_plot_results:
                    path = "./results/{}_{}/finite/multiple/num_noise_plot/".format(dist, noise_dist)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    save_data(path + 'drce_mean.pkl', output_J_DRCE_mean)
                    save_data(path + 'drce_std.pkl', output_J_DRCE_std)  
                    save_data(path + 'lqg_mean.pkl', output_J_LQG_mean)
                    save_data(path + 'lqg_std.pkl', output_J_LQG_std) 
                    save_data(path + 'wdrc_mean.pkl', output_J_WDRC_mean)
                    save_data(path + 'wdrc_std.pkl', output_J_WDRC_std) 
                    
                    #Summarize and plot the results
                    print('\n-------Summary-------')
                    print("dist : ", dist, "noise_dist : ", noise_dist, "/ num_disturbance_samples : ", num_samples, "/ theta_v : ", theta, " / noise sample effect PLOT / Seed : ",seed)
                    
    print("Data generation Completed!!")
    print("Use plot_J.py to draw noise_sample_size effect plot")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=10, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=10, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=20, type=int) #horizon length

    args = parser.parse_args()
    np.random.seed(100)
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon)
