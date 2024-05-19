#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRCE import DRCE
from controllers.inf_LQG import inf_LQG
from controllers.inf_WDRC import inf_WDRC
from controllers.inf_DRCE import inf_DRCE
from joblib import Parallel, delayed

import os
import pickle

# Define the function to be parallelized
def perform_simulation(i, lambda_, num_sim, use_lambda, WDRC_lambda, DRCE_lambda, idx, theta_w, theta_v, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat):
    print("iteration ", i, "/", num_sim)
    
    if use_lambda:  # If we use pre-calculated lambda
        lambda_ = WDRC_lambda[idx]
        
    wdrc_ = WDRC(lambda_, theta_w, T, dist, noise_dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat[i], M_hat[i], x0_mean_hat[i][0], x0_cov_hat[i][0], use_lambda)
    
    if use_lambda:  # If we use pre-calculated lambda
        lambda_ = DRCE_lambda[idx]
        
    drce_ = DRCE(lambda_, theta_w, theta_v, theta_x0, T, dist, noise_dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat[i],  M_hat[i], x0_mean_hat[i][0], x0_cov_hat[i][0], use_lambda)
    
    lqg_ = LQG(T, dist, noise_dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat[i], M_hat[i], x0_mean_hat[i][0], x0_cov_hat[i][0])
    
    drce_.backward()
    wdrc_.backward()
    lqg_.backward()
    
    result = {
        'wdrc': wdrc_,
        'drce': drce_,
        'lqg': lqg_,
        'WDRC_lambda': wdrc_.lambda_ if not use_lambda else None,
        'DRCE_lambda': drce_.lambda_ if not use_lambda else None
    }
    return result

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
    np.random.seed(seed) # fix Random seed!
    if noise_plot_results:
        num_noise_list = [5, 10, 15, 20, 25, 30, 35, 40]
    else:
        num_noise_list = [num_noise_samples]
    num_x0_samples = 15 # num x0 samples 
    # for the noise_plot_results!!
    output_J_LQG_mean, output_J_WDRC_mean, output_J_DRCE_mean=[], [], []
    output_J_LQG_std, output_J_WDRC_std, output_J_DRCE_std=[], [], []
    output_DRCE_prob = []
    #-------Initialization-------
    nx = 10 #state dimension
    nu = 10 #control input dimension
    ny = 10#output dimension
    temp = np.ones((nx, nx))
    A = np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2)
    B = C = Q = R = Qf = np.eye(10) 
    #----------------------------
    # change True to False if you don't want to use given lambda
    use_lambda = False
    lambda_ = 20 # will not be used if the parameter "use_lambda = False"
    noisedist = [noise_dist1]
    
    #theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # radius of noise ambiguity set
    theta_w_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] # radius of noise ambiguity set
    theta_x0 = 0.5
    theta_v_list = [0.5]
    num_noise_list = [5, 10, 15]
    # Save lambda list
    WDRC_lambda, DRCE_lambda = [],[]
    # if use_lambda == True and dist=="normal":
    #     WDRC_lambda = np.array([26.41121764, 26.16612111, 26.21934116,26.21934116, 26.36159012, 26.3621963, 26.42347884, 26.43546194, 26.42452678 ])
    #     DRCE_lambda = np.array([34.91749428, 40.98912609, 39.05410236, 38.13329201, 38.13329201, 46.68884815, 38.21829064, 45.01168795, 39.02915113])
    # if use_lambda == True and dist=="quadratic":
    #     WDRC_lambda = np.array([18.73975602, 18.67355442, 18.65363162, 18.68274912, 18.67348403, 18.69286136, 18.69810686, 18.7043255])
    #     DRCE_lambda = np.array([22.9903849, 23.78663288, 23.88587422,23.72079564, 24.0604179, 24.02581889, 24.07412486, 24.08671891])
        
    for noise_dist in noisedist:
        for theta_w in theta_w_list:
            for theta in theta_v_list:
                for idx, num_noise in enumerate(num_noise_list):
                    print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise, "/ theta_w : ", theta_w, "/ theta_v : ", theta)
                    np.random.seed(seed) # fix Random seed!
                    print("--------------------------------------------")
                    print("number of noise sample : ", num_noise)
                    print("number of disturbance sample : ", num_samples)
                    if infinite:
                        path = "./results/{}_{}/infinite/multiple/OS/N={}/".format(dist, noise_dist, num_noise)
                    else:
                        path = "./results/{}_{}/finite/multiple/OS/N={}/".format(dist, noise_dist, num_noise)    
                    if not os.path.exists(path):
                        os.makedirs(path)
                
                    #-------Disturbance distribution-------
                    if dist == "normal":
                        #disturbance distribution parameters
                        w_max = None
                        w_min = None
                        mu_w = 0.1*np.ones((nx, 1))
                        Sigma_w= 0.1*np.eye(nx)
                        #initial state distribution parameters
                        x0_max = None
                        x0_min = None
                        x0_mean = 0.1*np.ones((nx,1))
                        x0_cov = 0.1*np.eye(nx)
                    elif dist == "quadratic":
                        #disturbance distribution parameters
                        w_max = 0.2*np.ones(nx)
                        w_min = -0.5*np.ones(nx)
                        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                        Sigma_w = 3.0/20.0*np.diag((w_max - w_min)**2)
                        #initial state distribution parameters
                        x0_max = 0.5*np.ones(nx)
                        x0_min = 0.0*np.ones(nx)
                        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                        x0_cov = 3.0/20.0 *np.diag((x0_max - x0_min)**2)
                    elif dist =="uniform":
                        #disturbance distribution parameters
                        w_max = 0.2*np.ones(nx)
                        w_min = -0.4*np.ones(nx)
                        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                        Sigma_w = 1/12*np.diag((w_max - w_min)**2)
                        #initial state distribution parameters
                        x0_max = 0.5*np.ones(nx)
                        x0_min = 0.0*np.ones(nx)
                        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                        x0_cov = 1/12*np.diag((x0_max - x0_min)**2)
                        
                    #-------Noise distribution ---------#
                    if noise_dist =="normal":
                        v_max = None
                        v_min = None
                        M = 2.0*np.eye(ny) #observation noise covariance
                        mu_v = 0.5*np.ones((ny, 1))
                    elif noise_dist =="quadratic":
                        v_min = -1.0*np.ones(ny)
                        v_max = 2.0*np.ones(ny)
                        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
                    elif noise_dist == "uniform":
                        v_min = -2.0*np.ones(ny)
                        v_max = 6.0*np.ones(ny)
                        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                        M = 1/12*np.diag((v_max - v_min)**2) #observation noise covariance
                        
                        
                    #-------Estimate the nominal distribution-------
                    # Out-of-sample test
                    x0_mean_hat, x0_cov_hat = [], []
                    mu_hat, Sigma_hat = [], []
                    v_mean_hat, M_hat = [], []
                    for i in range(num_sim):
                        # Nominal initial state distribution
                        x0_mean_hat_, x0_cov_hat_ = gen_sample_dist(dist, 1, num_x0_samples, mu_w=x0_mean, Sigma_w=x0_cov, w_max=x0_max, w_min=x0_min)
                        # Nominal Disturbance distribution
                        mu_hat_, Sigma_hat_ = gen_sample_dist(dist, T+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                        # Nominal Noise distribution
                        v_mean_hat_, M_hat_ = gen_sample_dist(noise_dist, T+1, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)
                        M_hat_ = M_hat_ + 1e-8*np.eye(ny) # to prevent numerical error from inverse in standard KF at small sample size
                        
                        x0_mean_hat.append(x0_mean_hat_)
                        x0_cov_hat.append(x0_cov_hat_)
                        mu_hat.append(mu_hat_)
                        Sigma_hat.append(Sigma_hat_)
                        v_mean_hat.append(v_mean_hat_)
                        M_hat.append(M_hat_)
                        
                    
                    #-------Create a random system-------
                    system_data = (A, B, C, Q, Qf, R, M)
                    
                    #-------Perform n  independent simulations and summarize the results-------
                    output_lqg_list = []
                    output_wdrc_list = []
                    output_drce_list = []
                    
                    
                    # Parallelize the simulations
                    results = Parallel(n_jobs=-1)(
                        delayed(perform_simulation)(i, lambda_, num_sim, use_lambda, WDRC_lambda, DRCE_lambda, idx, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat)
                        for i in range(num_sim)
                    )
                    # results = []
                    # for i in range(num_sim):
                    #     result = perform_simulation(i, lambda_, num_sim, use_lambda, WDRC_lambda, DRCE_lambda, idx, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat)
                    #     results.append(result)
                        
                    # Process results
                    lqg = []
                    wdrc = []
                    drce = []
                    
                    for result in results:
                        lqg.append(result['lqg'])
                        wdrc.append(result['wdrc'])
                        drce.append(result['drce'])
                        if not use_lambda:
                            WDRC_lambda.append(result['WDRC_lambda'])
                            DRCE_lambda.append(result['DRCE_lambda'])
    
                    
                        
                    print('---------------------')
                    output_drce = np.empty(num_sim, dtype=object)
                    output_wdrc = np.empty(num_sim, dtype=object)
                    output_lqg = np.empty(num_sim, dtype=object)
                    drce_cert = np.empty(num_sim)
                    os_sample_size = 100
                    
                    
                    #----------------------------
                    print("Running DRCE Forward step ...")
                    #Perform state estimation and apply the controller
                    for i in range(num_sim):
                        drce_cert[i] = drce[i].objective(drce[i].lambda_)*drce[i].T
                        output_drce_sample = []
                        for j in range(os_sample_size):
                            output_drce_ = drce[i].forward()
                            output_drce_sample.append(output_drce_)
                        output_drce[i] = output_drce_sample
                        
                    #----------------------------
                    print("Running WDRC Forward step ...")
                    #Perform state estimation and apply the controller
                    for i in range(num_sim):
                        output_wdrc_sample = []
                        for j in range(os_sample_size):
                            output_wdrc_ = wdrc[i].forward()
                            output_wdrc_sample.append(output_wdrc_)
                        output_wdrc[i] = output_wdrc_sample
                    
                    #----------------------------
                    print("Running LQG Forward step ...")
                    #Perform state estimation and apply the controller
                    for i in range(num_sim):
                        output_lqg_sample = []
                        for j in range(os_sample_size):
                            output_lqg_ = lqg[i].forward()
                            output_lqg_sample.append(output_lqg_)
                        output_lqg[i] = output_lqg_sample
                
                    #-- Out-of-sample performance result SAVE -- #
                
                    J_LQG_list, J_WDRC_list, J_DRCE_list= [], [], []
                    J_LQG_OS_list, J_WDRC_OS_list, J_DRCE_OS_list= [], [], []
                    #lqg-----------------------
                    for i in range(num_sim):
                        for out in output_lqg_sample:
                            J_LQG_list.append(out['cost'][0])
                        
                        # Average cost when using lqg[i] controller   
                        J_LQG_mean= np.mean(J_LQG_list)
                        J_LQG_OS_list.append(J_LQG_mean)
                       
                    
                    output_J_LQG_mean.append( np.mean(J_LQG_OS_list) )
                    output_J_LQG_std.append( np.std(J_LQG_OS_list) )
                    
                    print(" Out-of-Sample cost (LQG) : ", np.mean(J_LQG_OS_list))
                    
                    #wdrc-----------------------
                    for i in range(num_sim):
                        for out in output_wdrc_sample:
                            J_WDRC_list.append(out['cost'][0])
                        
                        # Average cost when using wdrc[i] controller   
                        J_WDRC_mean= np.mean(J_WDRC_list)
                        J_WDRC_OS_list.append(J_WDRC_mean)
                       
                    
                    output_J_WDRC_mean.append( np.mean(J_WDRC_OS_list) )
                    output_J_WDRC_std.append( np.std(J_WDRC_OS_list) )
                    
                    print(" Out-of-Sample cost (WDRC) : ", np.mean(J_WDRC_OS_list))
                    
                    DRCE_prob = np.empty(num_sim)
                    #drce-----------------------
                    for i in range(num_sim):
                        for out in output_drce_sample:
                            J_DRCE_list.append(out['cost'][0])
                        
                        # Average cost when using drce[i] controller   
                        J_DRCE_mean= np.mean(J_DRCE_list)
                        J_DRCE_OS_list.append(J_DRCE_mean)
                        DRCE_prob[i] = (J_DRCE_mean <= drce_cert[i])
                       
                    
                    output_J_DRCE_mean.append( np.mean(J_DRCE_OS_list) )
                    output_J_DRCE_std.append( np.std(J_DRCE_OS_list) )
                    output_DRCE_prob.append(np.mean(DRCE_prob))
                    print(" Out-of-Sample cost (DRCE) : ", np.mean(J_DRCE_OS_list))
                    print(" Prob: ",np.mean(DRCE_prob) )
                    
                    #-------------------------------
                    # Data Save
                    theta_v_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                    theta_w_ = f"_{str(theta_w).replace('.', '_')}" # change 1.0 to 1_0 for file name
                    
                    save_data(path + 'drce_mean_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', output_J_DRCE_mean)
                    save_data(path + 'drce_std_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', output_J_DRCE_std)
                    
                    
                    save_data(path + 'drce_prob_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', output_DRCE_prob)

                    save_data(path + 'wdrc_mean_os_' + theta_w_ + '.pkl', output_J_WDRC_mean)
                    save_data(path + 'wdrc_std_os_' + theta_w_ + '.pkl', output_J_WDRC_std)
                    
                    save_data(path + 'lqg_mean_os.pkl', output_J_LQG_mean)
                    save_data(path + 'lqg_std_os.pkl', output_J_LQG_std)
                    
                    # -- Lambda Save -- 
                    save_data(path + 'drce_lambda_' + theta_w_ + 'and' + theta_v_+ '.pkl', DRCE_lambda)
                    save_data(path + 'wdrc_lambda_' + theta_w_ + '.pkl', WDRC_lambda)
                    #-------------------------------
                    print("num_noise_sample : ", num_noise, " / finished with dist : ", dist, "/ noise_dist : ", noise_dist, "/ seed : ", seed)
            

            
                        
                # # after running noise_samples lists!
                # if noise_plot_results:
                #     if infinite:
                #         path = "./results/{}_{}/infinite/multiple/num_noise_plot/".format(dist, noise_dist)
                #     else:
                #         path = "./results/{}_{}/finite/multiple/num_noise_plot/".format(dist, noise_dist)
                #     if not os.path.exists(path):
                #         os.makedirs(path)
                #     save_data(path + 'drce_mean.pkl', output_J_DRCE_mean)
                #     save_data(path + 'drce_std.pkl', output_J_DRCE_std)  
                #     save_data(path + 'lqg_mean.pkl', output_J_LQG_mean)
                #     save_data(path + 'lqg_std.pkl', output_J_LQG_std) 
                #     save_data(path + 'wdrc_mean.pkl', output_J_WDRC_mean)
                #     save_data(path + 'wdrc_std.pkl', output_J_WDRC_std) 
                    
                #     #Summarize and plot the results
                #     print('\n-------Summary-------')
                #     print("dist : ", dist, "noise_dist : ", noise_dist, "/ num_disturbance_samples : ", num_samples, "/ theta_v : ", theta, " / noise sample effect PLOT / Seed : ",seed)
                    
                #     # reset
                #     output_J_LQG_mean, output_J_WDRC_mean, output_J_DRCE_mean=[], [], []
                #     output_J_LQG_std, output_J_WDRC_std, output_J_DRCE_std=[], [], []
                    
    print("Data generation Completed!!")
    
    if noise_plot_results:
        if infinite:
            print("For noise sample size effect plot : Use python plot_J.py --infinite --dist "+ dist + " --noise_dist " + noise_dist)
        else:
            print("For noise sample size effect plot : Use python plot_J.py --dist "+ dist + " --noise_dist " + noise_dist)
    else:
        if infinite:
            print("For plot : Use python plot.py --infinite --dist "+ dist + " --noise_dist " + noise_dist)
        else:
            print("For plot : Use python plot.py --dist "+ dist + " --noise_dist " + noise_dist)
            
    #print("WDRC Lambda list : ", WDRC_lambda)
    #print("DRCE Lambda list : ", DRCE_lambda)
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--num_sim', required=False, default=100, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=15, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=15, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=20, type=int) #horizon length
    parser.add_argument('--plot', required=False, action="store_true") #plot results+
    parser.add_argument('--noise_plot', required=False, action="store_true") # noise sample size plot
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon, args.plot, args.noise_plot, args.infinite)
