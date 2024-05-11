Distributionally Robust Control and Estimation(DRCE) for Partially Observable Linear Stochastic Systems
====================================================

This repository includes the source code for implementing Linear-Quadratic-Gaussian(LQG), Wasserstein Distributionally Robust Controller(WDRC), and Distributionally Robust Control and Estimation(DRCE)

## Requirements
- Python (>= 3.5)
- numpy (>= 1.17.4)
- scipy (>= 1.6.2)
- matplotlib (>= 3.1.2)
- control (>= 0.9.4)
- **[CVXPY](https://www.cvxpy.org/)**
- **[MOSEK (>= 9.3)](https://www.mosek.com/)**
- (pickle5) if relevant error occurs
## Additional Requirements to run DRLQC paper
- Pytorch 2.0
- [Pymanopt] https://pymanopt.org/

## Code explanation

There are 2 main codes to generate data : main.py and main_param.py.

### main.py

main.py generates the cost for different noise sample size. To run the experiment in default setting, call the main python script:
```
python main.py --noise_plot
```
This will generate lqg, wdrc, drce mean and standard deviation of the total cost inside the results/normal_normal/finite/multiple/num_noiseplot directory.

The parameters can be changed by adding additional command-line arguments:
- dist : Select disturbance distribution (normal / uniform / quadratic) [default : normal]
- noise_dist : Select noise distribution (normal / uniform / quadratic) [default : normal]
- num_sim : Select the number of repetition [default : 500]
- num_samples : Select the number of disturbance samples [default : 10]
- num_noise_samples : Select the number of noise samples [default : 10]
- horizon : Select the time horizon  [default : 20]

Example
```
python main.py --dist quadratic --noise_dist quadratic --num_sim 1000 --num_samples 15
```
After the data generated, run 
```
python plot_J.py
```
You need to indicate what data you will use to draw the noise_sample_plot. For example,
```
python plot_J.py --dist quadratic --noise_quadratic
```
will draw the plot using the data inside results/quadratic_quadratic/finite/multiple/num_noise_plot/ 

### main_param.py

main_param.py generates the total cost for different lambda, theta_v parameters. To run the experiment in default setting, call the main python script:
```
python main_param.py
```
will generate the .pkl files inside the results/normal_normal/finite/multiple/params_lambda/ directory.

After the data generated, run 
```
python plot_params.py
```

Same as main.py instructions, you need to specifiy what distribution you used, if you didn't use default setting. For example, if you use quadratic distribution for both disturbance and noise distribution, run
```
python plot_params.py --dist quadratic --noise_dist quadratic
```


## Plot
### System Disturbance & Observation Noise : Nonzero-mean Normal distribution
Disturbance : N(0,1) , Initial State : N(0, 0.01), Noise : N(0.3, 0.3)
<center>
  <img src='/result_save/normal_normal_params/normal_normal_params.jpg' width='500'/>
  <figcaption>10 disturbance & noise samples</figcaption>
</center>
<center>
  <img src='/result_save/normal_normal_noiseplot/normal_normal_noiseplot.jpg' width='500' />
  <figcaption>lambda : 10, theta_v : 1.5, theta_x0 = 0.5</figcaption>
</center>

### System Disturbance & Observation Noise : Nonzero-mean Uniform distribution
Disturbance : U(-0.3,0.6) , Initial State : U(-0.05, 0.05), Noise : U(-0.5, 1.0)
<center>
  <img src='/result_save/uniform_uniform_params/uniform_uniform_params.jpg' width='500'/>
  <figcaption>10 disturbance & noise samples</figcaption>
</center>
<center>
  <img src='/result_save/uniform_uniform_noiseplot/uniform_uniform_noiseplot.jpg' width='500' />
  <figcaption>lambda : 10, theta_v : 2.0, theta_x0 = 0.5</figcaption>
</center>

### System Disturbance & Observation Noise : Nonzero-mean U-Quadratic distribution
Disturbance : UQ(-0.1,0.2) , Initial State : UQ(-0.1, 0.1), Noise : UQ(0.0, 1.0)
<center>
  <img src='/result_save/quad_quad_params/quad_quad_params.jpg' width='500'/>
  <figcaption>10 disturbance & noise samples</figcaption>
</center>
<center>
  <img src='/result_save/quad_quad_noiseplot/quad_quad_noiseplot.jpg' width='500' />
  <figcaption>lambda : 10, theta_v : 2.0, theta_x0 = 0.5</figcaption>
</center>
