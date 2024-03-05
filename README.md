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

## Plot
1) System Disturbance : Normal & Observation Noise : Normal
<center>
  <img src='/result_save/normal_normal_params/normal_normal_params.jpg' width='500' height='300' />
  <figcaption>Normal Disturbance and Noise distribution</figcaption>
</center>
<center>
  <img src='/result_save/normal_normal_noiseplot/normal_normal_noiseplot.jpg' width='500' height='300' />
  <figcaption>Normal Disturbance and Noise distribution</figcaption>
</center>

## Code explanation

1234

## Usage

To run the experiments, call the main python script:
```
python main.py
```

The parameters can be changed by adding additional command-line arguments:
```
python main.py --dist normal --sim_type multiple --num_sim 1000 --num_samples 5 --horizon 50 --plot
```

The results for both controllers are saved in separate pickle files in `/results/<dist>/<sim_type>/`. If command-line argument `--plot` is invoked, the state, control, output trajectories, and cost histograms are plotted.


An example output with default settings:

```
1234
```
