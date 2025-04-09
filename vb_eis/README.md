# Probabilistic deconvolution of impedance data using variational Bayes

This repository contains the supplementary material.
The implementation of the model is available at [tau_models.py](https://repo.ijs.si/e2pub/vb_eis/-/blob/main/tau_models.py?ref_type=heads).

Order and parameter estimation script is [loop_train_tau.py](https://repo.ijs.si/e2pub/vb_eis/-/blob/main/loop_train_tau.py?ref_type=heads).
The script requires the following parameters:
- `N` - integer that describes the maximal number of poles in the impedance function
- `epochs` - number of itterations. Optimising evidence lower-bound (ELBO) does not guarantee global optimum. Therefore one has to decide for stopping criterion
- `num_particles` - the number of Monte Carlo estimates when calculating the KL distance required for ELBO. Since we are using discrete latent variables some combinations of KL do not exist in closed form
- `Z` - `numpy` NPZ file that contans `f` as frequency and `Z` az `np.complex` values of impedance at those frequencies

## Dirichlet approximation
The same analysis as above but repeated with appriximation of the Dirichlet distribution using softmax.
The complete process is available in the notebook [logistic-normal.ipynb](https://repo.ijs.si/e2pub/vb_eis/-/blob/main/logistic-normal.ipynb?ref_type=heads).
The notebook defines the model differently.
In essense it is an implementation of Scenario 2 from the paper, where each model has its own parameter set.


## DRT derivation
The notebook [DRT_derivation.ipynb](https://repo.ijs.si/e2pub/vb_eis/-/blob/main/DRT_derivation.ipynb?ref_type=heads) contains the derivation of DRT from a transfer function (equivalent circuit model).
The implementation is based on [sympy](https://www.sympy.org/en/index.html).
This derivation is instrumental for calculation of pole importance.

## Note on the implementation
The analysis is based on [pyro-ppl](https://pyro.ai).
For scenario 3, in the calculation of the ELBO loss, one has to mask the likelihood values of parameter components that are not used for a particular number of components.
For that purpose we used a modification of the original `pyro-ppl` library available [here](https://repo.ijs.si/pboskoski/pyro-ppl).
However, one can use a workaround from the original `pyro-ppl` as described [here](https://github.com/pyro-ppl/pyro/issues/3305).
Shortly we will migrate our implementation to use this workaround so that our implementation will depend on the original [pyro-ppl](https://pyro.ai) library.

## Time domain simulation of fractional order RQ models
The time domain simulation is available in [EIS_time_domain_simulation.ipynb](EIS_time_domain_simulation.ipynb).
The notebook contains code for simulation of time-domain response of a fractional order system in the discrete domain.
The simulator implementation is available in [state_space_sim.py](state_space_sim.py).