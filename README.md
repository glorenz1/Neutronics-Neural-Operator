# Neutronics-Neural-Operator

An application of neural operators to reactor physics using a Fourier Neural Operator (FNO) trained on OpenMC Monte Carlo simulations. The model acts as a surrogate for neutron transport equation by learning the functional mapping from fuel enrichment distributions to neutron flux and effective multiplication factor.

The goal is to learn the operator:

    enrichment -> neutron flux + k_eff

so that new reactor configurations can be evaluated in milliseconds without rerunning expensive Monte Carlo simulations.

## Overview

1. Generate realistic reactor data using OpenMC
2. Sample diverse core configurations:
   - varying enrichment fields and spatial resolutions
   - multiple geometries (square, circular, hexagonal)
   - different assembly sizes
3. Train an FNO to approximate the solution operator
4. Predict:
   - full 2D neutron flux field
   - effective multiplication factor (k_eff)
