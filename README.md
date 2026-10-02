# BOREAS: Action-Conditioned World Modeling for Data Center Cooling Systems

BOREAS is a probabilistic, action-conditioned world model for forecasting and simulating the thermal dynamics of data center cooling systems. It is designed to support counterfactual action evaluation, multi-step rollout simulation, and downstream control-policy assessment.

## Overview

Data center cooling is governed by coupled thermal, airflow, and control dynamics. A useful world model must therefore capture:

- spatial interactions among thermal entities;
- recurrent temporal dependencies;
- the effect of alternative cooling actions;
- exogenous workload variations; and
- uncertainty in future thermal-state transitions.

Given the current recurrent memory state \(m_t\), a candidate action \(a_t\), and the observed exogenous workload \(\xi_t\), BOREAS predicts the probability distribution of the normalized observation increment

\[
\Delta o_t = o_{t+1} - o_t.
\]

The predicted observation is then fed back as the next input during recursive simulation. This enables BOREAS to evaluate how different control actions may affect future thermal states over multiple prediction steps.

The model follows an Update--Predict architecture:

1. **HETI encoder.** The Heterogeneous Entity Thermal Interaction encoder represents interactions among data-center entities and produces a graph-based thermal representation.

2. **RTCU updater.** The Recurrent Thermal-Context Updater maintains a compact recurrent memory of the evolving thermal context.

3. **Transition head.** The transition head maps the current memory, action, and exogenous context to the mean and variance of the next observation increment.

4. **Inverse-dynamics head.** An auxiliary inverse head reconstructs the action from the current memory state and the next observation. This explicitly constrains the action-related component of the state transition and improves action consistency during recursive rollout.

## Task Definition

The model predicts 46 output channels grouped into six physical categories:

| Output group | Number of channels | Description |
|---|---:|---|
| Cold-T | 12 | Cold-aisle temperature |
| Hot-T | 2 | Hot-aisle temperature |
| Cold-RH | 12 | Cold-aisle relative humidity |
| Hot-RH | 2 | Hot-aisle relative humidity |
| ACU LAT | 9 | Air-conditioning unit leaving-air temperature |
| ACU EAT | 9 | Air-conditioning unit entering-air temperature |
| **Total** | **46** | Complete output vector |

The evaluation uses 10,985 contiguous test windows. Each window is initialized with eight historical observations for recurrent warm-up. During recursive rollout, predicted observations are fed back as the next model inputs, while the recorded future control signals and IT-load inputs are provided to all methods.

The evaluated horizons are:

- \(H=1\): 5 minutes;
- \(H=5\): 25 minutes;
- \(H=10\): 50 minutes;
- \(H=15\): 75 minutes.

Physical-scale RMSE and MAE are reported for each variable group. For cross-variable comparison, we use channel-standardized physical RMSE, denoted as **cs-pRMSE**, in which each channel is normalized by its training-set standard deviation before aggregation. Lower values indicate better performance.

## Experimental Configuration

### Model Architecture

| Component | Configuration |
|---|---|
| Ensemble members | 5 |
| HETI representation width | 64 |
| Exogenous representation width | 64 |
| HETI graph-network layers | 2 |
| Attention heads | 4 |
| RTCU/GRU hidden-state width | 256 |
| Transition-head hidden width | 256 |
| Transition-head depth | 1 layer |
| Inverse-head hidden width | 256 |
| Reported forward parameters | Approximately 3.619M |

### Training Objective

| Component | Configuration |
|---|---|
| Main objective | One-step observation-increment Gaussian negative log-likelihood |
| Log-variance bound penalty | 0.01 |
| Inverse-loss weight | 0.03 |
| Inverse magnitude-loss weight | 1.0 |
| Inverse positive-sample weight exponent | 1.0 |
| Inverse-loss warm-up | First 1,000 updates |
| Inverse-loss components | Prevalence-weighted BCE for nonzero-action detection and normalized Smooth L1 loss for nonzero-action magnitude |
| Multi-step rollout loss | Not used in final training |

The inverse objective encourages the model to recover the action that caused the observed thermal transition. This provides an explicit action--state consistency constraint and helps reduce systematic drift during long-horizon recursive simulation.

### Optimization

| Setting | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | \(10^{-3}\) |
| Weight decay | \(10^{-5}\) |
| Training batch size | 128 |
| Validation batch size | 512 |
| Gradient-norm clipping | 100 |
| Maximum updates | 30,000 |
| Validation frequency | Every 250 updates |
| Early-stopping patience | 15 validation checks |
| Minimum relative improvement | 0.001 |

The final model is trained with a one-step probabilistic objective. Multi-step recursive rollout is used for evaluation and checkpoint selection rather than as a direct training loss.

## Main Results

The table reports overall cs-pRMSE across all 46 output channels. The \(H=1\) column measures single-step prediction, while \(H=5,10,15\) measure recursive rollout performance. Lower is better.

| Method | \(H=1\) (5 min) | \(H=5\) (25 min) | \(H=10\) (50 min) | \(H=15\) (75 min) |
|---|---:|---:|---:|---:|
| OLS--ARX | 0.2973 | 0.6263 | 0.8037 | 0.9293 |
| Ridge--ARX | 0.2566 | 0.5509 | 0.7075 | 0.8246 |
| DMDc | 0.3485 | 0.6228 | 0.7871 | 0.9171 |
| Subspace-SS | 0.5666 | 0.6236 | 0.6480 | 0.6720 |
| Kalman-SS | 0.4965 | 0.6544 | 0.7845 | 0.9171 |
| Informer | 0.5926 | 0.8155 | 0.9210 | 0.9963 |
| TSMixer | 0.2906 | 0.5534 | 0.6890 | 0.8108 |
| TimesNet | <u>0.2594</u> | <u>0.4631</u> | <u>0.5620</u> | <u>0.6523</u> |
| LightTS | 0.2953 | 0.4819 | 0.6138 | 0.7396 |
| **BOREAS** | **0.2145** | **0.4179** | **0.4605** | **0.4970** |

BOREAS achieves the lowest cs-pRMSE at every evaluated horizon. Its advantage becomes more pronounced as the rollout horizon increases, indicating lower recursive error amplification and stronger long-term state stability.

Relative to the strongest time-series baseline, TimesNet, BOREAS reduces cs-pRMSE by:

- 17.3% at \(H=1\);
- 9.8% at \(H=5\);
- 18.1% at \(H=10\); and
- 23.8% at \(H=15\).

The cs-pRMSE of BOREAS increases from 0.4179 at \(H=5\) to 0.4970 at \(H=15\), corresponding to an 18.9% increase. TimesNet increases from 0.4631 to 0.6523 over the same range, corresponding to a 40.9% increase.

## BOREAS Performance by Physical Group

The following table reports physical-scale RMSE/MAE for BOREAS. Temperature errors are measured in degrees Celsius, while humidity errors are measured in percentage points.

| Output group | \(H=1\) | \(H=5\) | \(H=10\) | \(H=15\) |
|---|---:|---:|---:|---:|
| Cold-T | 0.4137 / 0.1877 | 0.8188 / 0.3929 | 0.8763 / 0.4612 | 0.9250 / 0.5101 |
| Hot-T | 0.1275 / 0.0810 | 0.2524 / 0.1545 | 0.2989 / 0.1891 | 0.3169 / 0.2063 |
| Cold-RH | 1.2725 / 0.5625 | 2.5520 / 1.1740 | 2.7177 / 1.3643 | 2.8487 / 1.4889 |
| Hot-RH | 0.2410 / 0.1619 | 0.5145 / 0.3361 | 0.6678 / 0.4490 | 0.7522 / 0.5142 |
| ACU LAT | 0.2114 / 0.0857 | 0.3524 / 0.1456 | 0.4123 / 0.1945 | 0.4821 / 0.2424 |
| ACU EAT | 0.0897 / 0.0575 | 0.1584 / 0.0938 | 0.2162 / 0.1308 | 0.2765 / 0.1681 |

BOREAS provides particularly strong performance on ACU thermal variables. This is important for control-oriented applications because ACU leaving-air and entering-air temperatures directly characterize the response of the cooling equipment.

## Reproducibility Notes

- All reported results use the same 10,985 contiguous test windows.
- The 46 output channels are evaluated both individually and after aggregation into six physical groups.
- Rollout metrics are pooled over steps \(1{:}H\), rather than computed only at the terminal step.
- The final model does not use a multi-step rollout loss during training.
- The reported main checkpoint corresponds to seed 0.
- Across six independent BOREAS trainings, the \(H=15\) cs-pRMSE ranges from 0.4970 to 0.5285, indicating moderate sensitivity to random initialization.

## Intended Applications

BOREAS can be used as a learned thermal simulator for:

- counterfactual evaluation of alternative cooling actions;
- long-horizon control-policy validation;
- model-predictive control;
- offline reinforcement learning;
- thermal-risk analysis under changing IT workloads; and
- simulation-based cooling-system planning.

The model is intended to support decision-making and policy evaluation. Operational control decisions should remain subject to engineering constraints and expert review.
