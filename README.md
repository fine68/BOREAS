# BOREAS: Action-Conditioned World Modeling for Data Center Cooling Systems

<p align="center">
  <b>A learning-based digital twin for recursive simulation of data-center cooling dynamics under control actions and IT workloads.</b>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/Fine6868/BOREAS"><b>Dataset</b></a>
  &nbsp;|&nbsp;
  <a href="#citation"><b>Citation</b></a>
</p>

<p align="center">
  <img src="assets/figure2_boreas_overview.png" width="100%" alt="Overview of the BOREAS framework">
</p>

## Overview

**BOREAS** is an action-conditioned probabilistic world model for data-center cooling systems. It learns controlled thermal dynamics from operational telemetry and recursively predicts future thermal states under specified cooling-control and IT-workload trajectories.

The model addresses three challenges in real-world cooling-system modeling:

- **Heterogeneous spatial interactions:** temperature and humidity sensors and air-conditioning units (ACUs) have different physical roles and feature spaces.
- **History dependence:** partially observed thermal dynamics depend on recent system states, control actions, and workload conditions.
- **Action-response identification:** strong thermal inertia may allow a predictor to achieve low short-term error while underutilizing control information.

BOREAS addresses these challenges through four components:

1. a **Heterogeneous-Entity Thermal Interaction Encoder (HETI)**;
2. a **Recurrent Thermal-Context Updater (RTCU)**;
3. a probabilistic **Transition Head** for action-conditioned thermal dynamics; and
4. an auxiliary **Inverse-Dynamics Representation Regularization (IDRR)** objective.

Together, these components provide a compact latent representation for recursive simulation and counterfactual action evaluation.

## Motivation

Evaluating cooling strategies directly in a production data center can be costly and risky. A learned digital twin provides a simulation-based alternative for analyzing how thermal states evolve under candidate control actions before deployment in the physical system.

<p align="center">
  <img src="assets/figure1_digital_twin.png" width="95%" alt="Schematic of a data-center cooling digital twin">
</p>

The released operational dataset contains synchronized thermal observations, cooling-control signals, and IT workloads collected from a real-world data-center cooling system. After data cleaning, the study retains **64,825 valid state transitions** sampled at **5-minute intervals**.

## Method

BOREAS separates thermal-context encoding from action-conditioned transition prediction.

### 1. Heterogeneous-Entity Thermal Interaction Encoding

At each time step, sensors and ACUs are represented as heterogeneous entities. Because explicit sensor-to-ACU adjacency is facility-specific, BOREAS does not impose a manually designed graph topology. Instead, global self-attention is applied across all entities to learn interactions within and across entity types.

The resulting representation is processed together with the exogenous IT-workload embedding and the preceding control action by the **Recurrent Thermal-Context Updater (RTCU)**. The RTCU maintains a recurrent memory of the current operating condition and relevant thermal history.

### 2. Probabilistic Action-Conditioned Transition Prediction

Conditioned on the current recurrent memory, a candidate control action, and the concurrent IT workload, BOREAS predicts a Gaussian distribution over the normalized next-observation increment:

$$\Delta o_t = o_{t+1}-o_t.$$ 

The predicted increment is added to the current observation to obtain the next thermal state. During recursive simulation, the predicted state is fed back as the next input, enabling multi-step rollout under a specified control and workload sequence.

The probabilistic formulation represents both the expected thermal transition and the associated predictive uncertainty.

### 3. Inverse-Dynamics Representation Regularization

BOREAS introduces a training-only inverse-dynamics objective to ensure that the latent state preserves information about control effects. The inverse head reconstructs the action from the current recurrent memory and the subsequent observation.

This auxiliary objective constrains the action-related component of the learned transition representation. It discourages the model from explaining state changes solely through thermal inertia and improves action consistency during recursive rollout.

## Dataset

The experiments use operational data collected from a production data center in Guangdong Province, China, from **January 1 to December 11, 2025**.

| Item | Description |
|---|---|
| Sampling interval | 5 minutes |
| Valid state transitions | 64,825 |
| Predicted thermal variables | 46 channels |
| Cooling equipment | 9 ACUs |
| Equipment-status signals | 9 |
| Fan/valve control signals | 18 |
| IT-load signals | 8 |
| Data split | Chronological train/validation/test split, approximately 7:1:2 |

The 46 predicted channels are organized into six physical groups:

| Output group | Number of channels | Description |
|---|---:|---|
| Cold-T | 12 | Cold-aisle temperature |
| Hot-T | 2 | Hot-aisle temperature |
| Cold-RH | 12 | Cold-aisle relative humidity |
| Hot-RH | 2 | Hot-aisle relative humidity |
| ACU LAT | 9 | ACU leaving-air temperature |
| ACU EAT | 9 | ACU entering-air temperature |
| **Total** | **46** | Complete output vector |

Dataset: **[https://huggingface.co/datasets/Fine6868/BOREAS](https://huggingface.co/datasets/Fine6868/BOREAS)**

## Experimental Protocol

All methods are evaluated on the same **10,985 contiguous test windows**. Each window is initialized with eight historical observations for recurrent warm-up.

During recursive rollout:

- recorded future control inputs are provided to every method;
- the recorded future IT-load trajectory is provided as an exogenous input; and
- each predicted observation is fed back as the next model input.

The evaluated horizons are:

- $H=1$: 5 minutes;
- $H=5$: 25 minutes;
- $H=10$: 50 minutes; and
- $H=15$: 75 minutes.

Physical-scale RMSE and MAE are reported for each physical variable group. For cross-variable comparison, we use **channel-standardized pooled RMSE (cs-pRMSE)**. Each channel is normalized by its training-set standard deviation before errors are pooled across channels and rollout steps. Lower values indicate better performance.

## Experimental Configuration

### Model Architecture

| Component | Configuration |
|---|---:|
| Ensemble members | 5 |
| HETI representation width | 64 |
| Exogenous representation width | 64 |
| HETI graph-network layers | 2 |
| Attention heads | 4 |
| RTCU/GRU hidden-state width | 256 |
| Transition-head hidden width | 256 |
| Transition-head depth | 1 layer |
| Inverse-head hidden width | 256 |
| Total forward parameters | Approximately 3.619M |

### Training Objective

| Component | Configuration |
|---|---|
| Main objective | One-step observation-increment Gaussian NLL |
| Log-variance bound penalty | 0.01 |
| Inverse-loss weight | 0.03 |
| Inverse magnitude-loss weight | 1.0 |
| Inverse positive-sample weight exponent | 1.0 |
| Inverse-loss warm-up | First 1,000 updates |
| Inverse-loss components | Prevalence-weighted BCE for nonzero-action detection and normalized Smooth L1 loss for nonzero-action magnitude |
| Multi-step rollout loss | Not used in final training |

### Optimization

| Setting | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | $10^{-3}$ |
| Weight decay | $10^{-5}$ |
| Training batch size | 128 |
| Validation batch size | 512 |
| Gradient-norm clipping | 100 |
| Maximum updates | 30,000 |
| Validation frequency | Every 250 updates |
| Early-stopping patience | 15 validation checks |
| Minimum relative improvement | 0.001 |

The final model is trained with a one-step probabilistic objective. Multi-step recursive rollout is used for evaluation and checkpoint selection rather than as a direct training loss.

## Experimental Results

BOREAS is compared with classical dynamics models and deep time-series forecasting baselines, including OLS-ARX, Ridge-ARX, DMDc, Subspace-SS, Kalman-SS, Informer, TSMixer, TimesNet, and LightTS.

### Overall cs-pRMSE

The following table reports cs-pRMSE over all 46 output channels. For $H>1$, the metric is pooled over rollout steps $1{:}H$.

| Method | $H=1$ | $H=5$ | $H=10$ | $H=15$ |
|---|---:|---:|---:|---:|
| OLS-ARX | 0.2973 | 0.6263 | 0.8037 | 0.9293 |
| Ridge-ARX | 0.2566 | 0.5509 | 0.7075 | 0.8246 |
| DMDc | 0.3485 | 0.6228 | 0.7871 | 0.9171 |
| Subspace-SS | 0.5666 | 0.6236 | 0.6480 | 0.6720 |
| Kalman-SS | 0.4965 | 0.6544 | 0.7845 | 0.9171 |
| Informer | 0.5926 | 0.8155 | 0.9210 | 0.9963 |
| TSMixer | 0.2906 | 0.5534 | 0.6890 | 0.8108 |
| TimesNet | 0.2594 | 0.4631 | 0.5620 | 0.6523 |
| LightTS | 0.2953 | 0.4819 | 0.6138 | 0.7396 |
| **BOREAS** | **0.2145** | **0.4179** | **0.4605** | **0.4970** |

BOREAS achieves the lowest cs-pRMSE at every evaluated horizon. Relative to the strongest time-series baseline, TimesNet, BOREAS reduces cs-pRMSE by **17.3%** at $H=1$, **9.8%** at $H=5$, **18.1%** at $H=10$, and **23.8%** at $H=15$. At $H=15$, BOREAS improves over the strongest dynamics baseline, Subspace-SS, by **26.0%**.

The cs-pRMSE of BOREAS increases from 0.4179 at $H=5$ to 0.4970 at $H=15$, corresponding to an 18.9% increase. In comparison, TimesNet increases from 0.4631 to 0.6523, corresponding to a 40.9% increase. This indicates weaker recursive error amplification and better long-horizon stability.

### BOREAS Performance by Physical Group

The following table reports physical-scale RMSE/MAE for BOREAS. Temperature errors are measured in degrees Celsius, while humidity errors are measured in percentage points.

| Output group | $H=1$ | $H=5$ | $H=10$ | $H=15$ |
|---|---:|---:|---:|---:|
| Cold-T | 0.4137 / 0.1877 | 0.8188 / 0.3929 | 0.8763 / 0.4612 | 0.9250 / 0.5101 |
| Hot-T | 0.1275 / 0.0810 | 0.2524 / 0.1545 | 0.2989 / 0.1891 | 0.3169 / 0.2063 |
| Cold-RH | 1.2725 / 0.5625 | 2.5520 / 1.1740 | 2.7177 / 1.3643 | 2.8487 / 1.4889 |
| Hot-RH | 0.2410 / 0.1619 | 0.5145 / 0.3361 | 0.6678 / 0.4490 | 0.7522 / 0.5142 |
| ACU LAT | 0.2114 / 0.0857 | 0.3524 / 0.1456 | 0.4123 / 0.1945 | 0.4821 / 0.2424 |
| ACU EAT | 0.0897 / 0.0575 | 0.1584 / 0.0938 | 0.2162 / 0.1308 | 0.2765 / 0.1681 |

BOREAS performs particularly well on ACU thermal variables, which directly characterize the response of cooling equipment and are relevant to control-oriented applications.

## Ablation Findings

The ablation study evaluates the contribution of the principal architectural and training components under the same recursive rollout protocol.

- Removing **IDRR** increases cs-pRMSE from 1.4% at $H=1$ to 7.3% at $H=15$, indicating that inverse-dynamics supervision primarily improves recursive action consistency and long-horizon stability.
- Removing the **RTCU recurrent updater** produces a larger long-horizon degradation, reaching approximately 14.2% at $H=15$, confirming the importance of recurrent memory for partially observed thermal dynamics.
- Removing the **HETI interaction encoder** increases cs-pRMSE by approximately 4.3%-5.9% across the evaluated horizons, demonstrating the value of heterogeneous entity interactions.
- Removing action or exogenous inputs weakens the model's ability to distinguish state changes caused by control interventions and workload variations.

## Hyperparameter Sensitivity

<p align="center">
  <img src="assets/figure3_sensitivity.png" width="100%" alt="Hyperparameter sensitivity of BOREAS">
</p>

The sensitivity analysis covers HETI encoder depth, HETI representation width $d_{\mathrm{model}}$, RTCU recurrent-state width, attention-head count, Transition-head width, and Transition-head depth. The adopted configuration uses a **64-dimensional HETI representation**, a **64-dimensional exogenous representation**, a **256-dimensional RTCU state**, a **256-dimensional Transition Head**, and a **five-member ensemble**.

## Scope and Limitations

The current evaluation is based on operational data from a **single facility**. The recursive experiments provide recorded future control inputs and IT-load trajectories to all methods; therefore, they primarily assess recursive state propagation along observed operating trajectories.

The current results do not establish reliable counterfactual accuracy for arbitrary, experimentally unvalidated action sequences, nor do they directly demonstrate post-deployment energy savings. Assessing broader transferability requires additional facilities, independently collected operating regimes, and validation under controlled action interventions. Deployment decisions involving thermal safety or equipment limits should remain subject to facility-specific constraints, expert review, and independent safety mechanisms.


