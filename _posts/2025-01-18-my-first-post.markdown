---
layout: post
title: "Neural Synthesis of Binaural Speech from Mono Audio"
date: 2025-01-18
---

## Neural Synthesis of Binaural Speech from Mono Audio

This post summarizes and explains a paper from Facebook Reality Labs (Pittsburgh, USA) that introduces a **neural rendering approach for real-time binaural sound synthesis**. In essence, the model takes a single-channel (mono) audio signal and produces a two-channel (binaural) output, conditioned on the relative position and orientation of the listener and source. The authors also provide an in-depth look at why minimizing raw waveform error using an $ l_2 $-loss can be problematic, and propose a modified loss function incorporating phase information for better audio quality.

---

### Motivation: Binaural Audio in AR/VR

Augmented Reality (AR) and Virtual Reality (VR) demand immersive audio that closely mimics real-world listening experiences. Human spatial hearing relies on small differences between signals arriving at the left and right ears (i.e., binaural signals) to determine direction, distance, and movement of sound sources. Accurately generating these signals enhances user orientation in virtual worlds and fosters immersion by matching visual cues with congruent acoustic cues.

---

### Problem Statement

We have:
- A **mono (single-channel)** input signal $ x_{1:T} $ of length $ T $.  
- A **binaural (two-channel)** target output 
$$ 
\{ y^{(l)}_{1:T},\, y^{(r)}_{1:T} \} 
$$ for the left and right ears.  
- A **conditioning** signal $ c_{1:T} \in \mathbb{R}^{14} $ describing source and listener positions (3D) and orientations (as quaternions).

Formally, the task is to learn a function

$$
\hat{y}^{(l)}_t,\; \hat{y}^{(r)}_t \;=\; f\bigl(x_{t-\Delta : t} \;\big\vert\; c_{t-\Delta : t}\bigr),
$$

where $ \Delta $ is a temporal receptive field. In simpler terms, the model should transform the mono audio into binaural audio based on where the source and the listener are in 3D space.

---

### Overview of the Proposed Approach

The system has two main components:

1. **Neural Time Warping**  
   Aligns (warps) the mono signal into two separate streams, roughly corresponding to the left and right ears. The warping accounts for coarse time delays (e.g., interaural time differences and Doppler effects).

2. **Temporal ConvNet with Conditioned Hyper-Convolutions**  
   A deep network refines these warped signals to model more subtle acoustic effects, such as room reverberation, ear shape filtering, and dynamic head orientations.

#### 1. Neural Time Warping

Traditionally, **Dynamic Time Warping (DTW)** has been used to align source and target signals. However, the target binaural signal is unknown at inference, so the paper introduces **learned neural warping** guided by the geometry of the source-listener setup.

- **Geometric Warping**: A first approximation warps the signal based on distance and the speed of sound:
  
  $$
  \rho^{(\text{geom})}_t \;=\; t \;-\; 
    \frac{\|p_{\text{src}, t} - p_{\text{lstn}, t}\|\,\times\,(\text{audio sample rate})}
         {\nu_{\text{sound}}},
  $$
  
  where $ p_{\text{src}, t} $ and $ p_{\text{lstn}, t} $ are positions of the source and listener at time $ t $.

- **Neural Correction**: A small convolutional network, WarpNet, refines the geometric warping to account for head diffraction and other nuances. The final warp field $ \rho $ is constrained to be **monotonic** and **causal** via a special activation function.

- **Linear Interpolation**: Because $ \rho_t $ may not be an integer, the warped signal is computed by interpolating the original signal at fractional indices.

This yields initial left-ear and right-ear signals that are time-aligned in a physically consistent way.

#### 2. Conditioned Hyper-Convolutions

After warping, a **temporal convolutional network** refines the signals. Instead of standard convolutional filters (fixed over time), the authors propose **hyper-convolutions**, where **filter weights themselves** are functions of the conditioning variables $ c_{1:T} $.

- **Standard Approach**: A typical conditional convolution adds transformations of the conditioning signal to the output.  
- **Hyper-Convolution**: The filters become **time-varying**. A small hyper-network predicts the convolutional weights from $ c_{1:T} $, letting the system adapt dynamically to changes in source-listener geometry. This is crucial for modeling moving sound sources, rotating heads, and evolving room acoustics.

---

### The Loss Function: Mitigating Phase Errors

Simply minimizing an $ l_2 $-loss on raw waveforms often leads to **poor phase reconstruction**. While amplitude may be fitted well, large phase discrepancies can remain, creating audible artifacts (especially noticeable in speech). To address this, the paper adds a **phase-oriented term**:

$$
L\bigl(y,\hat{y}\bigr) 
\;=\;
\bigl\|y - \hat{y}\bigr\|_2^2 \;+\;
\lambda \, L_{\text{phase}}\Bigl(\text{STFT}(y),\;\text{STFT}(\hat{y})\Bigr),
$$

where $ L_{\text{phase}}(\cdot) $ measures phase discrepancies in the short-time Fourier transform (STFT) domain. Experiments show that this additional term significantly improves the phase fidelity, thus yielding higher-quality binaural audio.

---

### Evaluation and Results

- **Dataset**:  
  A two-hour paired mono-binaural dataset at 48 kHz from eight speakers (four male, four female) walking around a mannequin equipped with binaural microphones. Positions and orientations are tracked with OptiTrack.

- **Quantitative Metrics**:  
  The proposed method is compared to both classic DSP-based binauralization and other neural approaches. It achieves **lower amplitude and phase errors** and produces more realistic binaural signals.

- **Real-Time Performance**:  
  The system runs quickly enough for streaming (around 33 ms latency on an NVidia Tesla V100). This meets real-time requirements for AR/VR scenarios that typically refresh at 30–120 Hz.

#### Neural Time Warping Impact

Ablation studies show that geometric warping alone does not fully align the signals. The **learned neural warp** refines time shifts, substantially reducing phase error. Even if these corrections appear visually small, they greatly enhance perceptual fidelity.

#### Temporal HyperConv Network Impact

Beyond alignment, the warped signal still needs to account for reflections, reverberations, and ear shape filtering. A simple linear amplitude adjustment is insufficient, underscoring the importance of a **deep time-domain network**.

---

### Conclusion

This work demonstrates the first purely **data-driven, end-to-end** model for high-fidelity binaural audio synthesis from mono. Key highlights include:

1. **Neural Time Warping** for physically consistent alignment.
2. **Conditioned Hyper-Convolutions** to capture complex, time-varying acoustic phenomena.
3. A **phase-aware** loss function that significantly improves perceptual quality.
4. **Real-time** performance suitable for AR/VR and gaming.

In sum, the approach **outperforms classical DSP** methods in generating more realistic binaural sound and offers a promising direction for immersive audio in artificial environments.