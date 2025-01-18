---
layout: post
title: "Neural Synthesis of Binaural Speech from Mono Audio"
date: 2025-01-18
---

## Neural Synthesis of Binaural Speech from Mono Audio

This post summarizes and explains a paper from Facebook Reality Labs (Pittsburgh, USA) that introduces a **neural rendering approach for real-time binaural sound synthesis**. In essence, the model takes a single-channel (mono) audio signal and produces a two-channel (binaural) output, conditioned on the relative position and orientation of the listener and source. The authors also provide an in-depth look at why minimizing raw waveform error using an \(\ell_2\)-loss can be problematic, and propose a modified loss function that incorporates phase information for better audio quality.

---

### Motivation: Binaural Audio in AR/VR

Augmented Reality (AR) and Virtual Reality (VR) demand immersive audio that closely matches real-world listening experiences. Human spatial hearing relies on slight differences between signals arriving at the left and right ears (i.e., binaural signals) to determine direction, distance, and movement of sound sources. Accurately generating these signals enhances user orientation in virtual worlds and fosters immersion by matching visual cues with congruent acoustic cues. 

---

### Problem Statement

We have:

- A **mono (single-channel)** input signal \(x_{1:T}\) of length \(T\).
- A **binaural (two-channel)** target output \(\{y^{(l)}_{1:T}, y^{(r)}_{1:T}\}\) for the left and right ears.
- A **conditioning** signal \(c_{1:T} \in \mathbb{R}^{14}\) describing source and listener positions (3D) and orientations (as quaternions, each with 4 values).

The goal is to learn a function
\[
\hat{y}^{(l)}_t,\, \hat{y}^{(r)}_t = f(x_{t-\Delta : t} \,\vert\, c_{t-\Delta : t}),
\]
where \(\Delta\) is a temporal receptive field. Intuitively, the function should transform the mono audio into binaural audio given knowledge of where the source and the listener are in 3D space.

---

### Overview of the Proposed Approach

The system has two main components (see **Figure 1** in the original paper for a diagram):

1. **Neural Time Warping**  
   Aligns (warps) the mono signal into two separate streams, roughly corresponding to the left and right ears. The warping accounts for coarse time delays (e.g., interaural time differences and Doppler effects).

2. **Temporal ConvNet with Conditioned Hyper-Convolutions**  
   A deep network refines these warped signals to model more subtle effects, such as room reverberation, ear shape filtering, and dynamic head orientations.

#### 1. Neural Time Warping

In traditional signal processing, **Dynamic Time Warping (DTW)** has been used to align source and target signals. Here, the target binaural signal is not known beforehand, so the paper introduces a **learned neural warping** driven by the physical geometry of the source-listener setup.

- **Geometric Warping**: A first approximation warps the signal based on distance and the speed of sound:
  \[
    \rho^{(\text{geom})}_t = t \;-\; \frac{\|p_{\text{src}, t} - p_{\text{lstn}, t}\| \times \text{(audio sample rate)}}{\nu_{\text{sound}}},
  \]
  where \(p_{\text{src}, t}\) and \(p_{\text{lstn}, t}\) are the positions of the source and listener at time \(t\).

- **Neural Correction**: A small convolutional network (\(\text{WarpNet}\)) refines the geometric warping to handle nuances like head diffraction and interaural differences. The final warp field \(\rho\) is forced to be **monotonic and causal** through a special activation function.

- **Linear Interpolation**: Because \(\rho_t\) is generally non-integer, the warped signal is computed by interpolating the original signal at fractional indices.

The result is an initial left-ear signal and right-ear signal that are temporally aligned with the physical constraints of sound propagation.

#### 2. Conditioned Hyper-Convolutions

After warping, a **temporal convolutional network** refines the aligned signals. However, instead of using standard convolutional filters that stay fixed over time, the paper proposes **hyper-convolutions** whose weights **change dynamically** according to the conditioning variables \(c_{1:T}\).

- **Standard Approach**: A typical conditional convolution would just add transformations of the conditioning signal.  
- **Hyper-Convolution**: Predicts the convolutional weights as **functions of the conditioning signal**. In other words, the filters themselves become time-varying, guided by the source-listener configurations. This allows the model to capture complex, time-dependent transformations such as moving sound sources, shifting head orientations, and evolving room acoustics.

---

### The Loss Function: Mitigating Phase Errors

Minimizing the \(\ell_2\)-loss on raw waveforms often leads to **poor phase reconstruction**. While amplitude might fit well, large phase discrepancies remain, creating audible distortions (especially noticeable in speech). To address this, the paper adds a **phase-oriented term**:

\[
L(y, \hat{y}) = \underbrace{\| y - \hat{y} \|^2}_{\ell_2\text{-loss}} \;+\; \lambda \, L_{\text{phase}}\bigl(\text{STFT}(y), \text{STFT}(\hat{y})\bigr),
\]
where \(L_{\text{phase}}(\cdot)\) measures phase discrepancies in the short-time Fourier transform (STFT) domain. The authors show that this additional term significantly improves phase fidelity, leading to higher-quality binaural audio.

---

### Evaluation and Results

- **Dataset**:  
  A two-hour collection of paired mono and binaural recordings at 48 kHz with eight speakers (four male, four female) walking around a mannequin equipped with binaural microphones. Positions and orientations are tracked via an OptiTrack system.

- **Quantitative Metrics**:  
  The proposed method is compared against both classic DSP-based binauralization and other neural approaches. It achieves **lower amplitude and phase errors** and produces more realistic binaural signals.

- **Real-Time Performance**:  
  The model runs quickly enough for streaming applications (around 33 ms of latency on an NVidia Tesla V100). It comfortably meets real-time requirements for interactive AR/VR scenarios that typically refresh at 30–120 Hz.

#### Neural Time Warping Impact

Ablation studies indicate that geometric warping alone does not sufficiently align the signals. The **learned neural warp** further refines time shifts, substantially reducing phase error. Although the corrections might look visually small, the effect on perceptual quality can be significant.

#### Temporal HyperConv Network Impact

Beyond time alignment, additional layers are needed to handle reflections, reverberations, and the filtering effects of human ears (pinna, head). A simple linear amplitude adjustment yields only marginal improvement, emphasizing the importance of a **deep time-domain network**.

---

### Conclusion

This work demonstrates the first purely **data-driven, end-to-end** model for high-fidelity binaural audio synthesis from mono. Key takeaways include:

1. **Neural Time Warping** aligns signals based on geometry and learned corrections.
2. **Conditioned Hyper-Convolutions** capture complex, time-varying acoustic phenomena.
3. Incorporating a **phase term** in the loss function substantially improves perceptual audio quality.
4. The system operates in **real-time**, suitable for AR/VR or gaming scenarios.

Overall, the paper’s approach **outperforms classical DSP** methods in producing more realistic binaural sound and offers a promising direction for generating immersive audio in artificial environments.