# AcousticIQ Research Methodology & Mathematical Formulation

## 1. Research Motivation & Problem Formulation

In industrial predictive maintenance, anomalous machine sounds frequently precede catastrophic mechanical failure by hours or days. However:
1. Anomalous machine events are extremely rare in production, making standard supervised classification impractical due to extreme class imbalance.
2. Machine operating environments suffer from variable background ambient noise (pumps, compressors, forklifts) and varying SNR conditions.
3. Standard Log-Mel filterbanks non-linearly compress high-frequency acoustic bands, potentially obscuring delicate harmonic friction scrapes and micro-impact cavitation bursts.

AcousticIQ formulates anomaly detection as an **unsupervised acoustic representation reconstruction problem**.

## 2. Feature Representations

### Branch A: Log-Mel Spectrogram
Given audio $x[n]$ sampled at $f_s = 16,000 \text{ Hz}$, the Short-Time Fourier Transform (STFT) with Hann window $w[n]$, FFT length $N_{fft} = 1024$, and hop length $R = 512$ is computed as:
$$X(k, m) = \sum_{n=0}^{N_{fft}-1} x[n + mR] w[n] e^{-j \frac{2\pi k n}{N_{fft}}}$$
The power spectrogram is $P(k, m) = |X(k, m)|^2$. Applying the triangular Mel filterbank basis matrix $M \in \mathbb{R}^{B \times (1 + N_{fft}/2)}$ with $B = 64$ bins:
$$S_{mel}(b, m) = \sum_{k=0}^{N_{fft}/2} M(b, k) P(k, m)$$
Log compression yields:
$$L_{mel}(b, m) = 10 \log_{10} \left( \max(S_{mel}(b, m), 10^{-10}) \right)$$

### Branch B: Inverse-Mel-Inspired Linear Spectral Reconstruction
Because standard Mel filterbanks heavily integrate upper frequencies into wide triangular bands, fine harmonic resonances are attenuated. We compute the Moore-Penrose pseudo-inverse of the Mel basis matrix:
$$M^\dagger = (M^T M + \lambda I)^{-1} M^T$$
The reconstructed linear-frequency power distribution is:
$$\hat{P}_{lin} = M^\dagger S_{mel}$$
A linear downsampling operator $W_{lin} \in \mathbb{R}^{B \times (1 + N_{fft}/2)}$ projects $\hat{P}_{lin}$ into a complementary feature tensor $L_{inv}(b, m)$ with identical shape $(64, T)$, preserving complementary high-frequency harmonic structure.

## 3. Baseline Deep Convolutional Autoencoder

The baseline autoencoder $f_{\theta} = d_{\theta_d} \circ e_{\theta_e}$ maps $L_{mel} \in \mathbb{R}^{1 \times 64 \times T}$ into a compressed latent space $z \in \mathbb{R}^{128 \times 8 \times T/8}$, then reconstructs $\hat{L}_{mel}$.
Trained exclusively on normal operating audio using Mean Squared Error:
$$\mathcal{L}_{base}(\theta) = \frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{m=1}^{T} (L_{mel}(b, m) - \hat{L}_{mel}(b, m))^2$$

The anomaly score is the reconstruction error:
$$S_{base}(x) = \mathcal{L}_{base}$$

## 4. Proposed Contribution: Dual-Branch Adaptive Feature Fusion

The proposed research architecture features two parallel encoding pathways:
- Mel encoder: $z_{mel} = e_{mel}(L_{mel})$
- Inverse-Mel encoder: $z_{inv} = e_{inv}(L_{inv})$

### Dynamic Latent Gating Network
Global Average Pooling (GAP) extracts compact latent descriptors:
$$\bar{z}_{mel} = \text{GAP}(z_{mel}) \in \mathbb{R}^{128}, \quad \bar{z}_{inv} = \text{GAP}(z_{inv}) \in \mathbb{R}^{128}$$
A non-linear gating MLP computes the dynamic reliability coefficient $\alpha$:
$$\alpha = \sigma \left( W_2 \cdot \text{LeakyReLU}(W_1 [\bar{z}_{mel} \parallel \bar{z}_{inv}] + b_1) + b_2 \right) \in [0, 1]$$

The fused bottleneck representation is:
$$z_{fused} = \alpha \cdot z_{mel} + (1 - \alpha) \cdot z_{inv}$$

Parallel decoders reconstruct both representations: $\hat{L}_{mel} = d_{mel}(z_{fused})$ and $\hat{L}_{inv} = d_{inv}(z_{fused})$.

### Adaptive Anomaly Score
The composite anomaly score dynamically balances both representations:
$$S_{prop}(x) = \alpha \cdot \text{MSE}(L_{mel}, \hat{L}_{mel}) + (1 - \alpha) \cdot \text{MSE}(L_{inv}, \hat{L}_{inv})$$

## 5. Threshold Selection Protocol

To avoid data leakage into test evaluation:
1. Decision threshold $\theta$ is calibrated strictly on the validation set using the 95th percentile of normal operational reconstruction scores:
$$\theta = \text{Percentile}_{95} \left( \{ S(x_i) \mid x_i \in \mathcal{D}_{val}^{normal} \} \right)$$
2. Test recordings are classified using this frozen threshold:
$$\text{Decision}(x) = \begin{cases} \text{Normal Pattern}, & S(x) < \theta \\ \text{Potential Acoustic Anomaly}, & S(x) \ge \theta \end{cases}$$
