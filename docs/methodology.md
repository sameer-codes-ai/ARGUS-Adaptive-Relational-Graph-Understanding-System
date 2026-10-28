# Chapter 5: Methodology and Algorithmic Formulations

**Project:** Adaptive Relational Graph Understanding for Surveillance (ARGUS)  
**Schema Version:** `threat5`  
**Authors:** Omkar Joshi, Sameer Kumar (Vellore Institute of Technology, Vellore)

---

## 5.1 End-to-End Algorithmic Blueprint

```
Algorithm 1: ARGUS Threat-5 Online Streaming Inference Pipeline
────────────────────────────────────────────────────────────────────────
Input : Video stream V = {I_1, I_2, ...}, Sampling threshold tau_s = 0.12,
        Max gap G_max = 4, Token retention ratio rho = 0.50, Memory bank M
Output: Online frame anomaly scores {s_t}, Threat Telemetry {T_t}

1 Initialize FIFO memory bank M <- zeros(16, D)
2 Initialize tracker state for IoU trajectory matching
3 for each incoming frame I_t do
4    // Module 1: Adaptive Sampling & Coverage Bounding
5    Compute motion heuristics D_t, G_t, H_t, S_t from grayscale(I_t)
6    M_t <- w_1*D_t + w_2*G_t + w_3*H_t + w_4*S_t
7    gap_count <- gap_count + 1
8    if M_t >= tau_s or gap_count >= G_max or is_first_or_last(t) then
9        Sample I_t as keyframe
10       gap_count <- 0
11       mu_t <- 1.0 + tanh(M_t)
12   else
13       Skip frame processing; emit interpolated score s_{t-1}
14       continue
15   end
16
17   // Module 2: Spatial Feature Extraction
18   F, X_v <- ConvNeXt_Tiny(I_t)  // F: feature map, X_v: 64 patch tokens
19
20   // Module 3: 5-Class Threat Graph Construction
21   Detections <- YOLO11n(I_t) with COCO-to-ARGUS threat mapping
22   Extract 12D kinematic-semantic node vectors v_i
23   Extract 64D RoI appearance features f_i^RoI via bilinear_grid_sample(F, boxes)
24   Construct combined node features \tilde{v}_i <- [v_i || f_i^RoI]  (76D)
25   Construct 10D relational threat edge attributes e_{ij} for d_{ij} < 0.42
26
27   // Module 4: Vectorized EdgeGAT
28   h_i^(0) <- W_node * \tilde{v}_i,  e_k <- W_edge * e_{ij}
29   for head h = 1 to H do
30       Compute attention logits: l_{ij}^h <- LeakyReLU(a_h^T [h_i || h_j || e_k])
31       alpha_{ij}^h <- scatter_softmax(l_{ij}^h, index=target_nodes)
32       m_i^h <- scatter_add(alpha_{ij}^h * (W_v^h h_j + W_ev^h e_k), index=target_nodes)
33   end
34   H' <- LayerNorm(Concat(m_i^1, ..., m_i^H) + W_res h_i^(0))
35
36   // Module 5: Topology-Aware Gated Cross-Attention Fusion
37   X_cross <- CrossAttention(Q=X_v, K=H', V=H')
38   g <- sigmoid(W_g [X_v || X_cross])
39   \tilde{X}_v <- LayerNorm(X_v + g * X_cross)
40
41   // Module 6: Graph-Guided Saliency Token Pruning
42   for each patch token x_k in \tilde{X}_v do
43       Score_k <- MLP_saliency(x_k) + max_i cos_sim(x_k, h_i')
44   end
45   \hat{X}_v <- Select top 50% tokens from \tilde{X}_v by Score_k (32 tokens)
46   z_t <- MeanPool(\hat{X}_v)  // 1xD frame summary
47
48   // Modules 7 & 8: Causal Temporal Transformer & FIFO Memory
49   Z_temporal <- Concat(M, z_t)  // (16 + 1) x D
50   Z_out <- TemporalTransformer(Z_temporal, causal_mask, modulation=mu_t)
51   M <- detach(Concat(M[1:], z_t))  // Sliding FIFO memory queue update
52
53   // Module 9: Anomaly Scoring & Telemetry Emission
54   s_t <- sigmoid(W_score * Z_out[-1])
55   T_t <- ExtractTelemetry(Detections, s_t, boxes, classes)
56   Emit s_t, T_t to AlertDispatcher
57 end
```

---

## 5.2 Mathematical Formulation by Module

### Module 1: Adaptive Temporal Sampling & Dynamic Modulation
Candidate frames are evaluated across four physical motion metrics:
$$D_t = \frac{1}{HW}\sum_{x,y} |I_t(x,y) - I_{t-1}(x,y)|$$
$$G_t = \frac{1}{HW}\sum_{x,y} \left(\left|\frac{\partial I_t}{\partial x}\right| + \left|\frac{\partial I_t}{\partial y}\right|\right)$$
$$H_t = -\sum_{b=1}^{32} p_b \log_2(p_b + 10^{-7}), \quad p_b = \frac{\text{hist}_b}{\sum \text{hist}}$$
$$S_t = 1.0 - \frac{\sum_{b=1}^{32} \min(p_b^{(t)}, p_b^{(t-1)})}{\sum_{b=1}^{32} \max(p_b^{(t)}, p_b^{(t-1)})}$$

Composite motion metric:
$$M_t = 0.40 D_t + 0.30 G_t + 0.15 H_t + 0.15 S_t$$

Sampling decision rule:
$$\text{Sample}(t) = \begin{cases} 1 & \text{if } M_t \ge 0.12 \lor (t - t_{\text{last}}) \ge 4 \lor t \in \{1, T\} \\ 0 & \text{otherwise} \end{cases}$$

Attention modulation scale:
$$\mu_t = 1.0 + \tanh(M_t) \in [1.0, 2.0]$$

---

### Module 2: ConvNeXt-Tiny Spatial Backbone
ConvNeXt-Tiny employs a $4 \times 4$ strided convolution stem with LayerNorm2d, followed by inverted bottleneck stages with $7 \times 7$ depthwise separable convolutions and GELU activations. An unaccelerated, zero-dependency pure-PyTorch implementation (`NativeConvNeXt`) is provided for edge deployment portability.

---

### Module 3: 5-Class Threat Graph Construction
Bounding boxes from YOLO11n are mapped into the 5-class ontology:
$$\mathcal{C} = \{\text{Person (0)}, \text{Vehicle (1)}, \text{Bag (2)}, \text{Weapon (3)}, \text{Other Object (4)}\}$$

The 12-dimensional kinematic-semantic vector:
$$v_i = [c_{x,i}, c_{y,i}, w_i, h_i, \dot{c}_{x,i}, \dot{c}_{y,i}, \mathbf{1}_{C_i=0}, \mathbf{1}_{C_i=1}, \mathbf{1}_{C_i=2}, \mathbf{1}_{C_i=3}, \mathbf{1}_{C_i=4}, \text{conf}_i]^T$$

Bilinear RoI visual appearance feature ($64\text{D}$):
$$f_i^{\text{RoI}} = \text{GridSample}\left(F, \, \text{BoxGrid}(B_i)\right) \in \mathbb{R}^{64}$$
$$\tilde{v}_i = [v_i \,\|\, f_i^{\text{RoI}}] \in \mathbb{R}^{76}$$

---

### Module 4: Vectorized EdgeGAT Formulation
For $N$ entities and $E$ edges:
$$h_i^{(0)} = W_{\text{node}} \tilde{v}_i \in \mathbb{R}^{128}, \quad \tilde{e}_{ij} = W_{\text{edge}} e_{ij} \in \mathbb{R}^{64}$$

For attention head $h \in \{1, \dots, 4\}$:
$$\text{logit}_{ij}^h = \text{LeakyReLU}\left(a_h^T [h_i^{(0)} \,\|\, h_j^{(0)} \,\|\, \tilde{e}_{ij}]\right)$$
$$\alpha_{ij}^h = \frac{\exp(\text{logit}_{ij}^h)}{\sum_{k \in \mathcal{N}_i} \exp(\text{logit}_{ik}^h)} \quad (\text{via } \texttt{scatter\_reduce\_})$$
$$m_i^h = \sum_{j \in \mathcal{N}_i} \alpha_{ij}^h \left(W_v^h h_j^{(0)} + W_{ev}^h \tilde{e}_{ij}\right) \quad (\text{via } \texttt{scatter\_add\_})$$
$$H_i' = \text{LayerNorm}\left(\text{Concat}(m_i^1, m_i^2, m_i^3, m_i^4) + W_{\text{res}} h_i^{(0)}\right)$$

---

### Module 5: Gated Cross-Attention Fusion
Visual tokens $X_v \in \mathbb{R}^{64 \times 256}$ query relational node embeddings $H' \in \mathbb{R}^{N \times 128}$:
$$Q = X_v W_Q, \quad K = H' W_K, \quad V = H' W_V$$
$$X_{\text{cross}} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
$$g = \sigma\left(W_g [X_v \,\|\, X_{\text{cross}}] + b_g\right)$$
$$\tilde{X}_v = \text{LayerNorm}\left(X_v + g \odot X_{\text{cross}}\right)$$

---

### Module 6: Graph-Guided Saliency Token Pruning
Each patch token $x_k \in \tilde{X}_v$ is evaluated via:
$$\text{Score}_k = \text{MLP}_{\text{saliency}}(x_k) + \max_{i \in \{1,\dots,N\}} \frac{x_k \cdot h_i'}{\|x_k\|_2 \|h_i'\|_2}$$

The top $K = \lfloor \rho \cdot N_p \rfloor = \lfloor 0.50 \times 64 \rfloor = 32$ tokens are retained:
$$z_t = \frac{1}{32}\sum_{k \in \text{TopK}} \tilde{x}_k \in \mathbb{R}^{256}$$

---

### Modules 7 & 8: Causal Temporal Transformer & Detached Memory
Given the current summary $z_t$ and historical memory $M \in \mathbb{R}^{16 \times 256}$:
$$Z = [M \,\|\, z_t] \in \mathbb{R}^{17 \times 256}$$
$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\left(\frac{Q K^T}{\sqrt{d_k}} \odot \mu_t\right) + \text{Mask}\right) V$$

Memory update (detached from computation graph):
$$M \leftarrow \text{detach}\left(\big[M_{2:16} \,\|\, z_t\big]\right)$$

---

### Module 9: Multi-Objective Optimization
$$\mathcal{L} = \mathcal{L}_{\text{BCE}}(s, y) + \lambda_1 \mathcal{L}_{\text{smooth}}(s) + \lambda_2 \mathcal{L}_{\text{sparse}}(s) + \lambda_3 \mathcal{L}_{\text{MIL}}(s^+, s^-)$$
where:
$$\mathcal{L}_{\text{BCE}} = -y \log(s + \epsilon) - (1-y)\log(1 - s + \epsilon)$$
$$\mathcal{L}_{\text{smooth}} = \frac{1}{T-1}\sum_{t=2}^T (s_t - s_{t-1})^2$$
$$\mathcal{L}_{\text{sparse}} = \frac{1}{T}\sum_{t=1}^T s_t$$
$$\mathcal{L}_{\text{MIL}} = \max\left(0, \, 1.0 - \max_{t} s_t^+ + \max_{t} s_t^-\right)$$
Optimal hyperparameters: $\lambda_1 = 0.10, \lambda_2 = 0.01, \lambda_3 = 0.50$.
