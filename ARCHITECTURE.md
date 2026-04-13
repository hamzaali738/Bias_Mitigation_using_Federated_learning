# Architecture & Design Patterns

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FEDERATED LEARNING SYSTEM                      │
└─────────────────────────────────────────────────────────────────────────┘

                         GLOBAL (SERVER-SIDE)
┌─────────────────────────────────────────────────────────────────────────┐
│  FederatedLFFServer                                                     │
│  ├─ Global Model B (Biased)        ────▶ Aggregates updates             │
│  ├─ Global Model D (Debiased)      ────▶ from all clients               │
│  ├─ ClientManager                  ────▶ Tracks client metadata         │
│  └─ Evaluation on Global Test Set  ────▶ Monitors global performance   │
└────────────────┬────────────────────────────────────────────────────────┘
                 │ Broadcasts global models
                 │ Receives client updates
                 │ Aggregates & synchronizes
                 │
    ┌────────────┼────────────┬──────────────┐
    │            │            │              │
    ▼            ▼            ▼              ▼
    
         LOCAL (CLIENT-SIDE) - Parallel Training
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ...
│  Client 1        │ │  Client 2        │ │  Client N        │
│                  │ │                  │ │                  │
│ FederatedClient  │ │ FederatedClient  │ │ FederatedClient  │
│ ├─ Model B_1     │ │ ├─ Model B_2     │ │ ├─ Model B_N     │
│ ├─ Model D_1     │ │ ├─ Model D_2     │ │ ├─ Model D_N     │
│ ├─ Local Data 1  │ │ ├─ Local Data 2  │ │ ├─ Local Data N  │
│ └─ Reweighting 1 │ │ └─ Reweighting 2 │ │ └─ Reweighting N │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Training Loop Flow

```
START
  │
  ├─ Load Datasets
  │  └─ Partition among clients
  │
  ├─ Create Server & Clients
  │  └─ Initialize global models
  │
  └─ For each communication round:
     │
     ├─ [SERVER] Send global models B_G, D_G to all clients
     │
     ├─ [PARALLEL CLIENTS] Local training:
     │  │
     │  ├─ For each local epoch:
     │  │  │
     │  │  └─ For each batch:
     │  │     │
     │  │     ├─ Forward pass: logit_b = B(x), logit_d = D(x)
     │  │     │
     │  │     ├─ Compute losses:
     │  │     │  ├─ loss_b_raw = CE(logit_b, y)
     │  │     │  ├─ loss_d_raw = CE(logit_d, y)
     │  │     │  └─ [EMA] Update sample loss exponential moving average
     │  │     │
     │  │     ├─ Compute adaptive weights (LOCAL REWEIGHTING):
     │  │     │  ├─ Detect conflicts: ari = argmax(logit_b) ≠ argmax(logit_d)
     │  │     │  ├─ Compute loss weights: w_loss = inverse(EMA loss)
     │  │     │  ├─ Compute conflict weights: w_conf = confidence if conflict
     │  │     │  └─ Blend: alpha ← progress_ratio → weights
     │  │     │           w = α·w_loss + (1-α)·w_conf
     │  │     │
     │  │     ├─ Weighted loss:
     │  │     │  ├─ loss_b = GCE(logit_b, y)
     │  │     │  └─ loss_d = CE(logit_d, y) * weights
     │  │     │
     │  │     └─ Backward: ∇ ← gradient(loss_b + loss_d)
     │  │           B ← B - lr*∇_B
     │  │           D ← D - lr*∇_D
     │  │
     │  └─ Send updated models to server
     │
     ├─ [SERVER] Aggregate client models:
     │  │
     │  ├─ Collect models from participating clients
     │  │
     │  ├─ Calculate aggregation weights (by data size):
     │  │  w_k = |data_k| / Σ|data_i|
     │  │
     │  ├─ Optional compression:
     │  │  model ← compress(model, compression_ratio)
     │  │
     │  ├─ Aggregate:
     │  │  B_G ← Σ w_k * B_k    (weighted average)
     │  │  D_G ← Σ w_k * D_k
     │  │
     │  └─ Update global models
     │
     ├─ [SERVER] Evaluate global models:
     │  ├─ Val accuracy, test accuracy
     │  ├─ Conflict accuracy (bias mitigation metric)
     │  └─ Save best models
     │
     └─ Log metrics and continue
        
END
```

## Component Interactions

### Client-Server Communication

```
┌─────────────┐                      ┌─────────────┐
│   SERVER    │◄────────────────────►│   CLIENT    │
└─────────────┘                      └─────────────┘
    │                                    │
    │ [1] send_model_to_client()        ▼
    ├──────────────────────────────►  load_model_state()
    │    {B_state, D_state, meta}       │
    │                                   ▼
    │                              local_train()
    │                              (adapt weights,
    │                               gradient descent)
    │                                   │
    │ [2] receive_client_update()      ◄┤
    │◄──────────────────────────────  get_model_state()
    │    {B_state, D_state, metrics}    ▼
    │                                   │
    ▼                                   │
aggregate_client_models()
(weighted averaging,
 optional compression)
    │
    ▼
evaluate_global_models()
```

### Local Reweighting Decision Tree

```
FOR EACH SAMPLE:
    │
    ├─ Are models in agreement?
    │  │
    │  ├─ YES (argmax(B) = argmax(D))
    │  │  │
    │  │  ├─ Loss is HIGH?
    │  │  │  └─► MEDIUM weight (hard aligned sample)
    │  │  │
    │  │  └─ Loss is LOW?
    │  │     └─► LOW weight (easy aligned sample)
    │  │
    │  └─ NO (argmax(B) ≠ argmax(D))
    │     │
    │     ├─ Training phase EARLY (0-50%)?
    │     │  └─► HIGH weight (conflict - learn bias)
    │     │
    │     └─ Training phase LATE (50-100%)?
    │        └─► MEDIUM-HIGH weight (conflict - fine-tune)
    │
    └─ Final weight = normalized combination
```

## Reweighting Methods Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                 REWEIGHTING METHOD COMPARISON                   │
├──────────────────┬──────────┬──────────────┬───────────────────┤
│ Method           │ Conflict │ Loss-based   │ When to Use       │
│                  │ Focus    │ Focus        │                   │
├──────────────────┼──────────┼──────────────┼───────────────────┤
│ conflict         │  100%    │   0%         │ Pure bias learning│
│ loss_based       │   0%     │ 100%         │ Standard learning │
│ uncertainty      │   0%     │ Variable     │ When confident    │
│ importance       │   0%     │ Variable     │ Gradient-aware    │
│ adaptive_local   │ Blend    │   Blend      │ RECOMMENDED       │
│                  │ (100%→0%)│ (0%→100%)    │ Best overall      │
└──────────────────┴──────────┴──────────────┴───────────────────┘

Adaptive blending schedule:
Early Training:    loss=30%   conflict=70%
Mid Training:      loss=50%   conflict=50%
Late Training:     loss=90%   conflict=10%
```

## Aggregation Strategies

```
╔════════════════════════════════════════════════════════════════╗
║                    AGGREGATION STRATEGIES                      ║
╚════════════════════════════════════════════════════════════════╝

FedAvg (Federated Averaging)
───────────────────────────
  B_G ← Σ_k (|D_k| / |D_total|) * B_k

  Pros:
    • Simple and fast
    • Works well with IID data
  
  Cons:
    • Struggles with non-IID data
    • Model drift between rounds


FedProx (Federated Proximal)
────────────────────────────
  B_G ← Σ_k w_k * B_k - μ(B_G - B_G_initial)

  Pros:
    • Better for non-IID data
    • Reduces model drift ("proximal term")
    • More stable convergence
  
  Cons:
    • Slightly more computation
    • Requires storing initial state
```

## Model Update Aggregation Detail

```
CLIENT MODELS (after local training):
   Client 1: B₁ = {W₁, b₁, ...}
   Client 2: B₂ = {W₂, b₂, ...}
   Client 3: B₃ = {W₃, b₃, ...}

DATA SIZES (used for weighting):
   Client 1: 1000 samples  → weight = 1000/3000 = 0.333
   Client 2: 500 samples   → weight = 500/3000  = 0.167
   Client 3: 1500 samples  → weight = 1500/3000 = 0.500

AGGREGATION:
   B_global = 0.333 * B₁ + 0.167 * B₂ + 0.500 * B₃
   
   For each parameter (e.g., first layer weights):
   W_global = 0.333 * W₁ + 0.167 * W₂ + 0.500 * W₃
```

## EMA (Exponential Moving Average) for Loss Tracking

```
EMA tracks historical sample losses for stability:

Initialize:
  ema_loss[i] = 0 for all samples i

Each batch:
  raw_loss[i] = cross_entropy(pred_i, label_i)
  ema_loss[i] ← α * ema_loss[i] + (1-α) * raw_loss[i]
  
  where α = 0.9 (smoothing factor)

This means:
  • Recent losses have 10% weight
  • Historical average has 90% weight
  • Prevents noise from affecting weights
  • Creates smooth reweighting curve
```

## Conflict Detection Mechanism

```
SAMPLE CLASSIFICATION:
─────────────────────

For each sample x with label y:

1. Forward through both models:
   logit_b = Biased_Model(x)
   logit_d = Debiased_Model(x)

2. Get predictions:
   pred_b = argmax(logit_b)
   pred_d = argmax(logit_d)

3. Classification:
   if pred_b = pred_d:
       ALIGNED SAMPLE (both models agree)
       └─► Lower weight
   else:
       CONFLICTING SAMPLE (models disagree)
       └─► Higher weight (biased vs debiased)

4. WHY CONFLICT MATTERS:
   - If both models make same mistake → easy sample
   - If they disagree → reveals bias/debiasing difference
   - High weight on conflicts → learn to identify bias patterns
```

## Communication Complexity

```
Without Compression:
───────────────────
Per round: 
  • Client → Server: 2 * (model_params)  [B and D models]
  • Server → Client: 2 * (model_params)
  
  Example: MLP with 500K params
           Per round = 4 MB (one direction)


With Compression (top-k):
─────────────────────────
Per round:
  • Keep only 10% of parameters
  • Client → Server: 2 * (0.1 * model_params)
  • Server → Client: 2 * (0.1 * model_params)
  
  Example: Same MLP with 0.1 ratio
           Per round = 0.4 MB (10x reduction!)
           
Negligible accuracy loss if applied carefully.
```

## Memory Layout on Each Client

```
┌────────────────────────────────────────────────┐
│            CLIENT MEMORY LAYOUT                │
├────────────────────────────────────────────────┤
│ Global Models (from server):                   │
│  ├─ Model B: θ_b              (e.g., 500K)   │
│  └─ Model D: θ_d              (e.g., 500K)   │
│                                               │
│ Client Models (for training):                 │
│  ├─ Local B: θ_b              (copy)         │
│  └─ Local D: θ_d              (copy)         │
│                                               │
│ Optimizer States:                             │
│  ├─ Adam_B: {m, v for each param}            │
│  └─ Adam_D: {m, v for each param}            │
│                                               │
│ Sample Tracking:                              │
│  ├─ Loss EMA: [loss_1, ..., loss_n]          │
│  ├─ Sample weights: [w_1, ..., w_n]          │
│  └─ Conflict scores: [c_1, ..., c_n]         │
│                                               │
│ Local Data:                                   │
│  └─ Batches of (image, label) pairs          │
└────────────────────────────────────────────────┘

Total: ~2MB per model + 50MB per batch (for images)
```

## Training Timeline Example

```
Round 1: Communication + Local Training
┌─────────────────────────────────────────────────┐
│  0s:   [SERVER] Broadcast global models        │
│  1s:   [CLIENTS] Receive + load models         │
│  2s:   [CLIENTS] Local training starts         │
│  2-60s: [Each client trains independently]     │
│         Client 1: compute reweights, gradient  │
│         Client 2: compute reweights, gradient  │
│         Client 3: compute reweights, gradient  │
│  60s:  [CLIENTS] Training complete            │
│  60s:  [CLIENTS] Send updates to server        │
│  62s:  [SERVER] Aggregates updates             │
│  63s:  [SERVER] Evaluates on test set          │
│  65s:  [LOGGING] Save metrics, models          │
└─────────────────────────────────────────────────┘
Total: ~65 seconds per round

For 10 rounds: ~10 minutes
For 20 rounds: ~20 minutes
```

## Data Flow Diagram

```
INPUT DATA
    │
    ├─ Get Dataset (CMNIST, images + attributes)
    │
    ├─ Extract Labels (target attribute, bias attribute)
    │
    ├─ Partition (IID or non-IID among clients)
    │
    └─►┌─────────────────────────────────────────┐
       │ [CLIENT TRAINING LOOP]                  │
       │                                         │
       │ Batch: (image, target_label, bias_label)
       │   │                                     │
       │   ├─ Model B processes → logit_b       │
       │   ├─ Model D processes → logit_d       │
       │   │                                     │
       │   ├─ Loss functions:                    │
       │   │  ├─ loss_b = GCE(logit_b, target)  │
       │   │  └─ loss_d = CE(logit_d, target)   │
       │   │                                     │
       │   ├─ Reweighting module:                │
       │   │  ├─ EMA update loss history        │
       │   │  ├─ Detect conflicts               │
       │   │  ├─ Compute sample weights         │
       │   │  └─ Normalize to sum = 1           │
       │   │                                     │
       │   ├─ Apply weights: CE * weights       │
       │   │                                     │
       │   └─ Backprop: loss.backward()         │
       │                                         │
       └─────────────────────────────────────────┘
              │
              └─►[MODEL AGGREGATION]
                   │
                   ├─ Weighted average (by data size)
                   ├─ Optional compression
                   └─ Global model update
                      │
                      └─►[EVALUATION]
                          ├─ Validation accuracy
                          ├─ Test accuracy  
                          ├─ CONFLICT ACCURACY ◄── Main metric
                          └─ Save best models
```

## Design Patterns Used

### 1. **Client-Server Pattern**
```python
# Server orchestrates, clients execute
class FederatedLFFServer:
    def send_model_to_client(self, client_id)
    def receive_client_update(self, client_id, states, metrics)
    def aggregate_client_models(self, updates)

class FederatedLFFClient:
    def load_model_state(self, state_dict)
    def local_train(self, epochs)
    def get_model_state(self)
```

### 2. **Strategy Pattern**
```python
# Multiple reweighting strategies
class LocalReweightingManager:
    def update_weights(self, method='loss_based|importance|uncertainty')
    
class AdaptiveLocalReweighting:
    def compute_adaptive_weights(self, progress_ratio)
    # Dynamically selects strategy based on phase
```

### 3. **Manager Pattern**
```python
# ClientManager tracks all clients
class ClientManager:
    def register_client(self, id, data_size)
    def get_aggregation_weights(self)
    
# Centralized client state management
```

### 4. **Factory Pattern**
```python
# Create models, loss functions dynamically
model_b = get_backbone('MLP', num_classes)
criterion = GeneralizedCELoss(q=0.7)
```

### 5. **Observer Pattern**
```python
# Server observes client metrics
server.receive_client_update(client_id, metrics)
# Logs and tracks for monitoring
```

---

## Summary

This architecture provides:
✅ **Scalability**: Easily add more clients  
✅ **Modularity**: Swap components (aggregation, reweighting)  
✅ **Efficiency**: Communication compression, EMA  
✅ **Robustness**: Non-IID data support, privacy options  
✅ **Interpretability**: Detailed logging, conflict tracking  

All designed for distributed bias mitigation with local adaptation!
