# Federated Learning with Local Reweighting for LFF

## Overview

This implementation integrates **Federated Learning (FL)** with **Local Reweighting** methods into the Learning From Failure (LFF) framework for bias mitigation.

### Key Components

1. **Federated Learning**: Distributed training across multiple clients with central server coordination
2. **Local Reweighting**: Client-side adaptive reweighting based on sample losses and model disagreement
3. **Dual Model Architecture**: Maintains biased and debiased models at both global and local levels
4. **Conflict Detection**: Identifies bias-conflicting samples for better reweighting

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FEDERATED SERVER                         │
│            Aggregates Models from All Clients               │
│  • FedAvg / FedProx aggregation strategies                  │
│  • Communication compression support                        │
│  • Global model evaluation                                  │
└────────────┬──────────────────────────────┬─────────────────┘
             │                              │
        Round N                        Round N+1
             │                              │
    ┌────────┴───────────┐        ┌────────┴───────────┐
    │                    │        │                    │
┌───┴────────────┐  ┌───┴────────────┐  ┌──────────────┴──┐
│   CLIENT 1     │  │   CLIENT 2     │  │   CLIENT N     │
│ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐ │
│ │ Model B    │ │  │ │ Model B    │ │  │ │ Model B    │ │
│ │ Model D    │ │  │ │ Model D    │ │  │ │ Model D    │ │
│ ├────────────┤ │  │ ├────────────┤ │  │ ├────────────┤ │
│ │ Local Data │ │  │ │ Local Data │ │  │ │ Local Data │ │
│ └────────────┘ │  │ └────────────┘ │  │ └────────────┘ │
│                │  │                │  │                │
│ Reweighting:   │  │ Reweighting:   │  │ Reweighting:   │
│ • Adaptive     │  │ • Adaptive     │  │ • Adaptive     │
│ • Loss-based   │  │ • Loss-based   │  │ • Loss-based   │
│ • Conflict     │  │ • Conflict     │  │ • Conflict     │
│   detection    │  │   detection    │  │   detection    │
└────────────────┘  └────────────────┘  └────────────────┘
```

## File Structure

```
copy_lff/
├── federated/
│   ├── __init__.py          # Package initialization
│   ├── client.py            # FederatedLFFClient implementation
│   ├── server.py            # FederatedLFFServer implementation
│   └── utils.py             # Aggregation and utility functions
├── local_reweighting.py     # Local reweighting strategies
├── lff_federated.py         # Main federated training script
└── [existing files]
```

## Local Reweighting Methods

### 1. **Adaptive Local Reweighting** (Recommended)

Combines multiple reweighting strategies adaptively based on training progress:

- **Early Phase (0-50%)**: Emphasizes **conflict detection**
  - Detects samples where biased and debiased models disagree
  - Higher weight for conflicting samples to improve robustness
  
- **Late Phase (50-100%)**: Emphasizes **loss-based reweighting**
  - Reweights by EMA of sample losses
  - Prioritizes hard samples with high loss

```python
reweighting = AdaptiveLocalReweighting(num_classes=10)
weight_dict = reweighting.compute_adaptive_weights(
    indices, logits_biased, logits_debiased, 
    losses_biased, losses_debiased, labels, 
    progress_ratio=0.5
)
weights = weight_dict['weights']  # Adaptive weights
```

### 2. **Conflict Detection Reweighting**

Specifically targets bias-conflict samples:

```python
conflict_manager = ConflictDetectionReweighting()
weights, conflict_mask = conflict_manager.detect_and_reweight(
    indices, logits_biased, logits_debiased, labels
)
# conflict_mask: True where models disagree (conflicting samples)
```

### 3. **Loss-Based Reweighting**

Inverse weighting by sample loss (harder samples get higher weight):

```python
loss_manager = LocalReweightingManager('loss_based')
weights = loss_manager.update_weights(indices, losses, labels)
```

### 4. **Importance Reweighting**

Based on gradient magnitude:

```python
importance_manager = LocalReweightingManager('importance')
weights = importance_manager.update_weights(indices, losses, labels)
```

## Usage

### Basic Training

```bash
python lff_federated.py \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 1 \
    --dataset cmnist \
    --percent 1pct \
    --model MLP \
    --aggregation_strategy fedavg \
    --reweighting_method adaptive_local
```

### With Non-IID Data Distribution

```bash
python lff_federated.py \
    --num_clients 5 \
    --data_distribution non-iid \
    --aggregation_strategy fedprox
```

### With Communication Compression

```bash
python lff_federated.py \
    --num_clients 5 \
    --compression_enabled \
    --compression_ratio 0.1  # Keep only 10% of parameters
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_clients` | 5 | Number of federated clients |
| `--aggregation_strategy` | fedavg | FedAvg or FedProx |
| `--data_distribution` | iid | IID or non-IID data split |
| `--reweighting_method` | adaptive_local | Reweighting strategy |
| `--num_rounds` | 10 | Communication rounds |
| `--local_epochs` | 1 | Local training epochs per round |
| `--compression_enabled` | False | Enable communication compression |
| `--compression_ratio` | 0.1 | Fraction of parameters to keep |
| `--q` | 0.7 | GCE loss parameter |

## Advanced Usage

### Custom Aggregation Strategy

```python
from federated.server import FederatedLFFServer

# Create server with FedProx
server = FederatedLFFServer(
    global_model_b, global_model_d,
    num_clients=10,
    aggregation_strategy='fedprox'  # Handles non-IID better
)
```

### Monitoring Training

```python
# Get server statistics
stats = server.get_statistics()
print(f"Communication rounds: {stats['communication_rounds']}")
print(f"Aggregation history: {stats['aggregation_history']}")

# Per-client training history
for client in clients:
    print(f"Client {client.client_id} training history:")
    print(client.training_history)
```

### Client-Side Validation

```python
# Validate client models locally
val_metrics = client.validate(val_loader)
print(f"Local Acc D: {val_metrics['acc_d_total']:.2f}%")
print(f"Local Acc B: {val_metrics['acc_b_total']:.2f}%")
```

## Training Process

```
For each communication round:
  1. Server sends global models (B, D) to all clients
  
  2. Each client performs local training:
     - Load global models
     - For each local epoch:
       - Compute losses from both models
       - Apply adaptive reweighting
       - Update models using weighted loss
  
  3. Clients send updated models to server
  
  4. Server aggregates:
     - Weighted average by data size
     - Updates global models
     - Optionally applies compression
  
  5. Server evaluates global models on test set
```

## Loss Functions

### Biased Model (Model B)
Uses **GeneralizedCELoss** to learn biased features:
```
loss_b = GCE(logit_b, label)
```

### Debiased Model (Model D)
Uses weighted **CrossEntropyLoss** with adaptive weights:
```
weights = compute_adaptive_weights(...)
loss_d = CE(logit_d, label) * weights
```

## Performance Metrics

The framework tracks:

- **Accuracy metrics**:
  - Overall accuracy
  - Conflict accuracy (main metric for bias mitigation)
  - Aligned accuracy

- **Loss metrics**:
  - Per-client local loss
  - Global loss trends

- **Federated metrics**:
  - Number of communication rounds
  - Model aggregation statistics
  - Client participation

## Example Output

```
============================================================
Communication Round 1/10
============================================================
[Server] Sending global models to clients...
[Clients] Starting local training...
[Client 0] Loss_B: 0.5234, Loss_D: 0.4892
[Client 1] Loss_B: 0.5156, Loss_D: 0.4756
[Client 2] Loss_B: 0.5312, Loss_D: 0.4923
[Client 3] Loss_B: 0.5089, Loss_D: 0.4687
[Client 4] Loss_B: 0.5201, Loss_D: 0.4834
[Server] Aggregating client models...
[Server] Evaluating global models...
[Validation] Acc_D: 92.45%, Acc_B: 88.32%
[Test] Acc_D: 91.87%, Acc_B: 87.65%
       Conflict D: 85.23%, Conflict B: 78.91%
```

## Implementation Details

### Adaptive Weighting Formula

$$w_i = \alpha_{conflict} \cdot w_{conflict,i} + \alpha_{loss} \cdot w_{loss,i}$$

Where:
- $\alpha_{conflict} = \max(0.1, 1.0 - progress\_ratio)$
- $\alpha_{loss} = progress\_ratio + 0.1$
- $w_{conflict,i}$ based on confidence in debiased model
- $w_{loss,i}$ inverse of EMA loss

### Conflict Detection

Samples are classified as conflicting if:
$$\arg\max(logit_b) \neq \arg\max(logit_d)$$

Conflicting samples receive higher weights for training.

## Troubleshooting

### Issue: OOM Error
- Reduce `--batch_size`
- Enable `--compression_enabled`
- Reduce `--num_clients`

### Issue: Slow Training
- Enable `--compression_enabled`
- Reduce `--local_epochs`
- Use simpler model (MLP instead of ResNet18)

### Issue: Poor Convergence
- Increase `--num_rounds`
- Increase `--local_epochs` (2-5)
- Reduce learning rate `--lr`
- Use `--aggregation_strategy fedprox` for non-IID data

## References

1. **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
2. **LFF**: Li et al., "Just Train it Again: On the Power of Early Restarting in Domain Generalization"
3. **Local Reweighting**: Inspired by multi-task learning and curriculum learning strategies

## Citation

If you use this implementation, please cite:

```bibtex
@software{federated_lff_2024,
  title={Federated Learning with Local Reweighting for LFF},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo}
}
```

## License

Same as the original LFF implementation.
