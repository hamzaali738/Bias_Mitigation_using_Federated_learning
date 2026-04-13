# Quick Start Guide - Federated LFF with Local Reweighting

## 5-Minute Setup

### Option 1: Run with Default Settings
```bash
python lff_federated.py --num_clients 3 --num_rounds 5 --local_epochs 1
```

### Option 2: Optimized for Your Hardware
```bash
# For limited memory
python lff_federated.py \
    --num_clients 3 \
    --batch_size 32 \
    --compression_enabled \
    --compression_ratio 0.1

# For powerful GPU
python lff_federated.py \
    --num_clients 10 \
    --batch_size 256 \
    --num_rounds 20 \
    --local_epochs 2
```

### Option 3: Realistic Federated Scenario
```bash
python lff_federated.py \
    --num_clients 10 \
    --data_distribution non-iid \
    --aggregation_strategy fedprox \
    --reweighting_method adaptive_local
```

---

## What Each Component Does

### Server
```
┌──────────────────────────────────┐
│  Global Models (B, D)            │
│                                  │
│  1. Receives from clients        │
│  2. Aggregates parameters        │
│  3. Sends back to all clients    │
│  4. Evaluates on test set        │
└──────────────────────────────────┘
```

### Clients
```
┌──────────────────────────────────┐
│  Client K                        │
│                                  │
│  1. Gets global models           │
│  2. Trains locally for iterations│
│  3. Applies reweighting          │
│  4. Sends back updated models    │
└──────────────────────────────────┘
```

### Local Reweighting
```
For each sample:
  if biased_model ≠ debiased_model:
    weight = high  (conflict sample)
  else if loss is high:
    weight = moderate  (hard sample)
  else:
    weight = low  (easy aligned sample)
```

---

## Key Parameters Explained

| Parameter | What It Does | Recommended |
|-----------|--------------|-------------|
| `--num_clients` | Number of participants | 3-10 |
| `--num_rounds` | Communication rounds | 10-20 |
| `--local_epochs` | Local training per round | 1-2 |
| `--aggregation_strategy` | FedAvg or FedProx | FedProx (more stable) |
| `--data_distribution` | IID or non-IID | non-iid (realistic) |
| `--reweighting_method` | Weighting strategy | adaptive_local (best) |
| `--compression_enabled` | Reduce communication | True (for >20 clients) |

---

## Expected Output

```
============================================================
Communication Round 1/10
============================================================
[Server] Sending global models to clients...
[Clients] Starting local training...
[Client 0] Loss_B: 0.5234, Loss_D: 0.4892
[Client 1] Loss_B: 0.5156, Loss_D: 0.4756
...
[Server] Aggregating client models...
[Server] Evaluating global models...
[Validation] Acc_D: 92.45%, Acc_B: 88.32%
[Test] Acc_D: 91.87%, Acc_B: 87.65%
       Conflict D: 85.23%, Conflict B: 78.91%
```

### Key Metrics
- **Acc_D**: Overall accuracy of debiased model (main goal)
- **Conflict**: Accuracy on bias-conflicting samples (shows bias mitigation)
- **Acc_B**: Biased model accuracy (typically highest on conflict)

---

## Troubleshooting

### 🔴 Out of Memory
```bash
# Reduce batch size
python lff_federated.py --batch_size 16

# Reduce clients
python lff_federated.py --num_clients 2

# Enable compression
python lff_federated.py --compression_enabled
```

### 🟠 Slow Training
```bash
# Reduce local epochs
python lff_federated.py --local_epochs 1

# Reduce data per client
python lff_federated.py --num_clients 3

# Use simpler model
python lff_federated.py --model MLP
```

### 🟡 Accuracy Not Improving
```bash
# More communication rounds
python lff_federated.py --num_rounds 20

# Non-IID data (more challenging)
python lff_federated.py --data_distribution non-iid

# Longer local training
python lff_federated.py --local_epochs 2

# Better aggregation for non-IID
python lff_federated.py --aggregation_strategy fedprox
```

### 🟢 Already Training Well
Experiment with:
```bash
# Try conflict detection only
python lff_federated.py --reweighting_method conflict

# Try loss-based only
python lff_federated.py --reweighting_method loss_based

# Add compression for efficiency
python lff_federated.py --compression_enabled --compression_ratio 0.2
```

---

## Monitoring Results

Check results in: `log/{dataset}/{exp}/result/`

### Files Generated
- `global_model_b_round*.pt` - Biased model checkpoints
- `global_model_d_round*.pt` - Debiased model checkpoints
- `federated_training_log.csv` - All metrics

### Plot Training Curves
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('log/cmnist/federated_lff/result/federated_training_log.csv')

# Plot accuracy trends
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['round'], df['test_acc_d'], label='Model D')
plt.plot(df['round'], df['test_acc_b'], label='Model B')
plt.xlabel('Round'), plt.ylabel('Accuracy %')
plt.legend(), plt.title('Overall Accuracy')

plt.subplot(1, 2, 2)
# Only plot unique rounds
df_unique = df.drop_duplicates('round')
plt.plot(df_unique['round'], df_unique['test_acc_d'], marker='o')
plt.xlabel('Round'), plt.ylabel('Conflict Accuracy %')
plt.title('Bias Mitigation (Conflict Accuracy)')
plt.tight_layout()
plt.show()
```

---

## Example Scenarios

### Scenario 1: Quick Test (2 min)
```bash
python lff_federated.py \
    --num_clients 2 --num_rounds 2 --local_epochs 1
```

### Scenario 2: Standard Training (30 min)
```bash
python lff_federated.py \
    --num_clients 5 --num_rounds 10 --local_epochs 1
```

### Scenario 3: Production Ready (2 hours)
```bash
python lff_federated.py \
    --num_clients 10 \
    --num_rounds 20 \
    --local_epochs 2 \
    --data_distribution non-iid \
    --aggregation_strategy fedprox \
    --compression_enabled
```

### Scenario 4: Research (Variable)
```bash
# Test different aggregation strategies
for strategy in fedavg fedprox; do
    python lff_federated.py \
        --num_clients 5 \
        --aggregation_strategy $strategy \
        --exp federated_lff_$strategy
done

# Test different reweighting methods
for method in loss_based conflict adaptive_local; do
    python lff_federated.py \
        --num_clients 5 \
        --reweighting_method $method \
        --exp federated_lff_$method
done
```

---

## Understanding Reweighting

### What is "Reweighting"?

In normal training, each sample contributes equally. With reweighting, we adjust how much each sample affects the model:

```
Normal Training:
loss = sum(loss_per_sample) / num_samples

With Reweighting:
loss = sum(weight_i * loss_i) / sum(weights)
```

### Local Reweighting Methods

1. **Conflict Detection** (Best for bias mitigation)
   - High weight: Sample where biased and debiased models disagree
   - Low weight: Sample where both models agree
   - Focuses on learning bias patterns

2. **Loss-Based** (Standard approach)
   - High weight: Sample with high loss (hard samples)
   - Low weight: Sample with low loss (easy samples)
   - Focuses on learning difficult cases

3. **Adaptive** (Combines both)
   - Early training: More conflict detection
   - Late training: More loss-based
   - Best overall for our problem

---

## Files Modified/Created

### New Files
- `federated/` - Federated learning framework
- `local_reweighting.py` - Reweighting algorithms
- `lff_federated.py` - Main training script
- `federated_example.py` - Code examples
- `FEDERATED_README.md` - Full documentation
- `INTEGRATION_SUMMARY.md` - What was added
- This file - Quick start

### Original Files (Unchanged)
Your existing LFF code remains unchanged and compatible!
- `lff_production.py` - Original LFF training
- `train.py` - Original training script
- All models, losses, and datasets

---

## Common Commands

```bash
# View available options
python lff_federated.py --help

# Run with verbose output (recommended for first time)
python lff_federated.py --num_clients 3 --num_rounds 2 | tee training.log

# Save configuration
cat > config.txt << EOF
python lff_federated.py \\
    --num_clients 5 \\
    --num_rounds 10 \\
    --data_distribution non-iid \\
    --aggregation_strategy fedprox \\
    --compression_enabled
EOF

# Run from config
bash config.txt
```

---

## Next Steps

1. ✅ **Run your first federated training**
   ```bash
   python lff_federated.py --num_clients 3 --num_rounds 5
   ```

2. ✅ **Check the results**
   ```bash
   ls -la log/cmnist/federated_lff/result/
   ```

3. ✅ **Try different settings**
   ```bash
   python lff_federated.py --data_distribution non-iid --aggregation_strategy fedprox
   ```

4. ✅ **Read full documentation**
   - See `FEDERATED_README.md` for detailed explanations
   - See `INTEGRATION_SUMMARY.md` for architecture details

5. ✅ **Customize for your needs**
   - Modify reweighting method based on your data
   - Adjust number of clients for your setup
   - Enable compression if communication is bottleneck

---

## Support

If something doesn't work:

1. Check error message in terminal
2. Look up in Troubleshooting section above
3. Review the parameters in `python lff_federated.py --help`
4. Check `FEDERATED_README.md` for detailed documentation
5. Review `federated_example.py` for code examples

---

## Key Insights

- **Federated Learning** = Training across multiple clients + server aggregation
- **Local Reweighting** = Giving different importance to different samples
- **Bias Mitigation** = Focusing on samples where the bias appears
- **Conflict Detection** = Finding where biased model makes different predictions

By combining all three, you get a powerful distributed system that:
✅ Mitigates bias across datasets  
✅ Works on decentralized data  
✅ Maintains privacy (with differential privacy)  
✅ Reduces communication costs  
✅ Improves fairness and robustness  

---

Happy training! 🚀
