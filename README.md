# Bias Mitigation using Federated Learning 🚀

#-> Overview

This project explores **bias mitigation in machine learning** using **Federated Learning (FL)** combined with **Loss-based Feature Learning (LfF)**.

We compare standard federated training with bias-aware methods and analyze their performance, especially on **bias-conflicting samples**.



#-> Experiments

We implemented three different federated setups:

##->1. Baseline Federated Learning

* Single model per client
* Standard training using CrossEntropy loss
* Global aggregation using **FedAvg (equal weighting)**

---

##->2. LfF + FedAvg

* Two models per client:

  * **Biased model (model_b)**
  * **Debiased model (model_d)**
* Uses **loss-based reweighting** to focus on hard samples
* Aggregation using standard **FedAvg**

---

##->3. LfF + Weighted Aggregation

* Same LfF training setup
* Global aggregation uses **client validation accuracy as weights**
* Better-performing clients contribute more to global model

---

#-> Training Configuration

* Number of clients: **5**
* Communication rounds: **15**
* Local training: **3 epochs per round**
* Validation set: **200 samples** (balanced)

All results are obtained after **15 rounds with 3 local epochs per client per round**.

---

#-> Results

| Method         | Test Accuracy | Conflict Accuracy |
| -------------- | ------------- | ----------------- |
| Baseline       | 28.69%        | 20.82%            |
| LfF + FedAvg   | 53.05%        | 55.13%            |
| LfF + Weighted | **60.87%**    | **61.76%**        |

---

#-> Key Insights

* Baseline FL is heavily affected by dataset bias
* LfF significantly improves performance by focusing on difficult (bias-conflicting) samples
* Weighted aggregation further boosts performance by prioritizing better clients
* Conflict accuracy surpasses overall accuracy in LfF models, indicating strong bias mitigation

---

## ⚙️ How to Run

```bash
python train_federated.py --mode baseline
python train_federated.py --mode lff_avg
python train_federated.py --mode lff_weighted
```

---

## 📁 Project Structure

* `data/` → Dataset handling
* `federated/` → Client-server FL logic
* `module/` → Models and loss functions
* `train_federated.py` → Main training script

---

