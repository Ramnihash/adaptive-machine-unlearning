# Adaptive Machine Unlearning for Privacy-Preserving Recommender Systems

An influence-aware machine unlearning framework for graph collaborative filtering recommender systems (LightGCN) on MovieLens-1M.

---

## 1. Project Objective and Research Questions

When users request the removal of their personal interaction history from a recommendation engine (under regulations such as GDPR Article 17 "Right to Erasure"), full retraining from scratch ($\theta^*$) is computationally prohibitive on graph neural networks.

This project investigates **adaptive, influence-guided machine unlearning**:
- **Core Research Question:** Can we select unlearning strategies adaptively based on a user's graph structural influence to balance privacy erasure, model utility for retained users, and computational efficiency?
- **Model Architecture:** LightGCN (3 layers, embedding dimension 64, BPR loss) with graph message passing over user-item interaction bipartite graphs.
- **Dataset:** MovieLens-1M (6,040 users, 3,706 items, ~1 million ratings).

---

## 2. System Architecture & Adaptive Routing

The framework evaluates user influence using the training graph topology and routes unlearning requests across three tiered strategies:

```
                      +-----------------------------+
                      |   Target Unlearning User    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Influence Degree Estimator  |
                      |   (Percentile / Min-Max)    |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         | (Influence < 0.33)        | (0.33 <= Influence <= 0.66)| (Influence > 0.66)
         v                           v                           v
+------------------+       +-------------------+       +--------------------+
| Tier: LOW        |       | Tier: MEDIUM      |       | Tier: HIGH         |
| Gradient Ascent  |       | Localized Update  |       | Targeted Retrain   |
| (Target param    |       | (1-hop / 2-hop    |       | (Full retrain from |
|  gradient ascent |       |  subgraph update, |       |  scratch on D*)    |
|  or reset)       |       |  freeze mask)     |       |                    |
+------------------+       +-------------------+       +--------------------+
```

### Strategy Summary:
1. **Low-Tier (`gradient_ascent`):** Low-degree/inactive users whose interactions have minimal influence on the graph. Applies target embedding reset and fast gradient ascent steps.
2. **Medium-Tier (`localized_update`):** Moderate-degree users. Extracts the 2-hop ego-network (1-hop items and neighboring users sharing those items), freezes unaffected node embeddings via a boolean mask, and fine-tunes only the affected subgraph using optimized negative sampling.
3. **High-Tier (`targeted_retraining`):** High-influence hub users whose removal substantially disrupts graph connectivity. Retrains from scratch on $D^* = D_{\text{train}} \setminus D_{\text{target}}$.

---

## 3. Dataset Preparation & Split Location

The dataset used is **MovieLens-1M** (`ml-1m`).
- **Raw Data Location:** `data/ml-1m/` (`ratings.dat`, `users.dat`, `movies.dat`).
- **Processed Split Location:** `data/processed_ml1m/`
  - `split_summary.json` (Fingerprint: `c61c9d4683ecce7f`)
  - `train.csv` (800,309 interactions)
  - `val.csv` (99,950 interactions)
  - `test.csv` (99,950 interactions)
  - Split configuration: Seed `42`, 10% validation ratio, 10% test ratio, per-user stratified split ensuring every user is present in train.

---

## 4. Environment Setup (Windows & Cross-Platform)

On Windows systems, Python 3.10+ is typically invoked via `py`. On Linux/macOS, use `python3`.

```powershell
# Create virtual environment (optional)
py -m venv .venv
.venv\Scripts\activate

# Install dependencies
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Core dependencies (`requirements.txt`):
- `torch>=2.0.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scipy>=1.10.0`
- `scikit-learn>=1.3.0`
- `pyyaml>=6.0`
- `pytest>=7.0.0`

---

## 5. Verification & Testing

The test suite covers unit and integration tests across data splitting, graph propagation, influence calculation, strategy correctness, audit safeguards, and privacy evaluation.

```powershell
# Run full test suite (85 tests)
py -m pytest -v

# Run targeted tests for localized update
py -m pytest tests/test_localized_update.py -v

# Run tests for audit safeguards and unlearning router
py -m pytest tests/test_audit_safeguards.py tests/test_unlearning_router.py -v
```

---

## 6. How to Run Experiments

The experiment pipeline is driven by `experiments/run_experiment.py` and configurable via YAML or command-line key-value overrides.

### 6.1 Configuration Dry-Run (Fast Verification)
Verifies configuration loading, stage validation, and metadata generation without heavy computation:
```powershell
py -m experiments.run_experiment --dry-run experiment.name=dry_run_test
```

### 6.2 Baseline Model Training ($\theta_0$)
Trains the full LightGCN model on the complete MovieLens-1M training split:
```powershell
py -m experiments.run_experiment --config experiments/configs/default.yaml experiment.name=ml1m_theta0 experiment.stages=[train_theta0] model.epochs=1
```

### 6.3 Adaptive Unlearning for Target User
Performs influence-guided unlearning starting from an existing $\theta_0$ checkpoint:
```powershell
py -m experiments.run_experiment --config experiments/configs/default.yaml experiment.name=ml1m_unlearning experiment.stages=[unlearn] evaluation.checkpoints.theta_0.path=experiments/results/20261024-094203-ml1m_theta0_benchmark-54c86dac/checkpoints/theta_0.pt unlearning.target_users=[39]
```

### 6.4 Model Evaluation Stage
Evaluates recommendation utility, forgetting metrics, and membership inference attacks:
```powershell
py -m experiments.run_experiment --config experiments/configs/default.yaml experiment.stages=[evaluate] evaluation.mode=theta0_vs_thetau evaluation.checkpoints.theta_0.path=experiments/results/20261024-094203-ml1m_theta0_benchmark-54c86dac/checkpoints/theta_0.pt evaluation.checkpoints.unlearned.path=experiments/results/20261025-173556-ml1m_localized_update_benchmark-58a008f0/checkpoints/theta_u.pt evaluation.target_user=39
```

---

## 7. Verified Experimental Evidence (MovieLens-1M)

All benchmark numbers are traced directly to saved experiment artifacts in `experiments/results/`.

### 7.1 Experiment Run Identifiers & Artifacts
| Experiment | Run ID | Artifact Path | Checkpoint |
| :--- | :--- | :--- | :--- |
| **Baseline ($\theta_0$)** | `20261024-094203-ml1m_theta0_benchmark-54c86dac` | `experiments/results/20261024-094203-.../results.json` | `theta_0.pt` |
| **Preliminary Unlearning** | `20261024-100144-ml1m_unlearning_preliminary-2f597dd8` | `experiments/results/20261024-100144-.../results.json` | `theta_u.pt` |
| **Optimized Unlearning** | `20261025-173556-ml1m_localized_update_benchmark-58a008f0` | `experiments/results/20261025-173556-.../results.json` | `theta_u.pt` |

### 7.2 Routing Decision for Target User 39
- **User Degree:** 76 interactions in `train.csv`.
- **Influence Score:** `0.4998` (computed via percentile ranking across 6,040 users).
- **Thresholds:** Low $\le 0.33$, High $> 0.66$.
- **Selected Tier:** `medium` $\rightarrow$ Strategy: **`localized_update`**.
- **Affected Subgraph Size:** 76 hop-1 items, 5,541 neighbor users (5,618 total affected nodes out of 9,746 graph nodes).

### 7.3 Computational Runtime & Optimization Speedup
- **Preliminary `localized_update` Runtime:** $2,901.07\text{ s}$ ($\approx 48.35\text{ min}$)
- **Optimized `localized_update` Runtime:** $1,022.44\text{ s}$ ($\approx 17.04\text{ min}$)
- **Measured Speedup:** **$2.84\times$ faster** ($64.76\%$ reduction in unlearning execution time).
- **Optimization Nature:** Replaced slow per-candidate rejection filtering with vectorized sorted-key search (`is_member_sorted`, fully equivalent to `torch.isin`) covering all positive interactions in $D^*_{\text{train}}$.

### 7.4 Recommendation Utility & Forgetting Metrics
Evaluated on MovieLens-1M with ranking cutoff $k=10$:

| Metric | Before Unlearning ($\theta_0$) | After Unlearning ($\theta_u$) | Delta ($\Delta$) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Retained Users Recall@10** | 0.074160 | 0.074241 | +0.000081 | Preserved utility for 6,039 non-target users |
| **Retained Users NDCG@10** | 0.129559 | 0.130049 | +0.000490 | Recommendation ranking preserved |
| **Target User Recall@10** | 0.000000 | 0.000000 | 0.000000 | Target test recall was 0.0 before unlearning |
| **Target User NDCG@10** | 0.000000 | 0.000000 | 0.000000 | Target test NDCG was 0.0 before unlearning |
| **Forgotten Item Hit Rate@10** | 0.200000 (20.0%) | 0.000000 (0.0%) | -0.200000 | Erased from top-10 recommendations |
| **Forgotten Item Mean Percentile** | 0.803541 (80.4%) | 0.189957 (19.0%) | -0.613584 | Forgotten items demoted down ranking catalog |

*Note on Target-User Test Utility:* Target user 39 already had Recall@10 = 0.0 and NDCG@10 = 0.0 prior to unlearning on test interactions. Therefore, test set utility metrics for user 39 do not capture forgetting; the true erasure evidence is provided by the direct ranking and percentile demotion of the 76 forgotten training items.

### 7.5 Privacy Evaluation: Membership Inference Attack (MIA)
Interaction-level MIA trained on target user interactions (76 forgotten training interactions vs. 20 held-out test/validation interactions):

| Metric | Before Unlearning ($\theta_0$) | After Unlearning ($\theta_u$) |
| :--- | :--- | :--- |
| **MIA AUC** | 0.5201 | 0.5181 |
| **95% Confidence Interval** | [0.3757, 0.6645] | [0.3760, 0.6641] |
| **MIA Advantage** | 0.000219 | 0.002395 |

**Critical Scientific Finding & Limitation:**
- Both before and after MIA AUC scores are near random chance ($0.50$).
- The 95% bootstrap confidence intervals ($[0.376, 0.664]$) broadly contain $0.50$.
- Because the interaction-level MIA had insufficient signal even on $\theta_0$, these results **do not establish empirical privacy improvement or certified deletion**. Complete forgetting cannot be claimed solely on this basis.

---

## 8. Implementation Status, Architectural Boundaries & Limitations

### Fully Implemented:
1. Data pipeline with caching safeguards, split integrity checks, and target-user interaction exclusion.
2. 3-layer LightGCN model with BPR loss and evaluation metrics (Recall@k, NDCG@k).
3. Influence calculation (degree, percentile rank, tier classification).
4. Adaptive unlearning router with configurable thresholds.
5. Three unlearning strategies: `gradient_ascent`, `localized_update` (with freeze-mask and vectorized negative sampling), and `targeted_retraining`.
6. Experiment runner, configuration overrides, checkpoint saving, and audit validation safeguards.

### Tested but Explicitly Rejected:
- **CSR Sparse Adjacency for CPU Training:** While mathematically equivalent, PyTorch CPU sparse CSR matrix multiplication incurred higher overhead during backward passes than COO in this workload. COO adjacency is retained.

### Known Limitations:
1. **Single-Target User:** The current automated pipeline executes unlearning on one target user per run.
2. **CPU Runtime:** LightGCN full graph message passing on CPU takes approximately 17 minutes for 1 epoch of localized update on MovieLens-1M due to graph aggregation across 5,500+ neighbors.
3. **No Differential Privacy Guarantee:** Heuristic unlearning methods (gradient ascent, localized fine-tuning) provide empirical forgetting of training items but do not furnish formal $(\epsilon, \delta)$-differential privacy bounds.
