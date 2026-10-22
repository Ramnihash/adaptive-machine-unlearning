# Evaluation and privacy measurement

All commands run from the repository root. Models are compared under identical
conditions: same split, same target user, same masking of seen items.

## Compare saved models

    python -m experiments.run_experiment experiment.name=eval "experiment.stages=[evaluate]" \
      "evaluation.checkpoints.theta_0.path=PATH/theta_0.pt" \
      "evaluation.checkpoints.theta_star.path=PATH/theta_star.pt" \
      "evaluation.checkpoints.bpr.path=PATH/bpr.pt" \
      "evaluation.checkpoints.shadow.path=PATH/shadow.pt"

Result: `experiments/results/<run id>/results.json` (stage `evaluate`), schema in
`evaluation/harness.py` (`RESULT_SCHEMA`). Checkpoints keep the run folder layout
(`<run>/checkpoints/x.pt` next to `<run>/results.json`) so run IDs and training
times are picked up automatically.

## Charts

    python -m visualization.plots experiments/results/<run id>/results.json figures/

## What is measured

| Question | Metric | Module |
|---|---|---|
| Is recommendation quality kept? | Recall@10, NDCG@10 on held-out test items of retained users | `models/evaluate.py` |
| Are the removed items still recommended? | share of target user's top-10 that are removed items; mean rank percentile | `evaluation/forgetting_metrics.py` |
| Does the model match full retraining θ*? | top-10 Jaccard (retained users and target user), score correlation | `evaluation/consistency_metrics.py` |
| Can an attacker tell the user was in training? | user-level membership attack with a shadow model | `privacy/mia_user_level.py` |
| Same question per interaction | member items vs held-out items (weak on LightGCN, AUC about 0.52) | `privacy/mia_dataset.py` |
| How long did it take? | training / retraining / unlearning seconds with run ID | `evaluation/harness.py` |

## Membership-attack protocol

See the module docstring of `privacy/mia_user_level.py`. In short: a shadow model is
trained like θ₀ without 10% of users; the attack learns to tell users that were in its
training data from users that were removed; it is then applied to θ₀, θ*, baselines and
the unlearned model for the forgotten user. Attack train, validation and test users are
disjoint and the target user is never used to fit the attack.

## Limits to state when reporting

* The shadow model assumes the attacker knows the training recipe (standard shadow-model assumption).
* A model retrained without a user leaves that user's embedding untrained, which the attack
  recognises easily. It therefore measures "does the model look as if the user was never trained",
  a strict test for approximate unlearning.
* A single target user gives a point estimate. Use several target users for final claims.
