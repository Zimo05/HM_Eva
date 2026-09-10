# Hawkes Memory Tree Evaluation

Evaluation regime: **transductive**.

## Predictive quality

- **full_frozen**: NLL/event=6.337197, ACC=0.1600, macro-F1=0.0796, local-time MAE=4.5292
- **no_episodic**: NLL/event=6.355350, ACC=0.1700, macro-F1=0.0930, local-time MAE=4.5294
- **no_working**: NLL/event=6.346298, ACC=0.1525, macro-F1=0.0591, local-time MAE=4.5290
- **semantic_only**: NLL/event=6.364576, ACC=0.1525, macro-F1=0.0589, local-time MAE=4.5293
- **full_online**: NLL/event=6.335955, ACC=0.1625, macro-F1=0.0833, local-time MAE=4.5292

## Memory contribution

- **episodic_gain**: mean ΔNLL=+0.018153, improved events=35.2%, 95% CI=[0.013594068735837937, 0.02259098455309868]
- **working_gain**: mean ΔNLL=+0.009100, improved events=49.0%, 95% CI=[0.006279501616954803, 0.012014080584049226]
- **total_memory_gain**: mean ΔNLL=+0.027378, improved events=43.0%, 95% CI=[0.020834325104951857, 0.03397330567240715]
- **online_vs_frozen_gain**: mean ΔNLL=+0.001243, improved events=15.8%, 95% CI=[0.0007574854791164398, 0.001763540506362915]

## Controller utility diagnostics

- **retrieve**: Spearman=0.5537861433721222, ROC-AUC=0.9997535529450423, PR-AUC=0.9995566278046253, regret/event=0.000802.
- **adapt**: Spearman=-0.35129725979199333, ROC-AUC=0.31285225442834136, PR-AUC=0.4000570647447022, regret/event=0.001275.
- **write**: Spearman=0.6889194139194139, ROC-AUC=None, PR-AUC=None, regret/event=0.065037.
  Write ranking: pair-acc=0.9653, NDCG@4=0.7738, Top-4=0.642199, random-Top-4=0.390406, regret/sequence=0.203657.
- **split**: unavailable (No Sleep-evaluated Split labels in checkpoint replay.).
- Research thresholds overall: **FAIL**; details: `{'mean_writes_le_2': False, 'write_budget_utilization_le_0_5': False, 'memory_gain_target': True, 'adapt_spearman_ge_0_35': False, 'retrieve_spearman_gt_0_10': True, 'write_roc_auc_gt_0_60': False, 'write_ranking_spearman_gt_0_10': True, 'write_pairwise_accuracy_gt_0_55': True, 'write_top4_uplift_positive': True, 'write_ranking_better_than_baseline': True, 'write_positive_negative_ge_20': False, 'online_harmful_fraction_lt_0_45': True, 'full_online_gain_gt_0': True, 'frozen_acc_macro_f1_not_below_v4': None, 'frozen_controller_output_unchanged': None, 'retrieve_policy_unchanged': None, 'adapt_policy_unchanged': None}`.

## Memory call-chain diagnostics

- Read/nonempty/owner-path coverage: 100.0% / 100.0% / 100.0%.
- Raw→gated residual ratio: 0.5224; mean retrieve gate: 0.5224.
- Write funnel: argmax=0, candidate=53, gate-pass=400, priority-pass=0, window-complete=0, accepted=0.
- Accepted-write reuse/beneficial rate: 0.0% / 0.0%.
- Retrieval/NLL correlations: `{'alpha_mass': {'pearson': None, 'spearman': None}, 'alpha_per_visited_node': {'pearson': None, 'spearman': None}, 'similarity': {'pearson': 0.030790281159638985, 'spearman': -0.06436663612740678}, 'raw_residual_norm': {'pearson': None, 'spearman': None}, 'gated_residual_norm': {'pearson': 0.8935072624679288, 'spearman': 0.5537861433721222}, 'retrieve_gate': {'pearson': 0.8935072624679296, 'spearman': 0.5537861433721222}}`.

## Tree and memory health

- Nodes/leaves: 1/1; max depth: 0.
- Owner top-1 share: 100.0%; effective owners: 1.00.
- Memory rows: 1 (internal=0, leaf=1).
- Label diagnostics: `{'available': False, 'cluster_count': 1, 'represented_clusters': [0], 'note': 'Purity/NMI/ARI skipped because cluster coverage is incomplete.'}`.

## Warnings

- Routing collapse risk: one owner receives more than 80% of sequences/events.

> Time MAE/RMSE use the model's documented local-constant-rate approximation.
