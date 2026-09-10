# dws / HM / full

## Result

- accuracy: `0.16`
- brier_score: `0.874701518960709`
- cross_entropy: `2.078248387572664`
- ece_10bin: `0.03415957640856504`
- elapsed_seconds: `8.056132822297513`
- error_rate: `0.84`
- events: `400`
- events_per_second: `49.651614344402624`
- local_time_mae: `4.529171608984471`
- local_time_median_ae: `0.7926101684570312`
- local_time_rmse: `9.669751617936793`
- macro_f1: `0.07958888574560216`
- mean_episodic_residual_norm: `0.016895966604351997`
- mean_gated_episodic_residual_norm: `0.002206639484015588`
- mean_raw_episodic_residual_norm: `0.004223991651087999`
- mean_retrieval_alpha_mass: `1.0`
- mean_retrieval_alpha_per_visited_node: `1.0`
- mean_retrieval_effective_k: `1.0`
- mean_retrieval_null_alpha: `0.0`
- mean_retrieval_similarity: `0.9976038682460785`
- mean_retrieve_gate: `0.5224062134325504`
- memory_hit_fraction: `1.0`
- micro_f1: `0.16`
- nll_per_event: `6.337197383642197`
- nonempty_read_coverage_fraction: `1.0`
- owner_path_coverage_fraction: `1.0`
- perplexity: `565.2100226854122`
- raw_to_gated_residual_ratio: `0.5224062134325503`
- read_coverage_fraction: `1.0`
- sequence_macro_nll: `6.337197383642197`
- sequences: `2`
- top3_accuracy: `0.505`

## Reproduction command

```text
/home/lishuang/anaconda3/bin/python -m Train.Train --data-path /home/xinye/HawkesMemory_wfy_TEST/Evaluation/results/dws/HM/full/variant_13/seed_42/smoke-v3/prepared/canonical.csv --split-manifest /home/xinye/HawkesMemory_wfy_TEST/Evaluation/results/dws/HM/full/variant_13/seed_42/smoke-v3/prepared/split_manifest.json --split train --tree-init-depth 0 --checkpoint /home/xinye/HawkesMemory_wfy_TEST/Evaluation/results/dws/HM/full/variant_13/seed_42/smoke-v3/checkpoint/model.pt --best-checkpoint /home/xinye/HawkesMemory_wfy_TEST/Evaluation/results/dws/HM/full/variant_13/seed_42/smoke-v3/checkpoint/best.pt --seed 42 --device cuda:0 --epochs 1 --cold-start-epochs 1 --max-sequences 4 --max-events-per-sequence 16 --no-training-plots
```
