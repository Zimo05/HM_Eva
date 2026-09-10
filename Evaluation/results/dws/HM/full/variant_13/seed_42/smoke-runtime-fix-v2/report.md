# dws / HM / full

## Result

- accuracy: `0.16`
- brier_score: `0.8747015205681835`
- cross_entropy: `2.0782483957672406`
- ece_10bin: `0.03415957462042571`
- elapsed_seconds: `6.584215299997595`
- error_rate: `0.84`
- events: `400`
- events_per_second: `60.75135483497116`
- local_time_mae: `4.529171573221683`
- local_time_median_ae: `0.7926101684570312`
- local_time_rmse: `9.669751510428048`
- macro_f1: `0.07958888574560218`
- mean_episodic_residual_norm: `0.01689602993428707`
- mean_gated_episodic_residual_norm: `0.002206647697410989`
- mean_raw_episodic_residual_norm: `0.004224007483571768`
- mean_retrieval_alpha_mass: `1.0`
- mean_retrieval_alpha_per_visited_node: `1.0`
- mean_retrieval_effective_k: `1.0`
- mean_retrieval_null_alpha: `0.0`
- mean_retrieval_similarity: `0.9976038599014282`
- mean_retrieve_gate: `0.522406199797988`
- memory_hit_fraction: `1.0`
- micro_f1: `0.16`
- nll_per_event: `6.337197285890579`
- nonempty_read_coverage_fraction: `1.0`
- owner_path_coverage_fraction: `1.0`
- perplexity: `565.209967435221`
- raw_to_gated_residual_ratio: `0.522406199797988`
- read_coverage_fraction: `1.0`
- sequence_macro_nll: `6.337197285890579`
- sequences: `2`
- top3_accuracy: `0.505`

## Reproduction command

```text
D:\Anaconda3\envs\myenv\python.exe -m Train.Train --data-path D:\Files\School\lishuang\HawkesMemory_wfy\Evaluation\results\dws\HM\full\variant_13\seed_42\smoke-runtime-fix-v2\prepared\canonical.csv --split-manifest D:\Files\School\lishuang\HawkesMemory_wfy\Evaluation\results\dws\HM\full\variant_13\seed_42\smoke-runtime-fix-v2\prepared\split_manifest.json --split train --tree-init-depth 0 --checkpoint D:\Files\School\lishuang\HawkesMemory_wfy\Evaluation\results\dws\HM\full\variant_13\seed_42\smoke-runtime-fix-v2\checkpoint\model.pt --best-checkpoint D:\Files\School\lishuang\HawkesMemory_wfy\Evaluation\results\dws\HM\full\variant_13\seed_42\smoke-runtime-fix-v2\checkpoint\best.pt --seed 42 --device cpu --epochs 1 --cold-start-epochs 1 --max-sequences 4 --max-events-per-sequence 16 --no-training-plots
```
