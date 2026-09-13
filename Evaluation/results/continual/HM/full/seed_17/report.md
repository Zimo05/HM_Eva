# continual / HM / full

## Result

- tasks: `10`

## Reproduction command

```text
/home/xinye/miniconda3/envs/hm_eval/bin/python -m EvaluateCL --data-root /home/xinye/Benchmark/Datasets/CL/hm_continual_v2 --checkpoint-dir /home/xinye/Benchmark/Evaluation/results/continual/HM/full/seed_17/checkpoint --output-dir /home/xinye/Benchmark/Evaluation/results/continual/HM/full/seed_17/native --task-start 0 --task-end 9 --device cuda:0 --resume --eval-batch-size 64 --fwt-scratch-checkpoint /home/xinye/Benchmark/Evaluation/results/continual/HM/full/seed_17/checkpoint/initial_seed17.pt
```
