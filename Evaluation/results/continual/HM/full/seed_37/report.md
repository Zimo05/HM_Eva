# continual / HM / full

## Result

- tasks: `10`

## Reproduction command

```text
/home/xinye/miniconda3/envs/hm_eval/bin/python -m EvaluateCL --data-root /home/xinye/Benchmark/Datasets/CL/hm_continual_v2 --checkpoint-dir /home/xinye/Benchmark/Evaluation/results/continual/HM/full/seed_37/checkpoint --output-dir /home/xinye/Benchmark/Evaluation/results/continual/HM/full/seed_37/native --task-start 0 --task-end 9 --device cuda:1 --resume --eval-batch-size 64 --fwt-scratch-checkpoint /home/xinye/Benchmark/Evaluation/results/continual/HM/full/seed_37/checkpoint/initial_seed37.pt
```
