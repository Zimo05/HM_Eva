from core.arguments import continual_args
from core.runner import run_continual_job

if __name__ == "__main__":
    run_continual_job(model="HM", strategy="no_merge_prune", args=continual_args(), script="evaluate_continual_HM_no_merge_prune.py")
