from core.arguments import continual_args
from core.runner import run_continual_job

if __name__ == "__main__":
    run_continual_job(model="HM", strategy="fixed_topology", args=continual_args(), script="evaluate_continual_HM_fixed_topology.py")
