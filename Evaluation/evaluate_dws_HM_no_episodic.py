from core.arguments import stationary_args
from core.runner import run_stationary_job

if __name__ == "__main__":
    run_stationary_job(dataset="dws", model="HM", condition="no_episodic", args=stationary_args(dws=True), script="evaluate_dws_HM_no_episodic.py")
