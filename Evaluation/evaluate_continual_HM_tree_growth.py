from core.arguments import diagnostic_args
from core.runner import run_diagnostic_job

if __name__ == "__main__":
    run_diagnostic_job(kind="tree_growth", args=diagnostic_args("tree_growth"), script="evaluate_continual_HM_tree_growth.py")
