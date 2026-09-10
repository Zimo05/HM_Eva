from core.arguments import diagnostic_args
from core.runner import run_diagnostic_job

if __name__ == "__main__":
    run_diagnostic_job(kind="consolidation", args=diagnostic_args("consolidation"), script="evaluate_continual_HM_consolidation.py")
