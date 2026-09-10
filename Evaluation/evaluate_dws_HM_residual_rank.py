from core.arguments import diagnostic_args
from core.runner import run_diagnostic_job

if __name__ == "__main__":
    run_diagnostic_job(kind="residual_rank", args=diagnostic_args("residual_rank"), script="evaluate_dws_HM_residual_rank.py")
