from core.arguments import diagnostic_args
from core.runner import run_diagnostic_job

if __name__ == "__main__":
    run_diagnostic_job(kind="frontier", args=diagnostic_args("frontier"), script="evaluate_dws_HM_frontier.py")
