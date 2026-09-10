from core.arguments import diagnostic_args
from core.runner import run_diagnostic_job

if __name__ == "__main__":
    run_diagnostic_job(kind="law_recovery", args=diagnostic_args("law_recovery"), script="evaluate_dws_HM_law_recovery.py")
