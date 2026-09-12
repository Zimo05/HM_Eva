from core.arguments import stationary_args
from core.runner import run_stationary_job

if __name__ == "__main__":
    run_stationary_job(dataset="taobao", model="AttNHP", args=stationary_args(), script="evaluate_taobao_AttNHP.py")
