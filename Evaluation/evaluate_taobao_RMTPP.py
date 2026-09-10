from core.arguments import stationary_args
from core.runner import run_stationary_job

if __name__ == "__main__":
    run_stationary_job(dataset="taobao", model="RMTPP", args=stationary_args(dws=False), script="evaluate_taobao_RMTPP.py")
