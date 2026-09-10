from core.arguments import continual_args
from core.runner import run_continual_job

if __name__ == "__main__":
    run_continual_job(model="RMTPP", strategy="sequential", args=continual_args(replay=False), script="evaluate_continual_RMTPP_sequential.py")
