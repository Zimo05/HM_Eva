from core.arguments import continual_args
from core.runner import run_continual_job

if __name__ == "__main__":
    run_continual_job(model="AttNHP", strategy="joint", args=continual_args(replay=False), script="evaluate_continual_AttNHP_joint.py")
