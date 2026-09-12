from core.arguments import continual_args
from core.runner import run_continual_job

if __name__ == "__main__":
    run_continual_job(model="AttNHP", strategy="replay", args=continual_args(replay=True), script="evaluate_continual_AttNHP_replay.py")
