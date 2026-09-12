from core.arguments import continual_args
from core.runner import run_continual_job

if __name__ == "__main__":
    run_continual_job(model="S2P2", strategy="replay", args=continual_args(replay=True), script="evaluate_continual_S2P2_replay.py")
