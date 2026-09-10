from core.arguments import stationary_args
from core.runner import run_stationary_job

if __name__ == "__main__":
    run_stationary_job(dataset="retweet", model="THP", args=stationary_args(dws=False), script="evaluate_retweet_THP.py")
