"""Backward-compatible name for the standalone chronological HM runner.

New runs should use ``run_taobao_chronological_HM.py``. This alias no longer
creates or requires a synthetic CLProtocol for real Taobao windows.
"""

try:
    from .run_taobao_chronological_HM import main
except ImportError:  # Direct ``python Evaluation/script.py`` invocation.
    from run_taobao_chronological_HM import main


if __name__ == "__main__":
    main()
