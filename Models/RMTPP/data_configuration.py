"""Dataset adapters for the RMTPP implementation provided by EasyTPP."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _data_configuration_common import (  # noqa: E402
    ensure_directory,
    infer_dataset_root,
    infer_output_root,
    load_all_dws,
    load_covid_policy_tracker_splits,
    load_standard_splits,
    write_json,
)


class DataConfiguration:
    """Convert every benchmark to EasyTPP JSON for the neural RMTPP model.

    The repository in ``Models/RMTPP`` is PtPack rather than the neural RMTPP
    baseline.  These outputs are therefore intentionally compatible with
    ``Models/EasyTPP/easy_tpp/model/torch_model/torch_rmtpp.py``.
    """

    def __init__(self, dataset_root=None, output_root=None, seed=2024):
        self.dataset_root = Path(dataset_root or infer_dataset_root(__file__))
        self.output_root = Path(output_root or infer_output_root(__file__))
        self.seed = int(seed)

    def retweet(self, output_dir=None):
        return self._standard("retweet", output_dir)

    def taxi(self, output_dir=None):
        return self._standard("taxi", output_dir)

    def stackoverflow(self, output_dir=None):
        return self._standard("stackoverflow", output_dir)

    def taobao(self, output_dir=None):
        return self._standard("taobao", output_dir)

    def amazon(self, output_dir=None):
        return self._standard("amazon", output_dir)

    def covid_policy_tracker(self, output_dir=None, epsilon=1e-4):
        _, splits, metadata = load_covid_policy_tracker_splits(
            self.dataset_root, seed=self.seed, epsilon=epsilon
        )
        target = ensure_directory(output_dir or self.output_root / "covid_policy_tracker")
        write_json(target / "metadata.json", metadata)
        return {
            split: write_json(target / "{}.json".format(split), records)
            for split, records in splits.items()
        }

    def dws(self, output_dir=None, variants=None, epsilon=1e-6):
        base = ensure_directory(output_dir or self.output_root / "dws")
        outputs = {}
        for variant, (_, splits) in load_all_dws(
            self.dataset_root, variants=variants, seed=self.seed, epsilon=epsilon
        ).items():
            target = ensure_directory(base / "dws_{}".format(variant))
            outputs[variant] = {
                split: write_json(target / "{}.json".format(split), records)
                for split, records in splits.items()
            }
        return outputs

    def _standard(self, dataset_name, output_dir):
        _, splits = load_standard_splits(self.dataset_root, dataset_name)
        target = ensure_directory(output_dir or self.output_root / dataset_name)
        return {
            split: write_json(target / "{}.json".format(split), records)
            for split, records in splits.items()
        }
