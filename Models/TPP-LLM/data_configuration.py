"""Dataset adapters for TPP-LLM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _data_configuration_common import (  # noqa: E402
    ensure_directory,
    event_texts,
    infer_dataset_root,
    infer_output_root,
    load_all_dws,
    load_covid_policy_tracker_splits,
    load_standard_splits,
    write_json,
)


class DataConfiguration:
    """Create TPP-LLM JSON, including one text string per event."""

    def __init__(self, dataset_root=None, output_root=None, seed=2024):
        self.dataset_root = Path(dataset_root or infer_dataset_root(__file__))
        self.output_root = Path(output_root or infer_output_root(__file__))
        self.seed = int(seed)

    def retweet(self, output_dir=None, type_labels=None, anonymous_labels=False):
        return self._standard("retweet", output_dir, type_labels, anonymous_labels)

    def taxi(self, output_dir=None, type_labels=None, anonymous_labels=False):
        return self._standard("taxi", output_dir, type_labels, anonymous_labels)

    def stackoverflow(self, output_dir=None, type_labels=None, anonymous_labels=False):
        return self._standard("stackoverflow", output_dir, type_labels, anonymous_labels)

    def taobao(self, output_dir=None, type_labels=None, anonymous_labels=False):
        return self._standard("taobao", output_dir, type_labels, anonymous_labels)

    def amazon(self, output_dir=None, type_labels=None, anonymous_labels=False):
        return self._standard("amazon", output_dir, type_labels, anonymous_labels)

    def covid_policy_tracker(
            self, output_dir=None, type_labels=None, epsilon=1e-4,
            anonymous_labels=False):
        _, splits, metadata = load_covid_policy_tracker_splits(
            self.dataset_root, seed=self.seed, epsilon=epsilon
        )
        target = ensure_directory(output_dir or self.output_root / "covid_policy_tracker")
        write_json(target / "metadata.json", metadata)
        if type_labels is None:
            type_labels = metadata["event_type_mapping"]
        return self._write_splits(
            "covid_policy_tracker", splits, target, type_labels,
            anonymous_labels=anonymous_labels,
        )

    def dws(
            self, output_dir=None, variants=None, type_labels=None,
            epsilon=1e-6, anonymous_labels=False):
        base = ensure_directory(output_dir or self.output_root / "dws")
        outputs = {}
        for variant, (_, splits) in load_all_dws(
            self.dataset_root, variants=variants, seed=self.seed, epsilon=epsilon
        ).items():
            target = ensure_directory(base / "dws_{}".format(variant))
            outputs[variant] = self._write_splits(
                "dws", splits, target, type_labels,
                anonymous_labels=anonymous_labels,
            )
        return outputs

    def _standard(self, dataset_name, output_dir, type_labels, anonymous_labels):
        _, splits = load_standard_splits(self.dataset_root, dataset_name)
        target = ensure_directory(output_dir or self.output_root / dataset_name)
        return self._write_splits(
            dataset_name, splits, target, type_labels,
            anonymous_labels=anonymous_labels,
        )

    def _write_splits(
            self, dataset_name, splits, target, type_labels,
            anonymous_labels=False):
        outputs = {}
        for split, records in splits.items():
            payload = []
            for record in records:
                converted = {
                    "time_since_start": record["time_since_start"],
                    "time_since_last_event": record["time_since_last_event"],
                    "type_event": record["type_event"],
                    "type_text": (
                        [f"event_{value}" for value in record["type_event"]]
                        if anonymous_labels
                        else event_texts(
                            dataset_name, record["type_event"], type_labels
                        )
                    ),
                }
                if "source_index" in record:
                    converted["source_index"] = int(record["source_index"])
                payload.append(converted)
            outputs[split] = write_json(target / "{}.json".format(split), payload)
        return outputs
