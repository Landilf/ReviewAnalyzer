import csv
import json
import tempfile
import unittest
from pathlib import Path

from absa_training.data import load_examples, split_by_review
from absa_training.metrics import build_absa_metrics, primary_scores
from absa_training.trainer import activate_epoch


class AbsATrainingTests(unittest.TestCase):
    def _write_dataset(self, directory: Path) -> Path:
        path = directory / "annotations.csv"
        fields = [
            "review_id", "text", "aspect", "aspect_normalized", "aspect_start", "aspect_end",
            "aspect_sentiment", "annotation_status",
        ]
        rows = [
            ["1", "Хорошая камера", "камера", "камера", "8", "14", "positive", "accepted"],
            ["1", "Хорошая камера", "Хорошая", "качество", "0", "7", "positive", "accepted"],
            ["2", "Слабая батарея", "батарея", "аккумулятор", "7", "14", "negative", "accepted"],
            ["3", "Дорогая цена", "цена", "цена", "8", "12", "negative", "rejected"],
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def test_loader_checks_spans_filters_rejections_and_marks_aspect(self):
        with tempfile.TemporaryDirectory() as directory_name:
            dataset = self._write_dataset(Path(directory_name))
            examples = load_examples([dataset])

        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0].label, "камера::positive")
        self.assertIn("[ASPECT] камера [/ASPECT]", examples[0].marked_text())

    def test_split_keeps_all_aspects_of_a_review_together(self):
        with tempfile.TemporaryDirectory() as directory_name:
            examples = load_examples([self._write_dataset(Path(directory_name))])
            train, validation = split_by_review(examples, validation_fraction=0.5, random_state=2)

        self.assertTrue({item.group_id for item in train}.isdisjoint({item.group_id for item in validation}))
        self.assertEqual(len(train) + len(validation), len(examples))

    def test_metrics_include_taxonomy_and_taxonomy_conditioned_sentiment(self):
        metrics = build_absa_metrics(
            ["камера::positive", "аккумулятор::negative", "камера::negative"],
            ["камера::positive", "аккумулятор::positive", "экран::negative"],
            ["аккумулятор", "камера", "экран"],
        )

        self.assertIn("камера", metrics["per_taxonomy"])
        self.assertIn("precision", metrics["taxonomy"]["камера"])
        self.assertIn("joint_macro_f1", primary_scores(metrics))

    def test_activate_epoch_updates_manifest_without_deleting_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory_name:
            run_dir = Path(directory_name) / "run"
            checkpoint = run_dir / "epochs" / "epoch-002"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (run_dir / "run_manifest.json").write_text(json.dumps({"active_epoch": 1}), encoding="utf-8")

            result = activate_epoch(run_dir, 2)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            checkpoint_exists = checkpoint.is_dir()

        self.assertEqual(result, checkpoint)
        self.assertEqual(manifest["active_epoch"], 2)
        self.assertTrue(checkpoint_exists)


if __name__ == "__main__":
    unittest.main()
