"""Command-line entry point for local taxonomy-aware ABSA fine-tuning."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from absa_training.data import load_examples
from absa_training.trainer import TrainingConfig, activate_epoch, fine_tune, resume_fine_tune


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="дообучить модель и сохранить каждую эпоху")
    train.add_argument("datasets", nargs="+", type=Path, help="CSV-файлы аспектной разметки")
    train.add_argument("--run-dir", type=Path, default=None, help="папка запуска в artifacts/absa")
    train.add_argument("--model", default="cointegrated/rubert-tiny2")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--statuses", nargs="*", default=None, help="включить только указанные статусы разметки")
    train.add_argument("--initial-checkpoint", type=Path, default=None, help="папка сохранённой эпохи для нового дообучения")
    activate = subparsers.add_parser("activate", help="переключить активную модель на сохранённую эпоху")
    activate.add_argument("run_dir", type=Path)
    activate.add_argument("epoch", type=int)
    resume = subparsers.add_parser("resume", help="продолжить прерванный запуск с последней завершённой эпохи")
    resume.add_argument("run_dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "activate":
        checkpoint = activate_epoch(args.run_dir, args.epoch)
        print(f"Активна эпоха {args.epoch}: {checkpoint}")
        return
    if args.command == "resume":
        result = resume_fine_tune(args.run_dir)
        print(f"Продолжение завершено: {result['run_dir']}; лучшая эпоха: {result['best_epoch']}")
        return
    statuses = {item.lower() for item in args.statuses} if args.statuses is not None else None
    examples = load_examples(args.datasets, allowed_statuses=statuses)
    run_dir = args.run_dir or Path("artifacts/absa") / datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    result = fine_tune(
        examples,
        run_dir,
        TrainingConfig(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            random_state=args.seed,
        ),
        [str(path) for path in args.datasets],
        initial_checkpoint=args.initial_checkpoint,
    )
    print(f"Готово: {result['run_dir']}; лучшая эпоха: {result['best_epoch']}")


if __name__ == "__main__":
    main()
