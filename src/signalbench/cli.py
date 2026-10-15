"""Thin command dispatcher; no notebook-only research logic."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from signalbench.core import config, setup
from signalbench.db import connect


def main() -> None:
    """Run a configured pipeline stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "stage",
        choices=[
            "data",
            "universe",
            "bench",
            "finetune",
            "edgar",
            "prices",
            "signal",
            "backtest",
            "report",
            "publish",
        ],
    )
    parser.add_argument("--smoke", action="store_true", help="Isolated two-update training check")
    args = parser.parse_args()
    if args.smoke and args.stage != "finetune":
        parser.error("--smoke is only valid with finetune")
    load_dotenv()
    cfg = config(args.config)
    os.environ.setdefault("HF_HOME", str(Path(cfg["paths"]["cache"]) / "huggingface"))
    if args.smoke:
        from signalbench.models.training import smoke_config

        cfg = smoke_config(cfg)
    setup(cfg["seed"])
    if args.stage == "universe":
        from signalbench.data.universe import run

        run(cfg)
        return
    if args.stage == "report":
        from signalbench.report.render import run

        run(cfg)
        return
    if args.stage == "publish":
        from signalbench.report.hub import publish

        publish(cfg)
        return
    with connect(cfg["paths"]["db"]) as con:
        if args.stage == "data":
            from signalbench.data.datasets import download

            download(cfg, con)
        elif args.stage == "bench":
            from signalbench.eval.benchmark import run

            run(cfg, con)
        elif args.stage == "finetune":
            from signalbench.models.qlora import train
            from signalbench.models.registry import revision

            frame = con.execute("SELECT * FROM labeled_text WHERE source='sentences_allagree'").df()
            train(cfg, frame, revision(cfg, cfg["training"]["base"]))
        elif args.stage == "edgar":
            from signalbench.data.edgar import run

            run(cfg, con)
        elif args.stage == "prices":
            from signalbench.data.prices import run

            run(cfg, con)
        elif args.stage == "signal":
            from signalbench.signal.scoring import run

            run(cfg, con)
        elif args.stage == "backtest":
            from signalbench.backtest.runner import run

            run(cfg, con)


if __name__ == "__main__":
    main()
