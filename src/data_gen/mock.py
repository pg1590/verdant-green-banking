"""Compatibility shim. The uncalibrated mock was replaced by the v1 generator
(src/data_gen/generate.py); `make sample` and older imports still work.

    python -m src.data_gen.mock --n 200 --out data/sample
"""
from __future__ import annotations

from src.common import config
from src.data_gen.generate import MERCHANTS, BASE_RATES, generate as generate_mock, write_tables  # noqa: F401
from src.data_gen.generate import main as _main


def main(argv: list[str] | None = None) -> None:
    _main(argv, default_n=config.settings()["data"]["n_sample_customers"], default_out=config.SAMPLE_DIR)


if __name__ == "__main__":
    main()
