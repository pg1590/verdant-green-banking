"""Measure load time and request latency of the real runtime (API service + P4 provider + memory).

    python scripts/benchmark.py                                   # generated bundle, else the preview
    python scripts/benchmark.py --bundle data/preview/team_bundle.json

Feedback memory goes to a temporary database, so the local one is never touched.
Explanations use the offline path (cache or validated template); live LLM latency is
measured separately by `python -m src.llm.eval.run_eval`.
"""
import argparse
import asyncio
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('LLM_PROVIDER', 'offline')
os.environ['GREEN_MEMORY_PATH'] = os.path.join(tempfile.mkdtemp(), 'benchmark.db')

import httpx

from src.common import config
from src.recommend.runtime import load_runtime_service

ENDPOINTS = ['/customers/{cid}/insight', '/customers/{cid}/recommendations?include_suppressed=true',
             '/portfolio/summary', '/portfolio/opportunities', '/segments']


def median_ms(samples: list[float]) -> str:
    return f'{statistics.median(samples) * 1000:,.0f} ms'


async def api_latency(service, cid: str, repeats: int) -> None:
    import api.main as api
    api.get_service = lambda: service
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url='http://benchmark') as client:
        for template in ENDPOINTS:
            path, samples = template.format(cid=cid), []
            for _ in range(repeats):
                start = time.perf_counter()
                response = await client.get(path)
                samples.append(time.perf_counter() - start)
                response.raise_for_status()
            print(f'GET {path}: median {median_ms(samples)}')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    generated = config.PROCESSED_DIR / 'team_bundle.json'
    bundle = args.bundle or (generated if generated.exists() else config.DATA_DIR / 'preview' / 'team_bundle.json')
    print(f'Bundle: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)')
    start = time.perf_counter()
    service = load_runtime_service(bundle)
    print(f'Load and validate {len(service.customers):,} customers: {time.perf_counter() - start:.2f} s')
    ids = sorted(service.customers)
    sample = ids[::max(1, len(ids) // 40)][:40]
    per_customer = []
    for cid in sample:
        start = time.perf_counter()
        service.insight(cid)
        per_customer.append(time.perf_counter() - start)
    print(f'insight() for {len(sample)} customers: median {median_ms(per_customer)}, max {max(per_customer) * 1000:,.0f} ms')
    asyncio.run(api_latency(service, sample[0], args.repeats))


if __name__ == '__main__':
    main()
