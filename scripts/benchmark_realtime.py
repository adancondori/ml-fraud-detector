"""Benchmark real-time scoring latency against a live scorer instance.

Sends N sequential POST /api/v1/score requests and reports p50/p95/p99/max latency.

Usage:
    source venv/bin/activate
    python scripts/benchmark_realtime.py                      # default: 50 requests, localhost:8765
    python scripts/benchmark_realtime.py --n 100 --url http://ml-scorer:8000
    python scripts/benchmark_realtime.py --n 200 --concurrency 5

Gate: p95 < 150ms to enable real-time scoring in platform (timeout = 200ms).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URL = "http://localhost:8765"
DEFAULT_N = 50
DEFAULT_CONCURRENCY = 1

# Representativo de un pago real con todos los campos IF-40
SAMPLE_PAYLOAD = {
    "payment_id": 99999,
    "user_id": 1234,
    "facility_id": 56,
    "reservation_paid_out": 120.0,
    "currency": "USD",
    "created_at": "2026-06-17T14:00:00",
    "captured_at": "2026-06-17T14:00:05",
    "gateway": "stripe",
    "payment_source": "pbp_web",
    "discount": 0.0,
    "tip": 0.0,
    "payment_method": "card",
    "category": "reservation",
    "club_credit_flag": False,
    "paid_by_manager": False,
}


def percentile(data: list[float], p: float) -> float:
    idx = max(0, min(len(data) - 1, int(len(data) * p / 100)))
    return sorted(data)[idx]


def run_sequential(url: str, n: int) -> list[float]:
    latencies: list[float] = []
    with httpx.Client(timeout=5.0) as client:
        for i in range(n):
            t0 = time.perf_counter()
            r = client.post(f"{url}/api/v1/score", json=SAMPLE_PAYLOAD)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(elapsed_ms)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n}  last={elapsed_ms:.1f}ms", flush=True)
    return latencies


async def _async_request(client: httpx.AsyncClient, url: str) -> float:
    t0 = time.perf_counter()
    r = await client.post(f"{url}/api/v1/score", json=SAMPLE_PAYLOAD)
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


async def run_concurrent(url: str, n: int, concurrency: int) -> list[float]:
    latencies: list[float] = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded(i: int) -> float:
            async with semaphore:
                ms = await _async_request(client, url)
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{n}  last={ms:.1f}ms", flush=True)
                return ms

        latencies = await asyncio.gather(*[bounded(i) for i in range(n)])
    return list(latencies)


def print_report(latencies: list[float], concurrency: int) -> None:
    p95 = percentile(latencies, 95)
    gate_ok = p95 < 150

    print("\n─── Benchmark results ───────────────────────────────")
    print(f"  Requests  : {len(latencies)}")
    print(f"  Concurrency: {concurrency}")
    print(f"  p50       : {statistics.median(latencies):.1f} ms")
    print(f"  p95       : {p95:.1f} ms  {'✅ GATE OK (<150ms)' if gate_ok else '❌ GATE FAIL (≥150ms)'}")
    print(f"  p99       : {percentile(latencies, 99):.1f} ms")
    print(f"  max       : {max(latencies):.1f} ms")
    print(f"  mean      : {statistics.mean(latencies):.1f} ms")
    print(f"  stdev     : {statistics.stdev(latencies):.1f} ms" if len(latencies) > 1 else "")
    print("─────────────────────────────────────────────────────")

    if not gate_ok:
        print("\n⚠️  p95 ≥ 150ms — do NOT activate real-time scoring yet.")
        print("   Platform timeout is 200ms; latency spikes will cause silent drop of scores.")
        sys.exit(1)
    else:
        print("\n✅  p95 < 150ms — safe to activate real-time scoring in platform.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark scorer real-time latency")
    parser.add_argument("--url", default=DEFAULT_URL, help="Scorer base URL")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of requests")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Concurrent requests (1 = sequential)",
    )
    args = parser.parse_args()

    print(f"Benchmarking {args.url}  n={args.n}  concurrency={args.concurrency}")

    # Warm-up: 3 requests not counted
    print("Warm-up (3 requests)...")
    with httpx.Client(timeout=5.0) as c:
        for _ in range(3):
            try:
                c.post(f"{args.url}/api/v1/score", json=SAMPLE_PAYLOAD)
            except Exception:
                print(f"ERROR: scorer not reachable at {args.url}")
                sys.exit(1)

    print(f"Running {args.n} timed requests...")
    if args.concurrency > 1:
        latencies = asyncio.run(run_concurrent(args.url, args.n, args.concurrency))
    else:
        latencies = run_sequential(args.url, args.n)

    print_report(latencies, args.concurrency)


if __name__ == "__main__":
    main()
