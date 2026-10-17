"""Smoke test of the running compose stack (docker compose -f deploy/docker-compose.yml up -d).

  python deploy/smoke_test.py path/to/call1.flac [path/to/call2.flac ...]

Checks, in order: the API is healthy; /predict and /explain answer for every recording; both Prometheus targets are up and counting verdicts;
Grafana has its datasource and dashboard; and (with --kafka) the Kafka path: each recording is sent as chunks from a one-off container on the
compose network and a final verdict comes back on the result topic. Prints one line per check and exits 1 if any fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

API, PROM, GRAFANA = "http://localhost:8000", "http://localhost:9090", "http://localhost:3000"
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def get(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post(url: str, body: bytes, timeout: float = 300) -> dict:
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"), timeout=timeout).read())


def kafka_check(files: list[Path]) -> None:
    ids = {}
    for f in files:
        cid = f"smoke-{uuid.uuid4().hex[:6]}"
        ids[cid] = f.name
        out = subprocess.run(["docker", "run", "--rm", "--network", "deploy_default", "-e", "KAFKA_BOOTSTRAP_SERVERS=kafka:9092", "-v",
                              f"{f.parent.resolve()}:/data:ro", "deploy-api:latest", "python", "-m", "audiodf", "produce", f"/data/{f.name}",
                              "--call-id", cid], capture_output=True, text=True)
        check(f"kafka: sent {f.name}", out.returncode == 0, out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-120:])
    final = {}
    deadline = time.time() + 120
    while time.time() < deadline and len(final) < len(ids):
        time.sleep(5)
        read = subprocess.run(["docker", "exec", "deploy-kafka-1", "/opt/kafka/bin/kafka-console-consumer.sh", "--bootstrap-server", "kafka:9092",
                               "--topic", "deepfake-detection-verdicts", "--from-beginning", "--timeout-ms", "8000", "--property", "print.key=true"],
                              capture_output=True, text=True).stdout
        for line in read.splitlines():
            key, _, val = line.partition("\t")
            try:
                v = json.loads(val)
            except ValueError:
                continue
            if key in ids and v.get("final"):
                final[key] = v
    for cid, name in ids.items():
        v = final.get(cid)
        check(f"kafka: final verdict for {name}", v is not None, f"{v['action']} (score {v['fake_probability']:.4f})" if v else "none within 120 s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--kafka", action="store_true", help="also test the Kafka path (needs the docker CLI; recordings must share a folder)")
    args = ap.parse_args()
    try:
        h = get(f"{API}/health")
        check("api health", h.get("status") == "ok" and h.get("models_loaded"), str(h))
    except Exception as exc:
        check("api health", False, str(exc))
        return 1
    for f in args.recordings:
        v = post(f"{API}/predict", f.read_bytes())
        check(f"/predict {f.name}", v["action"] in ("allow", "verify", "escalate"), f"{v['action']}, score {v['fake_probability']:.4f}, {v['latency_ms']:.0f} ms")
        e = post(f"{API}/explain?nsamples=60", f.read_bytes())
        check(f"/explain {f.name}", bool(e["explanation"]["summary"]) and "facts" in e, f"check faithful: {e['facts']['attribution_check'].get('faithful')}")
    if args.kafka:
        kafka_check(args.recordings)
    time.sleep(10)
    targets = get(f"{PROM}/api/v1/targets")["data"]["activeTargets"]
    check("prometheus targets up", len(targets) >= 2 and all(t["health"] == "up" for t in targets), ", ".join(f"{t['labels']['job']}={t['health']}" for t in targets))
    q = get(f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": "sum(audiodf_verdicts_total)"}))["data"]["result"]
    check("prometheus counts verdicts", bool(q) and float(q[0]["value"][1]) > 0, q[0]["value"][1] if q else "no series")
    ds, dash = get(f"{GRAFANA}/api/datasources"), get(f"{GRAFANA}/api/search?query=Deepfake")
    check("grafana datasource and dashboard", any(d["type"] == "prometheus" for d in ds) and bool(dash), f"{[d['title'] for d in dash]}")
    print("\nALL CHECKS PASSED" if not FAILED else f"\nFAILED: {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
