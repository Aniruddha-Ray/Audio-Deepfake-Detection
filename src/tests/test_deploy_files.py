"""The deployment files must stay consistent with each other and with the code (found while running the stack for real: audit.md phase 33)."""
import re
from pathlib import Path

import yaml

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"


def _compose(name):
    return yaml.safe_load((DEPLOY / name).read_text())


def test_both_images_install_the_same_requirements_and_differ_only_in_the_pytorch_build():
    cpu, gpu = [(DEPLOY / n).read_text() for n in ("Dockerfile", "Dockerfile.gpu")]
    assert "whl/cpu" in cpu and re.search(r"whl/cu\d+", gpu) and "whl/cpu" not in gpu
    for text in (cpu, gpu):
        assert "pip install --no-cache-dir -r requirements.txt" in text and "COPY audiodf" in text and "HF_HOME=/cache/hf" in text
    def strip(t):
        t = re.sub(r"whl/(cpu|cu\d+)", "whl/X", re.sub(r"#.*", "", t)).replace(" NVIDIA_VISIBLE_DEVICES=all", "")
        return [ln.strip() for ln in t.splitlines() if ln.strip()]

    assert strip(cpu) == strip(gpu)


def test_the_runtime_requirements_cover_what_the_container_imports():
    req = (DEPLOY.parent / "requirements.txt").read_text().lower()
    for pkg in ("confluent-kafka", "shap", "matplotlib", "requests", "transformers", "prometheus_client", "fastapi", "uvicorn", "torch"):
        assert pkg in req, f"{pkg} is imported by the service but missing from requirements.txt"


def test_the_gpu_override_gives_the_api_and_the_worker_a_gpu_and_the_gpu_image():
    base, gpu = _compose("docker-compose.yml"), _compose("docker-compose.gpu.yml")
    assert set(gpu["services"]) == {"api", "worker"} <= set(base["services"])
    for name, svc in gpu["services"].items():
        assert svc["build"]["dockerfile"] == "deploy/Dockerfile.gpu" and svc["image"] == "audiodf-gpu", name
        dev = svc["deploy"]["resources"]["reservations"]["devices"][0]
        assert dev["driver"] == "nvidia" and "gpu" in dev["capabilities"], name


def test_the_stack_mounts_the_model_and_the_weight_caches_and_prometheus_scrapes_what_exists():
    base = _compose("docker-compose.yml")
    for name in ("api", "worker"):
        vols = " ".join(base["services"][name]["volumes"])
        assert "../../artifacts:/models:ro" in vols and "torch-cache:/cache/torch" in vols and "hf-cache:/cache/hf" in vols, name
    assert {"torch-cache", "hf-cache"} <= set(base["volumes"])
    assert "4.1" in base["services"]["kafka"]["image"]
    jobs = {s["targets"][0] for j in yaml.safe_load((DEPLOY / "prometheus.yml").read_text())["scrape_configs"] for s in j["static_configs"]}
    assert jobs == {"api:8000", "worker:9100"}


def test_the_grafana_dashboard_only_uses_metrics_the_service_exports():
    import json

    from audiodf.monitoring import metrics

    exported = set(re.findall(r"^# TYPE (\w+) ", metrics.render().decode(), flags=re.M))
    exported |= {f"{n}_{s}" for n in exported for s in ("bucket", "sum", "count")}
    text = (DEPLOY / "grafana-dashboard.json").read_text()
    used = set(re.findall(r"audiodf_\w+", text))
    assert used and used <= exported | {"audiodf_verdicts_total", "audiodf_errors_total"}, used - exported
    assert json.loads(text)["uid"] == "audiodf"
