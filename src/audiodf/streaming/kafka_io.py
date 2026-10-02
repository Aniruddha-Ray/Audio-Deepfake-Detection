"""Kafka producer (simulated call source) and consumer (detector worker).

Messages on the audio topic: key = call_id (keeps a call's chunks ordered on one partition),
value = 16 kHz mono PCM16 bytes (0.5 s by default), headers: seq, eof ("1" on the last chunk).
Verdicts are published as JSON on the result topic, keyed by call_id.
"""

from __future__ import annotations

import json
import time

from audiodf.config import Settings
from audiodf.data.audio import float_to_pcm16, load_audio
from audiodf.inference.engine import DetectionEngine
from audiodf.monitoring import metrics
from audiodf.streaming.processor import ChunkProcessor


def _kafka():
    try:
        import confluent_kafka
    except ImportError as exc:
        raise RuntimeError("confluent-kafka is not installed: pip install confluent-kafka") from exc
    return confluent_kafka


def simulate_call(settings: Settings, wav_path: str, call_id: str, realtime: bool = False) -> int:
    """Publish a recording as a stream of chunks. Returns the number of chunks sent."""
    kafka = _kafka()
    producer = kafka.Producer({"bootstrap.servers": settings.kafka.bootstrap_servers})
    wave = load_audio(wav_path, settings.audio.sample_rate)
    step = int(settings.stream.chunk_seconds * settings.audio.sample_rate)
    chunks = [wave[i:i + step] for i in range(0, len(wave), step)]
    for seq, chunk in enumerate(chunks):
        producer.produce(settings.kafka.audio_topic, key=call_id, value=float_to_pcm16(chunk),
                         headers=[("seq", str(seq).encode()), ("eof", b"1" if seq == len(chunks) - 1 else b"0")])
        producer.poll(0)
        if realtime:
            time.sleep(settings.stream.chunk_seconds)
    producer.flush()
    return len(chunks)


def run_consumer(settings: Settings, engine: DetectionEngine | None = None, metrics_port: int | None = 9100,
                 log=print) -> None:
    kafka = _kafka()
    engine = engine or DetectionEngine.from_artifacts(settings)
    processor = ChunkProcessor(engine)
    consumer = kafka.Consumer({"bootstrap.servers": settings.kafka.bootstrap_servers,
                               "group.id": settings.kafka.group_id, "auto.offset.reset": "latest",
                               "enable.auto.commit": True})
    producer = kafka.Producer({"bootstrap.servers": settings.kafka.bootstrap_servers})
    consumer.subscribe([settings.kafka.audio_topic])
    if metrics_port:
        from prometheus_client import start_http_server

        start_http_server(metrics_port, registry=metrics.REGISTRY)
    log(f"consuming {settings.kafka.audio_topic} -> {settings.kafka.result_topic}")

    last_sweep = time.monotonic()
    try:
        while True:
            msg = consumer.poll(1.0)
            if time.monotonic() - last_sweep > 5:
                processor.evict_idle()
                last_sweep = time.monotonic()
            if msg is None:
                continue
            if msg.error():
                metrics.ERRORS.labels("kafka").inc()
                log(f"kafka error: {msg.error()}")
                continue
            headers = {k: v for k, v in (msg.headers() or [])}
            call_id = (msg.key() or b"unknown").decode()
            try:
                verdicts = processor.handle(call_id, msg.value() or b"", eof=headers.get("eof") == b"1")
            except Exception as exc:  # a bad chunk must not kill the worker
                metrics.ERRORS.labels("kafka").inc()
                log(f"failed on call {call_id}: {exc}")
                continue
            for v in verdicts:
                metrics.record_verdict(v, "kafka")
                producer.produce(settings.kafka.result_topic, key=call_id, value=json.dumps(v.to_dict()).encode())
            producer.poll(0)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        producer.flush()
