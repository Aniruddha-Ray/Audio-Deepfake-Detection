from types import SimpleNamespace

import pytest

from audiodf.config import Settings
from audiodf.streaming.kafka_io import consumer_config, ensure_topics


class _Future:
    def __init__(self, error=None):
        self.error = error

    def result(self):
        if self.error:
            raise self.error


class FakeAdmin:
    def __init__(self, have, race=False):
        self.have, self.created, self.race = set(have), [], race

    def list_topics(self, timeout=None):
        return SimpleNamespace(topics={t: object() for t in self.have})

    def create_topics(self, new):
        self.created += [(t.name, t.partitions) for t in new]
        err = RuntimeError("KafkaError{code=TOPIC_ALREADY_EXISTS}") if self.race else None
        return {t.name: _Future(err) for t in new}


def _new(name, num_partitions, replication_factor):
    return SimpleNamespace(name=name, partitions=num_partitions, replication=replication_factor)


def test_topics_are_created_up_front_with_the_configured_partitions():
    s = Settings()
    s.kafka.partitions = 3
    admin = FakeAdmin(have=["__consumer_offsets", "unrelated"])
    created = ensure_topics(s, admin, _new)
    assert set(created) == {s.kafka.audio_topic, s.kafka.result_topic}
    assert dict(admin.created) == {s.kafka.audio_topic: 3, s.kafka.result_topic: 1}


def test_existing_topics_are_left_alone_and_a_creation_race_is_not_an_error():
    s = Settings()
    assert ensure_topics(s, FakeAdmin(have=[s.kafka.audio_topic, s.kafka.result_topic]), _new) == []
    assert ensure_topics(s, FakeAdmin(have=[s.kafka.result_topic], race=True), _new) == [s.kafka.audio_topic]  # created by someone else first
    with pytest.raises(RuntimeError):  # any other failure is raised
        admin = FakeAdmin(have=[])
        admin.create_topics = lambda new: {t.name: _Future(RuntimeError("boom")) for t in new}
        ensure_topics(s, admin, _new)


def test_a_new_group_reads_from_the_start_so_calls_sent_while_it_was_joining_are_not_lost():
    cfg = consumer_config(Settings())
    assert cfg["auto.offset.reset"] == "earliest" and cfg["enable.auto.commit"] is True and cfg["group.id"] == "deepfake-detector"
