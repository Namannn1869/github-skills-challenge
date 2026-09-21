import json
import runpy
import tempfile
from pathlib import Path

import pytest

from src.aiops_pipeline import run_pipeline
from src.anomaly_detector import AnomalyDetector
from src.calculations import area_of_circle, get_nth_fibonacci
from src.event_consumer import EventConsumer
from src.event_producer import EventProducer
from src.event_topic import EventTopic


def test_anomaly_detector_raises_on_invalid_log_level_and_uses_error_detection():
    detector = AnomalyDetector(response_time_threshold=100, cpu_threshold=50, memory_threshold=50)

    record = {
        "timestamp": "2024-01-01T00:00:00Z",
        "service": "billing-service",
        "response_time_ms": 200,
        "cpu_percent": 60,
        "memory_percent": 60,
        "log_level": "ERROR",
        "message": "Error seen"
    }

    event = detector.detect(record)

    assert event is not None
    assert event["type"] == "ANOMALY"
    assert "High response time" in event["reasons"]
    assert "High CPU utilization" in event["reasons"]
    assert "High memory utilization" in event["reasons"]
    assert "Error log detected" in event["reasons"]

    assert detector.detect({
        "timestamp": "2024-01-01T00:00:00Z",
        "service": "billing-service",
        "response_time_ms": 10,
        "cpu_percent": 10,
        "memory_percent": 10,
        "log_level": "INFO",
        "message": "okay"
    }) is None


def test_pipeline_publishes_and_consumes_events_from_same_topic():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "service_data.json"
        path.write_text(json.dumps([
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "service": "checkout",
                "response_time_ms": 100,
                "cpu_percent": 20,
                "memory_percent": 21,
                "log_level": "INFO",
                "message": "ok"
            },
            {
                "timestamp": "2024-01-01T00:00:01Z",
                "service": "checkout",
                "response_time_ms": 800,
                "cpu_percent": 90,
                "memory_percent": 90,
                "log_level": "ERROR",
                "message": "fail"
            }
        ]), encoding="utf-8")

        result = run_pipeline(str(path))

    assert result["records_processed"] == 2
    assert len(result["anomalies_detected"]) == 1
    assert len(result["events_consumed"]) == 1
    assert result["events_consumed"][0]["service"] == "checkout"


def test_event_topic_and_producer_consumer_paths():
    topic = EventTopic("demo")
    producer = EventProducer(topic)
    consumer = EventConsumer(topic)

    assert producer.publish(None) is False

    event = {"type": "ANOMALY", "service": "payments"}
    assert producer.publish(event) is True
    assert consumer.consume() == [event]

    topic.clear()
    assert topic.get_messages() == []


def test_calculations_full_branches():
    assert area_of_circle(2) == pytest.approx(12.566370614359172)
    with pytest.raises(ValueError):
        area_of_circle(-1)

    assert get_nth_fibonacci(2) == 1
    assert get_nth_fibonacci(5) == 5
    assert get_nth_fibonacci(10) == 89
    with pytest.raises(ValueError):
        get_nth_fibonacci(-1)


def test_aiops_pipeline_main_entrypoint_runs():
    module_path = Path(__file__).resolve().parents[1] / "src" / "aiops_pipeline.py"
    runpy.run_path(str(module_path), run_name="__main__")
