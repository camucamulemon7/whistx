from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

refine_latency_seconds = Histogram(
    "refine_latency_seconds",
    "High-accuracy refinement latency in seconds",
    buckets=(0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0),
)

refine_queue_depth = Gauge(
    "refine_queue_depth",
    "Number of queued refinement jobs waiting to be processed",
)

profile_switch_total = Counter(
    "profile_switch_total",
    "Number of profile selections initiated by clients",
    ("profile",),
)

longform_chunk_latency_seconds = Histogram(
    "longform_chunk_latency_seconds",
    "End-to-end latency from chunk emission to final output",
    buckets=(0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 30.0, 60.0, 120.0),
)

longform_queue_depth = Gauge(
    "longform_queue_depth",
    "Approximate number of longform chunks currently buffered",
)
