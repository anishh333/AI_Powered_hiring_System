"""
Observability and logging configuration using Logfire.
"""
import logfire
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Global in-memory span exporter to collect traces locally
memory_exporter = InMemorySpanExporter()


def init_observability():
    """Initializes Logfire observability/logging system and registers a memory exporter."""
    try:
        logfire.configure()
    except Exception:
        # Fallback to local console logging only if Logfire token or auth is missing
        logfire.configure(send_to_logfire=False)
    
    # Register our memory exporter with the active OpenTelemetry tracer provider
    try:
        provider = trace.get_tracer_provider()
        if not hasattr(provider, "add_span_processor"):
            # Set up a real TracerProvider if we are in proxy/uninitialized state
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
        provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    except Exception:
        pass

    # Instrument standard HTTP request client if logfire integrations are installed
    try:
        logfire.instrument_requests()
    except Exception:
        pass


def get_captured_spans():
    """Exposes captured spans for the dashboard."""
    return memory_exporter.get_finished_spans()


def clear_captured_spans():
    """Clears the buffer of captured spans."""
    memory_exporter.clear()
