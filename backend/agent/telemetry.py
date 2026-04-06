# backend/agent/telemetry.py
from langchain_core.callbacks.base import BaseCallbackHandler
from google.cloud import bigquery
import time
import datetime
import threading

class HandsOnObservability(BaseCallbackHandler):
    """Custom interceptor that asynchronously streams metrics to BigQuery."""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.run_start_time = None
        self.tool_start_time = None
        # Initialize BQ Client (It automatically picks up your Google Cloud credentials)
        self.bq_client = bigquery.Client()
        self.table_id = f"{self.bq_client.project}.llmops_telemetry.agent_metrics"

    def _insert_to_bq(self, row: dict):
        """Background task to insert data without blocking the LLM response."""
        def insert():
            try:
                errors = self.bq_client.insert_rows_json(self.table_id, [row])
                if errors:
                    print(f"[TELEMETRY ERROR] BQ Insert failed: {errors}")
            except Exception as e:
                print(f"[TELEMETRY ERROR] {e}")
                
        thread = threading.Thread(target=insert)
        thread.start()

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.run_start_time = time.time()
        
    def on_llm_end(self, response, **kwargs):
        elapsed_ms = (time.time() - self.run_start_time) * 1000
        
        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        
        row = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "event_type": "llm_call",
            "latency_ms": elapsed_ms,
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "target_name": "llama-3.3-70b-versatile",
            "status": "success"
        }
        self._insert_to_bq(row)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_start_time = time.time()
        
    def on_tool_end(self, output, **kwargs):
        elapsed_ms = (time.time() - self.tool_start_time) * 1000
        
        row = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "event_type": "tool_call",
            "latency_ms": elapsed_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "target_name": kwargs.get("name", "unknown_tool"),
            "status": "success"
        }
        self._insert_to_bq(row)