from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.observability.langfuse_adapter import LangfuseAdapter, get_langfuse_client, shutdown_langfuse


class TestLangfuseAdapter:
    """Langfuse 适配器测试。"""

    def setup_method(self):
        """每个测试前重置全局客户端状态。"""
        import app.services.observability.langfuse_adapter as adapter_module

        adapter_module._langfuse_client = None

    def teardown_method(self):
        """每个测试后清理全局客户端状态。"""
        import app.services.observability.langfuse_adapter as adapter_module

        adapter_module._langfuse_client = None

    def test_emit_trace_noop_when_disabled(self):
        """Langfuse 未启用时 emit_trace 应为 no-op。"""
        adapter = LangfuseAdapter()
        with patch("app.services.observability.langfuse_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                enable_langfuse=False,
                langfuse_public_key=None,
                langfuse_secret_key=None,
                langfuse_host=None,
            )
            # 不应抛出异常
            adapter.emit_trace({"trace_type": "chat_qa", "trace_id": "test-123"})

    def test_emit_trace_with_mock_client(self):
        """Langfuse 启用时 emit_trace 应创建 trace 和 generation。"""
        adapter = LangfuseAdapter()
        mock_trace = MagicMock()
        mock_client = MagicMock()
        mock_client.trace.return_value = mock_trace

        with patch("app.services.observability.langfuse_adapter.get_langfuse_client", return_value=mock_client):
            adapter.emit_trace(
                {
                    "trace_type": "chat_qa",
                    "trace_id": "trace-abc",
                    "user_id": "user-123",
                    "session_id": "session-456",
                    "query_text": "测试问题",
                    "model_name": "gpt-4.1-mini",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "latency_ms": 1200,
                    "error_text": None,
                    "trace_metadata": {"confidence": "high", "insufficient_evidence": False},
                }
            )

        mock_client.trace.assert_called_once()
        call_kwargs = mock_client.trace.call_args[1]
        assert call_kwargs["id"] == "trace-abc"
        assert call_kwargs["name"] == "chat_qa"
        assert call_kwargs["user_id"] == "user-123"

        mock_trace.generation.assert_called_once()
        gen_kwargs = mock_trace.generation.call_args[1]
        assert gen_kwargs["model"] == "gpt-4.1-mini"
        assert gen_kwargs["usage"]["input"] == 100
        assert gen_kwargs["usage"]["output"] == 50

        mock_client.flush.assert_called_once()

    def test_create_generation_returns_none_when_disabled(self):
        """Langfuse 未启用时 create_generation 应返回 (None, None)。"""
        with patch("app.services.observability.langfuse_adapter.get_langfuse_client", return_value=None):
            trace, generation = LangfuseAdapter.create_generation(
                name="test_call",
                model="gpt-4.1-mini",
            )
        assert trace is None
        assert generation is None

    def test_create_generation_with_mock_client(self):
        """Langfuse 启用时 create_generation 应创建 trace 和 generation。"""
        mock_generation = MagicMock()
        mock_trace = MagicMock()
        mock_trace.generation.return_value = mock_generation
        mock_client = MagicMock()
        mock_client.trace.return_value = mock_trace

        with patch("app.services.observability.langfuse_adapter.get_langfuse_client", return_value=mock_client):
            trace, generation = LangfuseAdapter.create_generation(
                trace_id="my-trace",
                name="answer_generation",
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "hello"}],
                metadata={"key": "value"},
            )

        assert trace is mock_trace
        assert generation is mock_generation
        mock_client.trace.assert_called_once_with(
            id="my-trace",
            name="answer_generation",
            metadata={"key": "value"},
        )

    def test_end_generation_calls_end(self):
        """end_generation 应正确调用 generation.end()。"""
        mock_generation = MagicMock()
        LangfuseAdapter.end_generation(
            mock_generation,
            output_content="回答内容",
            latency_ms=800,
            prompt_tokens=60,
            completion_tokens=30,
            model="gpt-4.1-mini",
        )
        mock_generation.end.assert_called_once()
        end_kwargs = mock_generation.end.call_args[1]
        assert end_kwargs["output"] == "回答内容"
        assert end_kwargs["usage"]["input"] == 60
        assert end_kwargs["usage"]["output"] == 30
        assert end_kwargs["model"] == "gpt-4.1-mini"

    def test_end_generation_with_error(self):
        """end_generation 在有错误时应设置 level=ERROR。"""
        mock_generation = MagicMock()
        LangfuseAdapter.end_generation(
            mock_generation,
            error_text="connection timeout",
        )
        mock_generation.end.assert_called_once()
        end_kwargs = mock_generation.end.call_args[1]
        assert end_kwargs["level"] == "ERROR"
        assert end_kwargs["status_message"] == "connection timeout"

    def test_end_generation_noop_when_none(self):
        """end_generation 传入 None 时应为 no-op。"""
        LangfuseAdapter.end_generation(None, output_content="test")

    def test_get_langfuse_client_returns_none_when_disabled(self):
        """enable_langfuse=False 时应返回 None。"""
        with patch("app.services.observability.langfuse_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                enable_langfuse=False,
                langfuse_public_key="pk-test",
                langfuse_secret_key="sk-test",
                langfuse_host=None,
            )
            client = get_langfuse_client()
        assert client is None

    def test_get_langfuse_client_returns_none_when_missing_keys(self):
        """enable_langfuse=True 但缺少密钥时应返回 None。"""
        with patch("app.services.observability.langfuse_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                enable_langfuse=True,
                langfuse_public_key=None,
                langfuse_secret_key=None,
                langfuse_host=None,
            )
            client = get_langfuse_client()
        assert client is None

    def test_shutdown_langfuse_flushes(self):
        """shutdown_langfuse 应调用 flush 并清理客户端。"""
        mock_client = MagicMock()
        import app.services.observability.langfuse_adapter as adapter_module

        adapter_module._langfuse_client = mock_client

        shutdown_langfuse()
        mock_client.flush.assert_called_once()
        assert adapter_module._langfuse_client is None

    def test_shutdown_langfuse_noop_when_not_initialized(self):
        """客户端未初始化时 shutdown_langfuse 应为 no-op。"""
        shutdown_langfuse()  # 不应抛出异常
