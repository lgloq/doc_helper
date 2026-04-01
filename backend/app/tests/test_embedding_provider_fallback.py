from app.services.ingestion.embeddings import DeterministicEmbeddingProvider, OpenAIEmbeddingProvider


class _FailingEmbeddingsClient:
    def create(self, **kwargs):
        raise RuntimeError("boom")


class _FailingClient:
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1"
        self.embeddings = _FailingEmbeddingsClient()


def test_openai_embedding_provider_falls_back_to_deterministic():
    provider = OpenAIEmbeddingProvider(api_key="test-key", model="test-model", dimensions=8, base_url="https://openrouter.ai/api/v1")
    provider.client = _FailingClient()

    vectors = provider.embed_texts(["hello world"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 8
    assert vectors == DeterministicEmbeddingProvider(8).embed_texts(["hello world"])
