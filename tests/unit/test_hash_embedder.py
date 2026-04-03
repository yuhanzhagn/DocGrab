from rag.embeddings.hash_embedder import HashingEmbedder


def test_hash_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder(dimension=32)

    first = embedder.embed_query("Chroma stores vectors")
    second = embedder.embed_query("Chroma stores vectors")

    assert first == second


def test_hash_embedder_returns_expected_shape() -> None:
    embedder = HashingEmbedder(dimension=48)

    vectors = embedder.embed_texts(["alpha beta", "gamma delta"])

    assert len(vectors) == 2
    assert all(len(vector) == 48 for vector in vectors)
    assert any(value != 0.0 for value in vectors[0])
