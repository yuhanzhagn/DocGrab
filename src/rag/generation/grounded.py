from rag.schemas.answer import Citation, FinalAnswer
from rag.schemas.retrieval import RetrievalResult


class GroundedAnswerBuilder:
    _WEAK_MATCH_THRESHOLD = 0.15

    def should_fallback(self, retrieval_results: list[RetrievalResult]) -> bool:
        if not retrieval_results:
            return True
        strongest_match = max(result.score for result in retrieval_results)
        return strongest_match < self._WEAK_MATCH_THRESHOLD

    def build_final_answer(
        self,
        answer_text: str,
        retrieval_results: list[RetrievalResult],
    ) -> FinalAnswer:
        selected = retrieval_results[:3]
        citations: list[Citation] = []
        retrieved_chunks: list[dict] = []

        for index, result in enumerate(selected, start=1):
            snippet = self._build_snippet(result.text)
            chunk_index = int(result.metadata.get("chunk_index", index - 1))
            start_char = int(result.metadata.get("start_char", 0))
            end_char = int(result.metadata.get("end_char", start_char + len(result.text)))

            citations.append(
                Citation(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    source_path=result.source_path,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    snippet=snippet,
                )
            )
            retrieved_chunks.append(
                {
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "distance": result.distance,
                    "relevance": result.relevance,
                    "source_path": result.source_path,
                    "chunk_index": chunk_index,
                    "text": result.text,
                }
            )

        return FinalAnswer(
            answer_text=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
        )

    def build_fallback_answer(
        self,
        retrieval_results: list[RetrievalResult] | None = None,
    ) -> FinalAnswer:
        retrieved_chunks: list[dict] = []
        for result in (retrieval_results or [])[:3]:
            retrieved_chunks.append(
                {
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "distance": result.distance,
                    "relevance": result.relevance,
                    "source_path": result.source_path,
                    "chunk_index": int(result.metadata.get("chunk_index", 0)),
                    "text": result.text,
                }
            )

        return FinalAnswer(
            answer_text=(
                "I could not find enough grounded support in the indexed documents "
                "to answer confidently."
            ),
            citations=[],
            retrieved_chunks=retrieved_chunks,
        )

    def build_prompt(self, query: str, retrieval_results: list[RetrievalResult]) -> str:
        selected = retrieval_results[:3]
        context_blocks: list[str] = []
        for index, result in enumerate(selected, start=1):
            chunk_index = int(result.metadata.get("chunk_index", index - 1))
            context_blocks.append(
                "\n".join(
                    [
                        f"[Chunk {index}]",
                        f"source_path: {result.source_path}",
                        f"chunk_index: {chunk_index}",
                        f"score: {result.score:.4f}",
                        "text:",
                        result.text.strip(),
                    ]
                )
            )

        return "\n\n".join(
            [
                "You are a grounded question answering assistant.",
                "Answer only from the retrieved context.",
                "If the context is insufficient, say that the context is insufficient.",
                "Do not invent facts, sources, or citations.",
                "Keep the answer concise and grounded.",
                f"Question: {query}",
                "Retrieved context:",
                "\n\n".join(context_blocks),
                "Answer:",
            ]
        )

    @staticmethod
    def _build_snippet(text: str) -> str:
        snippet = text.strip().replace("\n", " ")
        return snippet[:280] + ("..." if len(snippet) > 280 else "")
