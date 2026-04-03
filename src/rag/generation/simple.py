from rag.generation.base import AnswerGenerator
from rag.schemas.answer import Citation, FinalAnswer
from rag.schemas.retrieval import RetrievalResult


class SimpleGroundedAnswerGenerator(AnswerGenerator):
    """Minimal grounded generator for the MVP.

    This does not call an LLM yet. It extracts the most relevant chunks and returns
    a concise answer with citations so the whole project is runnable locally.
    """

    def generate(self, query: str, retrieval_results: list[RetrievalResult]) -> FinalAnswer:
        if not retrieval_results:
            return FinalAnswer(
                answer_text="I could not find relevant context in the indexed documents.",
                citations=[],
                retrieved_chunks=[],
            )

        selected = retrieval_results[:3]
        summary_lines = [f"Question: {query}", "", "Grounded context:"]
        citations: list[Citation] = []
        retrieved_chunks: list[dict] = []

        for index, result in enumerate(selected, start=1):
            snippet = result.text.strip().replace("\n", " ")
            snippet = snippet[:280] + ("..." if len(snippet) > 280 else "")
            chunk_index = int(result.metadata.get("chunk_index", index - 1))
            start_char = int(result.metadata.get("start_char", 0))
            end_char = int(result.metadata.get("end_char", start_char + len(result.text)))

            summary_lines.append(
                f"[{index}] {snippet} (source: {result.source_path}, chunk {chunk_index})"
            )
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
                    "source_path": result.source_path,
                    "chunk_index": chunk_index,
                    "text": result.text,
                }
            )

        answer_text = "\n".join(summary_lines)
        return FinalAnswer(
            answer_text=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
        )
