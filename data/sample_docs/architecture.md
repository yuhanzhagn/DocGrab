# RAG MVP Architecture

This sample project stores document embeddings in Chroma.

The ingestion pipeline loads markdown and text files, splits them into chunks, and
indexes each chunk for retrieval.

Answers should include citations that point back to the source file and chunk.
