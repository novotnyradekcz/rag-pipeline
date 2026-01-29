# Logs Directory

This directory stores query and session logs from the RAG pipeline.

## Log Files

- `query_YYYY-MM-DD_HH-MM-SS.txt` - Individual query logs
- `session_YYYY-MM-DD_HH-MM-SS.txt` - Interactive session transcripts

## Format

Each log file contains:
- Timestamp
- Query/Question
- Answer
- Source references
- Metadata (model used, number of chunks retrieved, etc.)

Logs are automatically created when using:
- `python main.py --query "your question"`
- `python main.py --interactive` (saved on exit)
