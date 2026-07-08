# SQL-aware tokenizer assets

This directory contains the released SQL-aware vocabulary additions used by `tokenizer.py`.

- `sql_tokens_139.txt`: one SQL-specific token per line, after case-insensitive de-duplication.
- `sql_tokens_category_summary.json`: category counts and provenance metadata.

The runnable tokenizer implementation remains in the repository root as `tokenizer.py` because the released training scripts import it directly.
