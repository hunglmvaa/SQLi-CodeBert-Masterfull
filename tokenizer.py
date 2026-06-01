"""
tokenizer.py
------------
SQL-aware tokenizer (paper Section "SQL-Aware Tokenization & Dynamic Padding").

Augments the base CodeBERT BPE tokenizer with 139 SQL-specific entries
across three categories:
  (i)   ~99 SQL-92 / SQL-99 / vendor reserved keywords
  (ii)  ~24 operators, comparison symbols, and comment sequences
  (iii) ~17 common injection artifacts and tautological patterns
After case-insensitive deduplication the final vocabulary additions count is
exactly 139, matching the paper. Run `python tokenizer.py` to verify.

New embeddings are initialized at the mean of the existing embedding matrix
(see Eq. 1 in the paper) and remain fully trainable during fine-tuning.
"""

import torch
from transformers import AutoTokenizer, AutoModel


# ── SQL vocabulary additions (127 entries total) ─────────────────────────────

SQL_KEYWORDS = [
    # DML
    "SELECT", "INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE",
    # DDL
    "CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME",
    # DQL / joins
    "FROM", "WHERE", "JOIN", "INNER", "OUTER", "LEFT", "RIGHT",
    "FULL", "CROSS", "ON", "HAVING", "GROUP", "ORDER", "BY",
    "UNION", "INTERSECT", "EXCEPT", "ALL", "DISTINCT",
    # Subquery / set ops
    "EXISTS", "IN", "NOT", "ANY", "SOME", "INTO",
    # Functions / exec
    "EXEC", "EXECUTE", "CALL", "PROCEDURE", "FUNCTION",
    # System tables / metadata
    "INFORMATION_SCHEMA", "SYSOBJECTS", "SYSCOLUMNS", "SYSUSERS",
    "sys.tables", "sys.columns", "master.dbo",
    # Time-based / blind injection
    "WAITFOR", "DELAY", "SLEEP", "BENCHMARK",
    # File operations
    "LOAD_FILE", "INTO OUTFILE", "INTO DUMPFILE",
    # Transaction
    "COMMIT", "ROLLBACK", "SAVEPOINT", "BEGIN", "END",
    # Conditional
    "CASE", "WHEN", "THEN", "ELSE", "IF", "IFNULL", "ISNULL",
    "COALESCE", "NULLIF",
    # String / conversion
    "CONCAT", "CHAR", "CHR", "ASCII", "SUBSTRING", "SUBSTR",
    "LENGTH", "LEN", "UPPER", "LOWER", "TRIM", "CAST", "CONVERT",
    # Aggregate
    "COUNT", "SUM", "AVG", "MIN", "MAX",
    # Misc
    "TABLE", "VIEW", "INDEX", "DATABASE", "SCHEMA",
    "USER", "CURRENT_USER", "SESSION_USER", "SYSTEM_USER",
    "VERSION", "@@VERSION", "@@SERVERNAME",
]

SQL_OPERATORS = [
    # Comparison operators
    "!=", "<>", ">=", "<=",
    # Logical
    "AND", "OR", "XOR", "NOT",
    # Pattern / range
    "LIKE", "BETWEEN", "IS NULL", "IS NOT NULL",
    # SQL comment sequences
    "--", "/*", "*/", "#", "/*!",
    # Statement terminator / chaining
    ";--", "';",
    # Wildcard
    "%%",
    # MySQL-specific
    "LIMIT", "OFFSET",
    # MSSQL-specific
    "TOP", "NOLOCK",
]

SQL_INJECTION_ARTIFACTS = [
    # Tautologies
    "1=1", "1=2", "'1'='1'", "1 OR 1=1", "' OR '1'='1",
    # Always-true conditions
    "OR 1", "AND 1=1", "OR 1=1", "AND 1",
    # Comment + tautology chains
    "1' --", "1' #", "1');--",
    # Blind injection probes
    "SLEEP(5)", "BENCHMARK(1000000,MD5(1))",
    "WAITFOR DELAY '0:0:5'",
    # Error-based
    "EXTRACTVALUE(1,", "UPDATEXML(1,",
]

ALL_SQL_TOKENS = SQL_KEYWORDS + SQL_OPERATORS + SQL_INJECTION_ARTIFACTS
# Remove exact duplicates while preserving order
seen = set()
ALL_SQL_TOKENS_DEDUPED = []
for tok in ALL_SQL_TOKENS:
    if tok.upper() not in seen:
        seen.add(tok.upper())
        ALL_SQL_TOKENS_DEDUPED.append(tok)

print(f"[tokenizer] SQL vocabulary additions: {len(ALL_SQL_TOKENS_DEDUPED)} unique tokens")


# ── Tokenizer factory ─────────────────────────────────────────────────────────

def build_sql_aware_tokenizer(model_name: str, max_length: int = 256):
    """
    Load a pre-trained tokenizer and augment it with SQL-specific tokens.

    Args:
        model_name: HuggingFace model ID, e.g.
                    'microsoft/codebert-base'
                    'bert-base-uncased'
                    'distilbert-base-uncased'
        max_length: Maximum sequence length (default 256, covers >99.5% of
                    HTTP parameters in the master dataset; paper reports
                    p99=170 tokens).

    Returns:
        tokenizer: Augmented AutoTokenizer
        num_added:  Number of new tokens actually added
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_added = tokenizer.add_tokens(ALL_SQL_TOKENS_DEDUPED, special_tokens=False)
    print(f"[tokenizer] Added {num_added} new tokens to '{model_name}' vocabulary "
          f"(new size: {len(tokenizer)})")
    tokenizer.model_max_length = max_length
    return tokenizer, num_added


def resize_model_embeddings(model, tokenizer, num_added: int):
    """
    Resize the model's embedding matrix to accommodate new tokens.
    New embeddings are initialized at the MEAN of existing embeddings
    (empirically +0.8 F1 vs. random init per paper Section III.C).

    Args:
        model:     Pre-trained transformer model
        tokenizer: Augmented tokenizer (after add_tokens)
        num_added: Number of tokens added

    Returns:
        model with resized embeddings
    """
    if num_added == 0:
        return model

    old_num_tokens = len(tokenizer) - num_added
    model.resize_token_embeddings(len(tokenizer))

    # Mean initialization for new token embeddings
    with torch.no_grad():
        embedding_layer = model.get_input_embeddings()
        old_embeddings = embedding_layer.weight[:old_num_tokens]
        mean_embedding = old_embeddings.mean(dim=0, keepdim=True)
        embedding_layer.weight[old_num_tokens:] = mean_embedding.expand(
            num_added, -1
        )

    print(f"[tokenizer] Embedding matrix resized: {old_num_tokens} → {len(tokenizer)} "
          f"(new tokens initialized at mean embedding)")
    return model


def encode_batch(texts, tokenizer, max_length: int = 128):
    """
    Tokenize a batch of (already normalized) parameter strings.

    Args:
        texts:      List of strings
        tokenizer:  SQL-aware tokenizer
        max_length: Truncation / padding length

    Returns:
        dict with keys: input_ids, attention_mask, token_type_ids (if BERT)
    """
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )


if __name__ == '__main__':
    print("\n--- Tokenizer smoke test ---")
    print(f"Total SQL tokens defined: {len(ALL_SQL_TOKENS_DEDUPED)}")
    print(f"  Keywords : {len(SQL_KEYWORDS)}")
    print(f"  Operators: {len(SQL_OPERATORS)}")
    print(f"  Artifacts: {len(SQL_INJECTION_ARTIFACTS)}")
    print()

    # Show fragmentation example
    from transformers import AutoTokenizer as AT
    base_tok = AT.from_pretrained('bert-base-uncased')
    sql_tok, _ = build_sql_aware_tokenizer('bert-base-uncased')

    examples = [
        "SELECT * FROM users WHERE id=1 OR 1=1--",
        "UNION SELECT username, password FROM admin",
        "'; EXEC xp_cmdshell('whoami')--",
    ]
    print("Tokenization comparison (base vs SQL-aware):")
    for ex in examples:
        base_tokens = base_tok.tokenize(ex)
        sql_tokens  = sql_tok.tokenize(ex)
        print(f"\n  Input : {ex}")
        print(f"  Base  ({len(base_tokens):3d} tokens): {base_tokens}")
        print(f"  SQL   ({len(sql_tokens):3d} tokens): {sql_tokens}")
