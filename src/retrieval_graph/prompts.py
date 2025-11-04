"""Default prompts."""

RESPONSE_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.

{retrieved_docs}

System time: {system_time}"""
QUERY_SYSTEM_PROMPT = """Generate between 1 and {max_queries} focused search queries to retrieve documents that may help answer the user's question. When it is helpful, decompose complex requests into multiple narrower sub-queries. Previously, you made the following queries:
    
<previous_queries/>
{queries}
</previous_queries>

Return concise keyword-style queries without conversational phrasing.

System time: {system_time}"""
GRADE_SYSTEM_PROMPT = """You are a document relevance grader. Use the user's latest question, the active search query, and the provided document metadata and content to decide if the document can help answer the question. Respond with a `binary_score` of 'yes' or 'no' in the structured output. Only answer 'yes' when the document contains information that directly addresses the question; otherwise reply 'no'."""
REWRITE_QUERY_SYSTEM_PROMPT = """You improve search queries when the previous retrieval returned irrelevant results. Craft between 1 and {max_queries} concise query strings that will fetch documents useful for answering the user's question. Consider the most recent queries, the discarded document summaries, and avoid conversational phrasing. Return only the revised query strings in the structured output."""
