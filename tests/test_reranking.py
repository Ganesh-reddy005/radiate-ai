from radiate.llm import LLMClient
from radiate.core import Radiate

# Initialize LLM
llm = LLMClient(
    provider="openrouter",
    api_key='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',
    model='nvidia/nemotron-nano-12b-v2-vl:free'
)

# Initialize Radiate WITH reranker enabled
radiate = Radiate(
    embedding_provider='local',
    enable_reranker=True  # NEW: Enable reranker at initialization
)

# Ingest source
radiate.ingest('test_data/ml-book.pdf')

user_query = 'what is machine learning'

# Query WITH reranking
result = radiate.query(
    user_query,
    mode='hybrid',
    top_k=3,
     # NEW: Enable reranking for this query
)

# Generate answer
llm_output = llm.answer(user_query, result)

print("🔍 Context Used (Reranked):")
print("=" * 60)
for doc in result:
    print(f"- {doc['text'][:150]}...")
print("\n" + "=" * 60)

print("\n✨ LLM Answer:")
print(llm_output['answer'])
