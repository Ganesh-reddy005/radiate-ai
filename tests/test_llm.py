from radiate.llm import LLMClient
from radiate.core import Radiate

#calling the llm
llm=LLMClient(provider="openrouter",api_key='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',
              model='nvidia/nemotron-nano-12b-v2-vl:free')

#using embedding model
radiate=Radiate(embedding_provider='local')
#ingesting our source 
radiate.ingest('test_data/ml-book.pdf')


user_query='what is machine learning'

#coding and decoding resonse
result=radiate.query(user_query,mode='hybrid',top_k=3)

llm_output=llm.answer(user_query,result['text'])

print(llm_output)