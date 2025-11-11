from radiate.llm import LLMClient
from radiate.core import Radiate

#calling the llm
llm=LLMClient(provider="openrouter",api_key='KEY',
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