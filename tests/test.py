'''from radiate.core import Radiate
radiate=Radiate(embedding_provider='local')
text=radiate.ingest('ml-book.pdf')

result= radiate.search('what are api endpoints',mode='hybrid')

for r in result:
    print(f"Score: {r['score']:.4f}. Text: {r['text'][:200]} \n")

print('------------------------------')
result2=radiate.query('what are api endpoints',mode='hybrid')
for r in result:
    print(f"Score: {r['score']:.4f}. Text: {r['text'][:200]}\n")

    '''

#async vs sync
'''
from radiate import Radiate

radiate= Radiate(embedding_provider='local',collection_name='new_pdf_collection')
result=radiate.ingest('test_data/ml-book.pdf')
print(f"Ingested {result['chunks_ingested']} chunks")
'''


import asyncio
from radiate import Radiate

async def main():
    radiate = Radiate(embedding_provider="local")
    result = await radiate.ingest_async('test_data/ml-book.pdf')
    print(f"Ingested {result['chunks_ingested']} chunks")
    print(f"{result['text'][:200]}")

asyncio.run(main())
