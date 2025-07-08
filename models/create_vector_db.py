from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore

import os


def store_as_vectordb(all_docs, es_cloud_id, es_api_key):
    embeddings=OpenAIEmbeddings()
    vector_db=ElasticsearchStore.from_documents(
        all_docs,
        embedding=embeddings,
        index_name="policy_docs",
        es_cloud_id= es_cloud_id,
        es_api_key= es_api_key
    )

    vector_db=ElasticsearchStore(
        embedding=embeddings,
        index_name="policy_docs",
        es_cloud_id= es_cloud_id,
        es_api_key= es_api_key
    )

    return vector_db