
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from models import create_vector_db



def build_rag_chain(vector_db, template, return_metadata = True, temperature = 0, model_name="gpt-4o"):
    retriever=vector_db.as_retriever(return_metadata=return_metadata)
    prompt=ChatPromptTemplate.from_template(template)
    llm=ChatOpenAI(model=model_name,temperature=temperature)
    rag_chain=(
        {"context":retriever,"question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
