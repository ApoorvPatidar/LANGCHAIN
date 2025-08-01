'''
FastAPI application using LangChain and Groq for language translation.
'''

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes

import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="deepseek-r1-distill-llama-70b", reasoning_format="parsed")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following to {output_lang}:"),
        ("human", "{text}"),
    ]
)

parser = StrOutputParser()

# Create the Chain
chain = prompt_template | model | parser

## App Definition
app = FastAPI(title="Language Translation API", 
              version = 1.0, 
              description="API for translating text using Groq LLM")

## Adding chain routes
add_routes(
    app ,
    chain,
    path = "/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
