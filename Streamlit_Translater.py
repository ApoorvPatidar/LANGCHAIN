from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st


def query(input_lang, output_lang, text):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Translate the following text from {input_lang} to {output_lang}:"),
            ("human", "{text}"),
        ]
    )

    llm = ChatGroq(model="deepseek-r1-distill-llama-70b",
                reasoning_format="parsed")
    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "input_lang": input_lang, 
            "output_lang": output_lang, 
            "text": text
        }
    )

st.title("Language Translation with Groq LLM")
input_lang = st.text_input("Enter input language:")
output_lang = st.text_input("Enter output language: ")
text = st.text_input("Enter text to translate: ")

if input_lang and output_lang and text:
    st.write("Translation:")
    st.success("Please wait, this may take a few seconds...")
    with st.spinner("Translating..."):
        # Call the query function and display the result
        st.write("Result:", query(input_lang, output_lang, text))
