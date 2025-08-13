import requests
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import load_summarize_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    reasoning_format = "parsed"
)

link = requests.get(st.text_input("Enter the link to the PDF file:"))
