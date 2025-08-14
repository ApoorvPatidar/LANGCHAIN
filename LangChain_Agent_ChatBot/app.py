import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

## Tools
api_runner_wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars=500)
wiki = WikipediaQueryRun(api_wrapper = api_runner_wiki)
api_runner_arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars=500)
arxiv = ArxivQueryRun(api_wrapper = api_runner_arxiv)
search = DuckDuckGoSearchRun(name = 'Search') # 'Search' Defines the name when the tool is used

st.secrets["GOOGLE_API_KEY"]

st.title("LangChain Agent")
'''
This is a simple LangChain Agent that can answer questions using RAG.
'''
st.sidebar.title("Agent Configuration")
# api_key = st.sidebar.text_input("Enter your Google API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", 
         "content": "How can I help you?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input(placeholder="Please enter your prompt"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-2.0-flash")
    tools = [wiki, arxiv, search]
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True ,verbose=True)
    with st.chat_message("assistant"):
        ''' Shows the thinking of the agent in the chat interface'''
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
