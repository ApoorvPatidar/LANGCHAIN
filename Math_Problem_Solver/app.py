import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import llm_math
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title = "LangChain: Math Problem Solver", page_icon = "📄")
st.title("LangChain: Math Probelem Solver")
st.subheader("Math Probelem")


llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    reasoning_format = "parsed"
)

api_wrapper_wiki = WikipediaAPIWrapper()
wiki = Tool(
    name = 'Wikipedia',
    func = api_wrapper_wiki.run,
    description = "A tool for searching the internet and finding the various information for solving the math problem"
)

# calcr = llm_math.LLMMathChain.from_llm(llm)
# calc = Tool(
#     name = "Calculator",
#     func = calcr.run,
#     description = "A tool to perform calculations. It can be used to solve math problems that require calculations."
# )


template = """
You are a helpful assistant that solves the math problem provided,
Problem: {text}

You will be given a math problem, and you should provide the correct solution to that problem.
Make sure to include the formulas, key details, and any important information that would help someone
understand the solution eazily.
"""

prompt_template = PromptTemplate(
    input_variables = ["text"],
    template = template
)

chain = prompt_template | llm

reasoning_tool = Tool(
    name = "Reasoning Tool",
    func = chain.invoke,
    description = "A tool to solve logic and reasoning based math problems. It will provide a detailed solution to the problem, including the steps taken to arrive at the answer."
)

math_agent = initialize_agent(
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools = [wiki, reasoning_tool],
    llm = llm,
    verbose = True,
    handle_parsing_errors = True
)

# problem = st.text_input("Enter the problem: ", label_visibility="collapsed")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "What are we solving today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if problem := st.chat_input(placeholder="Please enter the problem to be solved"):
    with st.spinner("Thinking..."):
        st.session_state.messages.append({"role": "user", "content": problem})
        st.chat_message("user").write(problem)
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = math_agent.invoke({"input": problem}, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response["output"] if isinstance(response, dict) and "output" in response else response})
        st.write('### Response:')
        st.success(response["output"] if isinstance(response, dict) and "output" in response else response)
    