import os
from dotenv import load_dotenv
load_dotenv()

from history.Message_History import History

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


# Instantiate LLM model
model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash')

# Initialize session state for persistent history
if 'history_manager' not in st.session_state:
    st.session_state.history_manager = History()

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = st.session_state.history_manager.new_chat()

# Setup prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the conversation history to provide contextual responses."),
        ("human", "{input}"),
    ]
)




st.title("Q&A Chatbot with Conversation History")




# Sidebar for session management
with st.sidebar:
    st.header("Chat Sessions")
    
    # New Chat button
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.current_session_id = st.session_state.history_manager.new_chat()
        st.rerun()
    
    st.divider()
    
    # List existing sessions
    sessions = st.session_state.history_manager.list_sessions()
    if sessions:
        st.subheader("Previous Conversations")
        for session_id, summary in sessions:
            # Create a shortened display name
            display_name = summary if summary else f"Chat {session_id[:8]}..."
            if len(display_name) > 50:
                display_name = display_name[:47] + "..."
            
            # Highlight current session
            if session_id == st.session_state.current_session_id:
                st.write(f"**🟢 {display_name}** (Current)")
            else:
                if st.button(f"💬 {display_name}", key=f"session_{session_id}"):
                    st.session_state.current_session_id = session_id
                    st.rerun()




# Main chat interface
st.write(f"**Current Session:** {st.session_state.current_session_id[:8]}...")




# Display chat history for current session
try:
    current_history = st.session_state.history_manager.existing_chat_access(st.session_state.current_session_id)
    
    # Display conversation history
    if hasattr(current_history, 'messages') and current_history.messages:
        st.subheader("Conversation History")
        for message in current_history.messages:
            if message.type == "human":
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):
                    st.write(message.content)
except ValueError:
    st.error("Session not found. Creating a new session.")
    st.session_state.current_session_id = st.session_state.history_manager.new_chat()
    st.rerun()



# Chat input
chain = prompt | model | StrOutputParser()

with st.form("chat_input_form", clear_on_submit=True):
    user_input = st.text_input("Enter your message:", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send", use_container_width=True) 




if submitted and user_input:
    # Create chain with message history
    with_message_history = RunnableWithMessageHistory(
        chain, 
        st.session_state.history_manager.existing_chat_access
    )
    config = {"configurable": {"session_id": st.session_state.current_session_id}}
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = with_message_history.invoke({"input": user_input}, config)
            st.write(response)
    
    # Update chat summary if it's the first message
    st.session_state.history_manager.chat_summary(st.session_state.current_session_id, user_input)
    
    # Rerun to show the updated conversation
    st.rerun()

    