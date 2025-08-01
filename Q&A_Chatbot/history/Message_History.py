from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uuid
import random as rd

summary_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a summarizer bot that summarizes the conversation into a single sentence."),
    ("human", "{input}")
])
summary_chain = prompt | summary_model | StrOutputParser()


class History:
    def __init__(self):
        self.storage = {}
        self.summaries = {}
    

    # Returns the chat history for a given session ID
    def existing_chat_access(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.storage:
            raise ValueError(f"Session {session_id} does not exist.")
        return self.storage[session_id]


    # For making a new chat session
    def new_chat(self):
        while True:
            session_id = str(uuid.uuid4())
            if session_id not in self.storage:
                self.storage[session_id] = ChatMessageHistory()
                self.summaries[session_id] = None  # To be set after the first message
                return session_id
    

    # Updates summary for a session if not already set
    def chat_summary(self, session_id: str, message: str) -> None:
        if self.summaries[session_id] is None:
            self.summaries[session_id] = self.summarize_message(message)
    

    # Uses the LLM chain to create a summary
    def summarize_message(self, message: str) -> str:
        chat_summary = summary_chain.invoke({"input": message})
        return chat_summary
    

    # For UI: List all session IDs and their chat summaries
    def list_sessions(self) -> list[tuple[str, str | None]]:
        return [(sid, self.summaries[sid]) for sid in self.storage]
    
    
    # Check if a session exists
    def session_exists(self, session_id: str) -> bool:
        return session_id in self.storage
    
    
    # Get session count
    def get_session_count(self) -> int:
        return len(self.storage)