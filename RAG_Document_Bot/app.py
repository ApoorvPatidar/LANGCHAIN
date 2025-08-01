import streamlit as st
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

# --- LLM and Prompt ---
model = ChatGroq(model_name="deepseek-r1-distill-llama-70b", reasoning_format="parsed")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the query.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# --- Vector Creation Function ---
def create_vectors_embedding():
    if not st.session_state.get("vector", False):
        # Embedding model
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")
        
        # Load PDFs
        loader = PyPDFDirectoryLoader(path="Research_Papers")
        documents = loader.load()

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = splitter.split_documents(documents)

        # Store vector DB in session
        st.session_state.vector_stores = FAISS.from_documents(final_docs, st.session_state.embeddings)
        st.session_state.vector = True  # ✅ Important: flag to avoid rerunning
        st.success("✅ Vector Database Created Successfully!")
    else:
        st.info("ℹ️ Vector database already exists in session.")

# --- Streamlit UI ---
st.title("📄 Research Paper Q&A Assistant")

user_prompt = st.text_input("🧠 Enter your question here:")

if st.button("📦 Create Document Embedding"):
    create_vectors_embedding()

# --- Query Execution ---
if user_prompt:
    if not st.session_state.get("vector", False):
        st.warning("⚠️ Please click 'Create Document Embedding' first.")
    else:
        with st.spinner("🤖 Thinking..."):
            document_chain = create_stuff_documents_chain(model, prompt)
            retriever = st.session_state.vector_stores.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.time()
            response = retrieval_chain.invoke({"input": user_prompt})
            elapsed = time.time() - start

            st.markdown("### 📝 Response")
            st.write(response["answer"])
            st.caption(f"⏱️ Response generated in {elapsed:.2f} seconds.")

            # Context display
            with st.expander("📚 Context Used"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.markdown("---")
