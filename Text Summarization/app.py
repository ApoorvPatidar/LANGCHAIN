import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title = "LangChain: Summarize Text from Any Link", page_icon = "📄")
st.title("LangChain: Summarize Text from Any Link")
st.subheader("Summarize URL")



llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    reasoning_format = "parsed"
)

template = """
You are a helpful assistant that summarizes the content of the provided text,
Text: {text}

You will be given a text, and you should summarize the in a concise and informative manner.
Make sure to include the main points, key details, and any important information that would help someone
understand the content eazily.
The summary should be clear, well-structured, and easy to read.
"""

prompt_template = PromptTemplate(
    input_variables = ["text"],
    template = template
)

url = st.text_input("Enter the URL to summarize:", label_visibility="collapsed")

if st.button("Summarize"):
    if not url.strip():
        st.error("Please enter a URL to summarize.")
    else:
        if validators.url(url):
            st.success("Valid URL")
            if "youtube.com" in url:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            else:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    }
                loader = UnstructuredURLLoader(urls = [url], ssl_verify=False, headers=headers)
            docs = loader.load()

            chain = load_summarize_chain(
                llm=llm,
                chain_type="stuff",
                prompt=prompt_template,
                verbose=True
            )
            summary = chain.run(docs)
            st.success(summary)
        else:
            st.error("URL is not reachable or returned a non-200 status code.")