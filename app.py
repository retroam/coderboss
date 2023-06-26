from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pinecone
import os
import streamlit as st
from langchain.document_loaders import GitLoader
from urllib.parse import urlparse
from streamlit_chat import message


load_dotenv()
llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings(disallowed_special=())


@st.cache_resource
def load_repo_and_initialize_bot(repo):
    parsed = urlparse(repo)
    user, repo_name = parsed.path.strip('/').split('/')
    repo_name = repo_name.replace('.git', '')
    loader = GitLoader(repo_name, 
                       clone_url=repo,
                      branch='main')
    
    contents = loader.load()
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment="us-west4-gcp-free")
    vectordb = Pinecone.from_documents(contents, embeddings, index_name='langchain-demo')
    bot = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectordb.as_retriever()
)

    return loader, bot,vectordb

    
st.title('GitHub Repo Explorer')

# Input field for GitHub repo
repo = st.text_input('Enter a GitHub repo URL')

if repo:
    # Show a loading message while the repository is being cloned
    with st.spinner('Loading GitHub repo...'):
        loader, bot, vectordb = load_repo_and_initialize_bot(repo)


    # Chatbot interface
    st.title('Chatbot')
    query = st.text_input('Ask a question about the repo')

    if query:
        with st.spinner('Fetching answer...'):
            docs = vectordb.similarity_search(query)
            message(query, is_user=True)  # align's the message to the right
            answer = bot.run(input_docs = docs, query=query)

        message(answer)
