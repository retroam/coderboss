from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pinecone
import os


load_dotenv()
llm = OpenAI(temperature=0)


st.title('GitHub Repo Explorer')

# Input field for GitHub repo
repo = st.text_input('Enter a GitHub repo URL')

if repo:
    # Show a loading message while the repository is being cloned
    with st.spinner('Loading GitHub repo...'):
        loader = GitLoader('repo', 
                   clone_url=repo,
                  branch='master')

        documents = loader.load()
        loader = GitLoader('repo', 
                   clone_url='https://github.com/retroam/tinygrad',
                  branch='master')

        documents = loader.load()
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment="us-west4-gcp-free")
        vectordb = Pinecone.from_documents(texts, embeddings, index_name='langchain-demo')

    # Once the repo is loaded, we can initialize the bot
    bot = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectordb.retriever()
)

    # Chatbot interface
    st.title('Chatbot')
    query = st.text_input('Ask a question about the repo')

    if query:
        with st.spinner('Fetching answer...'):
            answer = bot.run(input_docs = docs, query=query)

        st.write('Answer:', answer)
