from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import re

def is_valid_youtube_url(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    match = re.match(youtube_regex, url)
    return bool(match)

def load_documents(link):
    if not is_valid_youtube_url(link):
        return None
    loader = YoutubeLoader.from_youtube_url(
        link,
        add_video_info=True,
        language=["en","hi","id"],
        translation="en",
    )
    documents = loader.load()
    if not documents:
        st.error("No documents loaded from YouTube URL.")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ".", ",", ""],
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(documents)
    if not splits:
        st.error("Document splitting failed.")
    return splits

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def define_vector_store(splits):
    # Ensure splits are not empty
    if not splits:
        st.error("No document splits available for FAISS indexing.")
        return None

    embeddings = embedding_function.embed_documents(splits)
    if not embeddings:
        st.error("Embedding generation failed.")
        return None
    
    # load it into FAISS
    db = FAISS.from_documents(documents=splits, embedding=embedding_function)
    return db

def retriever_chain(db):
    retriever = db.as_retriever()

    # Prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # llm
    HUGGINGFACEHUB_API_TOKEN = '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'
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 1024}, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def main():
    st.title('Chat with YouTube Video')

    # Sidebar for YouTube URL input
    st.sidebar.header("YouTube URL")
    youtube_url = st.sidebar.text_input("Enter YouTube URL")

    # Process URL button
    if st.sidebar.button("Process URL"):
        if youtube_url:
            st.sidebar.write("Processing URL:", youtube_url)
            if is_valid_youtube_url(youtube_url):
                st.sidebar.write('URL is valid')
            else:
                st.sidebar.write('URL is invalid')
    
    question = st.text_input('Enter your query')
    if st.button("Ask Question"):
        if question:
            documents = load_documents(youtube_url)
            if documents:
                split = split_documents(documents)
                if split:
                    db = define_vector_store(split)
                    if db:
                        rag_chain = retriever_chain(db)
                        answer = rag_chain.invoke(question)
                        idx = answer.find('Question:')
                        answer = answer[idx:]
                        st.write(answer)
        else:
            st.warning("Please enter a question")

if __name__ == '__main__':
    main()
