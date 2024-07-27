import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from htmlTemplates import css
from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain.docstore.document import Document

load_dotenv()
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def add_to_vectorstore(texts,store):
    product_documents = []
    for product in texts:
        product_documents.append(
            Document(
                page_content=product,
                metadata={
                    
                }
            )
        )
    store.add_documents(product_documents, batch_size=100, embedding_chunk_size=200)
def main():
    load_dotenv()
    st.set_page_config(page_title="Add data to OrgocatAI",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Add data to OrgocatAI :books:")
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            
            # get Upstash credentials from environment variables
            upstash_url = os.environ.get("UPSTASH_VECTOR_REST_URL")
            upstash_token = os.environ.get("UPSTASH_VECTOR_REST_TOKEN")
            print(upstash_url)

            # create vector store
            embeddings = OpenAIEmbeddings()
            store = UpstashVectorStore(
                embedding=embeddings
            )
            store = UpstashVectorStore(embedding=OpenAIEmbeddings())
            add_to_vectorstore(text_chunks,store)
            


if __name__ == '__main__':
    main()
