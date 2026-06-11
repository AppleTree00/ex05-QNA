import os
import tempfile
import streamlit as st

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# 문서 Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma Vector DB
from langchain_chroma import Chroma


# OpenAI 모델
from langchain_openai import ( OpenAIEmbeddings,   ChatOpenAI )

# 최신 Retrieval Chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate

# Prompt
from langchain_core.prompts import (  ChatPromptTemplate )

st.title("PDF File Reader")
st.write("----------------")

uploaded_file = st.file_uploader(   "PDF 파일을 선택하세요",    type=["pdf"] )
st.write("----------------")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()

    temp_filepath = os.path.join( temp_dir.name,  uploaded_file.name    )

    with open(   temp_filepath,    "wb"   ) as f:
        f.write(  uploaded_file.getvalue()   )

    loader = PyPDFLoader(    temp_filepath   )
    pages = loader.load()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(  uploaded_file   )
    st.success(  f"PDF 로딩 완료 : {len(pages)} 페이지"   )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = text_splitter.split_documents( pages  )
    st.info( f"분할된 문서 개수 : {len(texts)}"    )

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(

        documents=texts,

        embedding=embeddings

    )
    retriever = db.as_retriever(

        search_kwargs={
            "k":3
        }

    )

    st.header(
        "PDF에게 질문하세요"
    )

    question = st.text_input(
        "질문 입력"
    )

    if st.button(
        "질문하기"
    ):

        if question:
            with st.spinner(
                "답변 생성 중..."
            ):
                llm = ChatOpenAI(

                    model="gpt-4.1-mini",

                    temperature=0
                )
                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 분석 전문가입니다.

                    아래 Context만 이용해서
                    질문에 답하세요.

                    Context:
                    {context}

                    질문:
                    {input}

                    답변:
                    """
                )
                document_chain = (
                    create_stuff_documents_chain(
                        llm,
                        prompt
                    )
                )

                qa_chain = create_retrieval_chain(
                    retriever,
                    document_chain
                )

                response = qa_chain.invoke(
                    {
                        "input": question
                    }

                )

                # 답변 출력
                st.write(
                    response["answer"]
                )

        else:
            st.warning(
                "질문을 입력하세요."
            )