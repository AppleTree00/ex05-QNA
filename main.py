import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# 👉 변경점 1: TextLoader, Docx2txtLoader 추가 임포트
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 세션 상태 초기화
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 2. 사이드바 - 설정 및 드래그 앤 드롭 업로드 영역
with st.sidebar:
    st.header("📂 디렉토리 문서 업로드")
    
    # 문서 최대 허용 개수 리스트박스 (50부터 100까지 10단위)
    max_files = st.selectbox(
        "최대 허용 문서 개수", 
        options=[50, 60, 70, 80, 90, 100],
        index=0 # 기본값 50
    )
    
    # 👉 변경점 2: type 파라미터에 "txt", "docx", "doc" 추가
    uploaded_files = st.file_uploader(
        "📁 아래 점선 영역에 '폴더'를 통째로 드래그 앤 드롭하세요!", 
        type=["pdf", "txt", "docx", "doc"], 
        accept_multiple_files=True,
        help="탐색기에서 폴더를 끌어다 놓으면 폴더 안의 지원되는 파일들이 자동으로 선택됩니다."
    )

    if st.button("🚀 문서 학습 시작", type="primary"):
        if uploaded_files:
            files_to_process = uploaded_files[:max_files]
            
            with st.spinner(f"총 {len(files_to_process)}개의 문서를 분석하고 있습니다..."):
                all_pages = []
                
                for uploaded_file in files_to_process:
                    # 👉 변경점 3: 파일 확장자 추출
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    # 확장자에 맞춰 임시 파일 생성
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # 👉 변경점 4: 확장자에 따른 로더 분기 처리
                        if file_extension == ".pdf":
                            loader = PyPDFLoader(tmp_file_path)
                        elif file_extension == ".txt":
                            loader = TextLoader(tmp_file_path, encoding="utf-8")
                        elif file_extension in [".docx", ".doc"]:
                            loader = Docx2txtLoader(tmp_file_path)
                        else:
                            st.warning(f"지원하지 않는 파일 형식입니다: {uploaded_file.name}")
                            continue
                            
                        pages = loader.load_and_split()
                        all_pages.extend(pages)
                        
                    except Exception as e:
                        st.error(f"{uploaded_file.name} 처리 중 오류가 발생했습니다: {str(e)}")
                    finally:
                        # 메모리 관리를 위해 처리 후 즉시 임시 파일 삭제
                        if os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)

                if not all_pages:
                    st.error("처리된 문서 내용이 없습니다. 파일 내용을 확인해주세요.")
                    st.stop()

                # 텍스트 분할
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=200,
                    chunk_overlap=20,
                    length_function=len,
                    is_separator_regex=False,
                )
                texts = text_splitter.split_documents(all_pages)

                # 임베딩 및 벡터 DB 생성
                embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
                db = Chroma.from_documents(texts, embeddings_model)

                # LLM 및 Retriever 설정
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                retriever_from_llm = MultiQueryRetriever.from_llm(
                    retriever=db.as_retriever(), 
                    llm=llm
                )

                # 프롬프트 설정
                system_prompt = (
                    "너는 질문-답변을 돕는 유능한 비서야. "
                    "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
                    "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
                    "{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # RAG 체인 생성 및 세션 저장
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)
                
                st.session_state.rag_chain = rag_chain
                
            st.success(f"선택하신 {len(files_to_process)}개의 문서 학습이 완료되었습니다!")
        else:
            st.warning("업로드된 파일이 없습니다. 폴더나 파일을 먼저 끌어다 놔주세요.")


# 3. 메인 화면 - 질문 및 답변 인터페이스
st.title("문서 내용 정리 요약봇 🤖")

# 파일이 업로드되어 RAG 체인이 준비된 경우에만 입력창 표시
if st.session_state.rag_chain is not None:
    question = st.text_input("업로드된 문서에 대해 뭐든지 물어보세요.")

    if st.button("궁금증 해결해줘", type="primary", icon="💡"):
        if question:
            with st.spinner("답변을 생성하는 중입니다...", show_time=True):
                response = st.session_state.rag_chain.invoke({"input": question})
                st.success("완료")
                st.write(response['answer'])
        else:
            st.warning("질문을 먼저 입력해주세요.")
else:
    st.info("👈 먼저 왼쪽 사이드바 점선 영역에 파일이 담긴 폴더를 드래그 앤 드롭하고 '문서 학습 시작' 버튼을 눌러주세요.")