import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate

# -------------------- Environment Setup --------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Electronics + AI: Paper Q&A",
    page_icon="🤖",
    layout="wide"
)

# -------------------- Header UI --------------------
st.markdown("<h1 style='text-align: center; color: #0A9396;'>🤖 Electronics & AI Research Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask technical questions based on the paper <em>“Overview of Emerging Electronics Technologies for AI / LLM Ideas”</em></p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- LLM and Prompt --------------------
llm = ChatGroq(model_name='llama-3.1-8b-instant', groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
"""
)

# -------------------- Vector Embedding Function --------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("🔄 Loading and embedding 'ELECTRONICSwithAI.pdf'..."):
            st.session_state.embeddings = OpenAIEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("✅ Vector Database is ready!")

# -------------------- Input UI --------------------
st.markdown("<h3 style='color:#333;'>🧠 Ask something about Electronics and AI:</h3>", unsafe_allow_html=True)
user_prompt = st.text_input("", key="query_input")

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("📂 Create Vector Embeddings"):
        create_vector_embedding()

# -------------------- Response Generation --------------------
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please embed the document first by clicking 'Create Vector Embeddings'")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        response_time = time.process_time() - start

        st.success(f"✅ Answer generated in {response_time:.2f} seconds")

        st.markdown("### 💡 Answer")
        st.write(response['answer'])

        with st.expander("📄 Document Chunks Used in Answer"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("--------------------------------------------------")
