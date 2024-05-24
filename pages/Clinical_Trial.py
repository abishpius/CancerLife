import os
import random
import bs4
import streamlit as st
from openai import OpenAI
from streamlit_feedback import streamlit_feedback
from langchain import hub
from langchain.chains import create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma, FAISS


OPENAI_key = st.secrets["OPENAI_key"]
WHISPER_key = st.secrets["WHISPER_key"]

os.environ["OPENAI_API_KEY"] = OPENAI_key
os.environ["WHISPER_API_KEY"] = WHISPER_key 

st.session_state["file_uploaded"] = False

# Initialize chat history
if "clinical_trial_messages" not in st.session_state:
    st.session_state.clinical_trial_messages = []

if "clinical_response" not in st.session_state:
    st.session_state["clinical_response"] = None

# Intialize APIs
clinical_trials_agent = ChatOpenAI(openai_api_key=OPENAI_key, model ='gpt-4o')
tts_client = OpenAI(api_key = OPENAI_key)

#### Begin Streamlit app ####

st.title("♋ CancerLife")
st.divider()
st.markdown('<center><h2>Clinical Trials</h2></center>', unsafe_allow_html=True)
st.markdown('*This is a demo of Agent Chatbot against a Clincal Trial*')

clinical_trials_system_prompt = (
    "You are an assistant for question-answering tasks regarding a clinical trials document. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

clinical_trials_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", clinical_trials_system_prompt),
        ("human", "{input}"),
    ]
)

uploaded_file = st.file_uploader(
    "Upload a Clinical Trial pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="File must be machine readable and not a scanned image"
)

st.divider()

if uploaded_file is not None:

    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue())

    # Load PDF File
    loader = PyPDFLoader(uploaded_file.name)
    pages = loader.load_and_split()

    vectorstore = FAISS.from_documents(pages, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    question_answer_chain = create_stuff_documents_chain(clinical_trials_agent, clinical_trials_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)


    # Establish prior messages
    if len(st.session_state.clinical_trial_messages) < 1:
        
        st.session_state.clinical_trial_messages.append({"role": "assistant", "content": f"Hi, I'm Mary Beth, your Clinical Trials Coordinator. I'm an AI trained to help answer questions about the clinical trial {uploaded_file.name} you just uploaded."})
        with st.chat_message("assistant", avatar= 'Health_Coach_MaryBeth.jpeg' ):
            st.markdown(f"Hi, I'm Mary Beth, your Clinical Trials Coordinator. I'm an AI trained to help answer questions about the clinical trial '**{uploaded_file.name}**' you just uploaded.",unsafe_allow_html=True)
    else:
        for message in st.session_state.clinical_trial_messages:
            if message["role"] == "assistant":
                avt = 'Health_Coach_MaryBeth.jpeg'
            else:
                avt = None
            with st.chat_message(message["role"], avatar= avt ):
                st.markdown(message["content"])


    if prompt := st.chat_input("Enter your question"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.clinical_trial_messages.append({"role": "user", "content": prompt})

        # Inference
        response = rag_chain.invoke({"input": prompt})

        # Create Audio
        speech_file_path =  "sample_speech.mp3"
        audio_response = tts_client.audio.speech.create(
        model="tts-1-hd",
        voice="shimmer",
        input=response["answer"]
        )

        audio_response.stream_to_file(speech_file_path)

        st.session_state["clinical_response"] = response["answer"] + '\n\n**Sources:**'+ '\n\n' + '\n\n'.join([doc.metadata['source'] + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Page: ' + str(doc.metadata['page']) for doc in response['context']])

        with st.chat_message("assistant", avatar='Health_Coach_MaryBeth.jpeg'):
            st.session_state.clinical_trial_messages.append({"role": "assistant", "content": st.session_state["clinical_response"]})
            st.markdown(st.session_state["clinical_response"], unsafe_allow_html=True)

        st.audio(speech_file_path)

    # Feedback
    if st.session_state["clinical_response"]:
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{len(st.session_state.messages)}",
        )
        # Add thumbs up thumbs down feedback
        if feedback:
            with open("feedback.txt", "a", encoding="utf-8") as file:
                # Write the feedback to the file
                file.write("Begin Feedback \n")
                file.write(f"Score: {feedback['score']} ; Reason: {feedback['text']} ; Message: {st.session_state['response']}" + "\n")
                file.write("End Feedback \n")
            
            st.toast("Feedback recorded!", icon="📝")
            speech_file_path =  "sample_speech.mp3"
            st.audio(speech_file_path)

        

        
