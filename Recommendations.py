# Imports
import bs4
import random
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableParallel
from openai import OpenAI
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import streamlit as st
from streamlit_feedback import streamlit_feedback

OPENAI_key = st.secrets["OPENAI_key"]
WHISPER_key = st.secrets["WHISPER_key"]

os.environ["OPENAI_API_KEY"] = OPENAI_key
os.environ["WHISPER_API_KEY"] = WHISPER_key 

# Intialize APIs
agent1 = ChatOpenAI(openai_api_key=OPENAI_key, model ='gpt-4o')
tts_client = OpenAI(api_key = OPENAI_key)


# Make Agent1
agent1_prompt = '''
You are an oncology nurse providing remote self care advice to cancer patients about symptoms and side effects that may occur during cancer treatment. Please read the patient's last three personal journal posts and provide a summary of what you have learned.
Then, provide self care instructions for each of the symptoms and side effects on the Impact Symptoms List.
Your goal is to mitigate and reduce the severity impact of these symptoms as best you can by providing self care advice.
By lowering the impact of each specific symptom your goal is to improve each patient's Quality of Life scores measured by the EQ-5D.
Please be supportive in your tone at all times.
Each symptom care instruction response should be easy to understand and be no more than 150 words.

'''

# Agent 1
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=agent1_prompt
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{reports_input}"
        ),  # Where the reports list will be input
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k= 5)

agent1_chain = LLMChain(
    llm=agent1,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

#### Begin Streamlit app ####

st.title("♋ CancerLife")
st.divider()
st.markdown('<center><h2>Recommendations</h2></center>', unsafe_allow_html=True)
st.markdown('*This is a demo of Agent 1*')

# Initialize State Variables
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm Mary Beth, your Cancer Health Coach. I'm an AI trained resource to help you manage your symptoms and navigate your cancer journey."}
    ]

if "response" not in st.session_state:
    st.session_state["response"] = None


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "assistant":
        avt = 'Health_Coach_MaryBeth.jpeg'
    else:
        avt = None
    with st.chat_message(message["role"], avatar= avt ):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your question"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Access LLM
    response = agent1_chain.predict(reports_input=prompt)
    
    # Create Audio
    speech_file_path =  "sample_speech.mp3"
    audio_response = tts_client.audio.speech.create(
    model="tts-1-hd",
    voice="shimmer",
    input=response
    )

    audio_response.stream_to_file(speech_file_path)

    st.session_state["response"] = response
    with st.chat_message("assistant", avatar='Health_Coach_MaryBeth.jpeg'):
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["response"]})
        st.write(st.session_state["response"])

    st.audio(speech_file_path)

if st.session_state["response"]:
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
        st.audio(speech_file_path)

