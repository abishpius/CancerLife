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

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

#### Begin Streamlit app ####

st.title("♋ CancerLife")
st.divider()
st.markdown('<center><h2>Symptoms</h2></center>', unsafe_allow_html=True)
st.markdown('*This is a demo of Agent Responding to Symptoms*')

agent_symp = ChatOpenAI(openai_api_key=OPENAI_key, model ='gpt-4-1106-preview')
tts_client_symp = OpenAI(api_key = OPENAI_key)

# List of symptoms for the multiselect widget
symptoms = [
    "Achieve and maintain erection",
    "Acne",
    "Anxiety",
    "Back Pain",
    "Bleeding",
    "Bloating",
    "Blood in Stool",
    "Blood in Urine",
    "Blurred vision",
    "Chest Pain",
    "Chills",
    "Confusion",
    "Constipation",
    "Cough",
    "Decreased appetite",
    "Decreased libido",
    "Delayed orgasm",
    "Difficulty swallowing",
    "Dry mouth",
    "Fatigue",
    "Feeling Down",
    "Feeling Stressed",
    "Hair Loss",
    "Headaches",
    "Insomnia",
    "Joint pain",
    "Lightheadedness",
    "Muscle pain",
    "Nausea",
    "Neuropathy",
    "Numbness",
    "Pain",
    "Shortness of Breath",
    "Skin Change",
    "Trouble Walking",
    "Weakness"
]


# Create a form in Streamlit
with st.form(key='symptom_form'):
    selected_items = st.multiselect('The following symptoms have been reported by you over the last 30 days. Which ones are impacting you the most or you would like help with?', symptoms)
    submit_button = st.form_submit_button(label='Submit', on_click=click_button)


if st.session_state.clicked:
    # Make Agent1
    agent_symp_prompt_part1 = '''
    You are an oncology nurse providing remote self care advice to cancer patients about symptoms and side effects that may occur during cancer treatment. Please read the patient's most concerning symptoms here'''

    agent_symp_prompt_part2 = '\n'.join(selected_items)

    agent_symp_prompt_part3 = '''
    Then, provide self care instructions for each of the symptoms and side effects on the Impact Symptoms List.
    Your goal is to mitigate and reduce the severity impact of these symptoms as best you can by providing self care advice.
    By lowering the impact of each specific symptom your goal is to improve each patient's Quality of Life scores measured by the EQ-5D.
    Please be supportive in your tone at all times.
    Each symptom care instruction response should be easy to understand and be no more than 150 words.
    '''
    agent_symp_prompt = agent_symp_prompt_part1 + agent_symp_prompt_part2 + agent_symp_prompt_part3

    # Symptoms Agent
    prompt_symp = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=agent_symp_prompt
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

    agent_symp_chain = LLMChain(
        llm=agent_symp,
        prompt=prompt_symp,
        verbose=False,
        memory=memory,
    )

    response = agent_symp_chain.predict(reports_input="My most concerning symptoms are" + '\n'.join(selected_items))


    # Initialize State Variables
    formatted_string = "Your most concerning symptoms are: " + ','.join(selected_items)
    st.session_state["messages_symptoms"] = [
        {"role": "user", "content": formatted_string},
        {"role": "assistant", "content": response}
    ]

    st.session_state["response_symptoms"] = response

    # Create Audio
    speech_file_path =  "sample_speech.mp3"
    audio_response = tts_client_symp.audio.speech.create(
    model="tts-1-hd",
    voice="shimmer",
    input=response
    )

    audio_response.stream_to_file(speech_file_path)

    st.audio(speech_file_path)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages_symptoms:
        if message["role"] == "assistant":
            avt = '👩‍⚕️'
        else:
            avt = None
        with st.chat_message(message["role"], avatar= avt ):
            st.markdown(message["content"])
    
    #     # React to user input
    # if prompt := st.chat_input("Enter your question"):
    #     # Display user message in chat message container
    #     st.chat_message("user").markdown(prompt)
    #     # Add user message to chat history
    #     st.session_state.messages.append({"role": "user", "content": prompt})
        
    #     # Access LLM
    #     response = agent_symp_chain.predict(reports_input=prompt)
        
    #     # Create Audio
    #     speech_file_path =  "sample_speech.mp3"
    #     audio_response = tts_client_symp.audio.speech.create(
    #     model="tts-1-hd",
    #     voice="fable",
    #     input=response
    #     )

    #     audio_response.stream_to_file(speech_file_path)

    #     st.session_state["response"] = response
    #     with st.chat_message("assistant", avatar='👩‍⚕️'):
    #         st.session_state.messages.append({"role": "assistant", "content": st.session_state["response"]})
    #         st.write(st.session_state["response"])

    #     st.audio(speech_file_path)

    if st.session_state["response_symptoms"]:
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{len(st.session_state.messages_symptoms)}",
        )
        # Add thumbs up thumbs down feedback
        if feedback:
            with open("feedback.txt", "a", encoding="utf-8") as file:
                # Write the feedback to the file
                file.write("Begin Feedback \n")
                file.write(f"Score: {feedback['score']} ; Reason: {feedback['text']} ; Message: {st.session_state['response_symptoms']}" + "\n")
                file.write("End Feedback \n")
            
            st.toast("Feedback recorded!", icon="📝")

