#Streamlit is a free, open-source Python library that helps developers 
#and data scientists create interactive web applications for machine learning 
#and data science
import streamlit as st
from streamlit_chat import message


from langchain.chains import LLMChain


from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

# All utility functions
import utils




def initialize_session_state():
    """
    Session State is a way to share variables between reruns, for each user session.
    """

    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Hello! I am here to provide answers to Finance related questions and advices."])
    st.session_state.setdefault('past', ["Hello Buddy!"])

def create_conversational_chain(llm):

    template = """Act as a finance advisor, would respond in a conversation. Remember, it will use only the information provided in the "Relevant Information" section and will be honest if it doesn't know the answer.

            Relevant Information:

            {chat_history}

            Conversation:
            Human: {input}
            AI:"""
            
    prompt = PromptTemplate(
                input_variables=["chat_history", "input"], template=template
            )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chain = LLMChain(
                                llm=llm, 
                                verbose=True, 
                                prompt = prompt,
                                memory=memory
                            )
    return chain

def display_chat(conversation_chain):
    """
    Streamlit relatde code wher we are passing conversation_chain instance created earlier
    It creates two containers
    container: To group our chat input form
    reply_container: To group the generated chat response

    Args:
    - conversation_chain: Instance of LangChain ConversationalRetrievalChain
    """
    #In Streamlit, a container is an invisible element that can hold multiple 
    #elements together. The st.container function allows you to group multiple 
    #elements together. For example, you can use a container to insert multiple 
    #elements into your app out of order.
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me about Finance related questions", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
        
        #Check if user submit question with user input and generate response of the question
        if submit_button and user_input:
            generate_response(user_input, conversation_chain)
    
    #Display generated response to streamlit web UI
    display_generated_responses(reply_container)


def generate_response(user_input, conversation_chain):
    """
    Generate LLM response based on the user question by retrieving data from Vector Database
    Also, stores information to streamlit session states 'past' and 'generated' so that it can
    have memory of previous generation for converstational type of chats (Like chatGPT)

    Args
    - user_input(str): User input as a text
    - conversation_chain: Instance of ConversationalRetrievalChain 
    """

    with st.spinner('Spinning a snazzy reply...'):
        output = conversation_chat(user_input, conversation_chain, st.session_state['history'])

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

def conversation_chat(user_input, conversation_chain, history):
    """
    Returns LLM response after invoking model through conversation_chain

    Args:
    - user_input(str): User input
    - conversation_chain: Instance of ConversationalRetrievalChain
    - history: Previous response history
    returns:
    - result: Response generated from LLM

    """
   
    result = conversation_chain.predict(input=user_input)
    print("======================result====================", result)
    history.append((user_input, result))
    return result


def display_generated_responses(reply_container):
    """
    Display generated LLM response to Streamlit Web UI

    Args:
    - reply_container: Streamlit container created at previous step
    """
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def main():
    """
    First function to call when we start streamlit app
    """
    # Step 1: Initialize session state
    initialize_session_state()
    
    st.title("FinQA Chatbot (Mohamed Azharudeen - Scaler Portfolio Projects Feb'24)")
    
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>

            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
    # Create instance of Mistral 7B   
    llm = utils.create_llm()

    # then Create the chain object
    chain = create_conversational_chain(llm)

    # Display Chat to Web UI
    display_chat(chain)


if __name__ == "__main__":
    main()
