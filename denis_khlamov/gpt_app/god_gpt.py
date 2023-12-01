import streamlit as st
from GodChatGPT import GodChatGPT

@st.cache_resource
def load_god_chatgpt():
    god_chatgpt = GodChatGPT(st.secrets["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"],st.secrets["c29c0a5f135d0130111ba2e3490eb68ecf8ebff2a49ac4662409faf14ede3af7"])
    return god_chatgpt

god_chatgpt = load_god_chatgpt()

if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

def submit():
    st.session_state.user_input = st.session_state.query
    st.session_state.query = ''

st.title("🔥 Welcome to GOD-ChatGPT 🔥")

st.text_input("Play with me:", key='query', on_change=submit)

user_input = st.session_state.user_input

st.write("Your entred: ", user_input)

if user_input:
    result = god_chatgpt.agent_executor({"input":user_input})
    print(result)
    st.write("🔥 GOD-ChatGPT Answer: ", result["output"])