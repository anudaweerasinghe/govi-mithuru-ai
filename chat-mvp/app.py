import streamlit as st

SI_SYSTEM_PROMPT = "ඔබ ශ්‍රී ලංකාවේ වී වගාව පිළිබඳ විශේෂඥයෙක්. ගොවියාගේ ප්‍රශ්නයට පිළිතුරු දීමට සපයා ඇති තොරතුරු භාවිතා කරන්න. සිංහලෙන් පමණක් පිළිතුරු දෙන්න."

st.set_page_config(page_title="ගොවි-මිතු​රු AI", page_icon="👨🏾‍🌾", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("ගොවි-මිතු​රු AI 👨🏾‍🌾")

st.info("ශ්‍රී ලංකාවේ වී වගා කිරීම පිළිබඳව ඔබට ඇති ඕනෑම ප්‍රශ්නයක් අසන්න")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "system", "content": SI_SYSTEM_PROMPT}
    ]
