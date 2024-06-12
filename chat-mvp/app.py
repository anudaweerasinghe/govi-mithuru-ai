import streamlit as st
from pinecone import Pinecone

import cohere
import anthropic

from siconv import singlish_to_sinhala


SI_SYSTEM_PROMPT = "ඔබ ශ්‍රී ලංකාවේ වී වගාව පිළිබඳ විශේෂඥයෙක්. ගොවියාගේ ප්‍රශ්නයට පිළිතුරු දීමට සපයා ඇති තොරතුරු භාවිතා කරන්න. කෙටි හා සරල පිළිතුරු දෙන්න - අවශ්‍ය නම් පාරිභෝගිකයාගෙන් තවත් ප්‍රශ්න අසන්න. සිංහලෙන් පමණක් පිළිතුරු දෙන්න."

st.set_page_config(page_title="ගොවි-මිතු​රු AI", page_icon="👨🏾‍🌾", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("ගොවි-මිතු​රු AI 👨🏾‍🌾")

st.info("ශ්‍රී ලංකාවේ වී වගා කිරීම පිළිබඳව ඔබට ඇති ඕනෑම ප්‍රශ්නයක් අසන්න")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
    ]


@st.cache_resource(show_spinner=False)
def init():
    with st.spinner(text="Loading..."):
        pc = Pinecone(api_key=st.secrets["pinecone_key"])
        pc_index = pc.Index("govi-mithuru-ai")

        co = cohere.Client(st.secrets["cohere_key"])

        llm = anthropic.Anthropic(api_key=st.secrets["anthropic_key"])

        return pc_index,co,llm
    
pc_index, co, llm = init()

def extract_from_stream(steam):
    for event in stream:
        yield event.delta.text

if user_input_en := st.chat_input("ඔබේ ප්‍රශ්නය, English අකුරු වලින්න්"):
  user_input = singlish_to_sinhala(user_input_en)
  st.session_state.messages.append({"role": "user", "content": user_input})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if len(st.session_state.messages)>0 and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
            
            query_emb = co.embed(texts=[user_input], input_type="search_query", model="embed-multilingual-v3.0").embeddings 
            pc_results = pc_index.query(vector=query_emb, top_k=1, include_metadata=True, filter={"language": {"$eq": "si"}})

            context = "\n\n".join([result["metadata"]["content"] for result in pc_results["matches"]])

            messages_for_anthropic = st.session_state.messages[:-1]


            message_with_context = f"ප්රශ්නය: {user_input}\n\nඅදාල තොරතුරු: '{context}'"
            messages_for_anthropic.append({"role": "user", "content": message_with_context})

            with llm.messages.stream(
                model="claude-3-opus-20240229",
                messages=messages_for_anthropic,
                system=SI_SYSTEM_PROMPT,
                max_tokens=4096,
            ) as stream:
                text_response = st.write_stream(stream.text_stream)

            message = {"role": "assistant", "content": text_response}
            st.session_state.messages.append(message) # Add response to message history

