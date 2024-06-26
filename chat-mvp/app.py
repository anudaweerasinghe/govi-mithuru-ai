import streamlit as st
from pinecone import Pinecone

import cohere
import anthropic

from translit import translit

from prompts_and_strings import get_system_prompt, get_title, get_info, format_message_with_context, get_input_placeholder

MODEL = "claude-3-5-sonnet-20240620"

languages = {"සිංහල": "si", "தமிழ்": "ta", "English": "en"}
st.set_page_config(page_title="ගොවි-මිතු​රු AI", page_icon="👨🏾‍🌾", layout="centered", initial_sidebar_state="auto", menu_items=None)


if "lang" not in st.query_params:
    st.query_params["lang"] = "si"

def set_language():
    if "selected_language" in st.session_state:
        st.session_state.messages = []
        st.query_params["lang"] = languages[st.session_state.selected_language]
        

sel_lang = st.radio(
    "Language",
    options=languages,
    horizontal=True,
    on_change=set_language,
    key="selected_language",
)

st.title(get_title(st.query_params["lang"]))

st.info(get_info(st.query_params["lang"]))

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

if user_input_en := st.chat_input(get_input_placeholder(st.query_params["lang"])):
  user_input =  translit(st.query_params["lang"], user_input_en)
  st.session_state.messages.append({"role": "user", "content": user_input})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if len(st.session_state.messages)>0 and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
            
            query_emb = co.embed(texts=[user_input], input_type="search_query", model="embed-multilingual-v3.0").embeddings 
            pc_results = pc_index.query(vector=query_emb, top_k=3, include_metadata=True, filter={"language": {"$eq": st.query_params["lang"]}})

            context = "\n\n".join([result["metadata"]["content"] for result in pc_results["matches"]])

            messages_for_anthropic = st.session_state.messages[:-1]


            message_with_context = format_message_with_context(st.query_params["lang"], user_input, context)
            messages_for_anthropic.append({"role": "user", "content": message_with_context})

            with llm.messages.stream(
                model=MODEL,
                messages=messages_for_anthropic,
                system=get_system_prompt(st.query_params["lang"]),
                max_tokens=4096,
            ) as stream:
                text_response = st.write_stream(stream.text_stream)

            message = {"role": "assistant", "content": text_response}
            st.session_state.messages.append(message) # Add response to message history

