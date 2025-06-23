import streamlit as st
from langchain_core.messages import HumanMessage
from graph_builder import Graph  # your LangGraph builder
import random
import tempfile
import os
from tools import rag

@st.cache_resource
def get_graph():
    return Graph(), random.random()

agent, thread_id = get_graph()

st.set_page_config(page_title="ChatRAG Agent", layout="wide")
st.title("🧠 ChatRAG Assistant")


uploaded_file = st.sidebar.file_uploader("📄 Upload a document (.pdf or .txt)", type=["pdf", "txt"])

if uploaded_file:
    file_name = uploaded_file.name

    if (
        "cached_file_name" not in st.session_state
        or st.session_state.cached_file_name != file_name
    ):
        # Save to a reusable temp file path
        ext = os.path.splitext(file_name)[1]
        temp_path = os.path.join(tempfile.gettempdir(), f"rag_cached{ext}")

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.cached_file_name = file_name
        st.session_state.cached_file_path = temp_path

        # Load into RAG pipeline
        rag.load_and_index(temp_path)

        st.sidebar.success(f"📄 Loaded new document: {file_name}")
    else:
        st.sidebar.info(f"📄 Reusing cached: {file_name}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask me something...")

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.cache_resource.clear()
    st.rerun()

# Display chat so far
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        if msg.type == "ai":
            st.markdown(msg.content.split("</think>")[-1])
        else:
            st.markdown(msg.content)

# Handle new message
if user_input:
    # Show user message
    st.chat_message("human").markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    result = agent.graph.invoke({"messages": user_input}, config={"configurable": {"thread_id": thread_id}})
    final_response = result["messages"][-1]

    # Display assistant response
    st.chat_message("ai").markdown(final_response.content)
    # st.chat_message("ai").markdown(final_response.content.split("</think>")[-1])

    # Update full message history
    st.session_state.messages = result["messages"]
