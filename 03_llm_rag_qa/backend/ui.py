import streamlit as st
from langchain_core.messages import HumanMessage
from graph_builder import main  # your LangGraph builder
import random
import tempfile
import os
from tools import rag

# Load the graph (this should load with tools and LLM set up)
graph = main()
thread_id = 42

st.set_page_config(page_title="ChatRAG Agent", layout="wide")
st.title("🧠 ChatRAG Assistant")


uploaded_file = st.sidebar.file_uploader("📄 Upload a document (.pdf or .txt)", type=["pdf", "txt"])

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Reset and re-index RAG pipeline
    rag.load_and_index(tmp_path)
    st.sidebar.success(f"Loaded: {uploaded_file.name}")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask me something...")

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    thread_id = random.random()

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

    result = graph.invoke({"messages": user_input}, config={"configurable": {"thread_id": thread_id}})
    final_response = result["messages"][-1]

    # Display assistant response
    st.chat_message("ai").markdown(final_response.content.split("</think>")[-1])

    # Update full message history
    st.session_state.messages = result["messages"]
