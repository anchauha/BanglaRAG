import os
import getpass

# Required libraries:
# pip install langchain chromadb
# pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph
# pip install -qU "langchain[openai]"
# pip install -qU langchain-openai
# pip install -qU langchain-chroma
# pip install -qU langchain_community pypdf
# pip install --upgrade --quiet langgraph langchain-community 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

from langgraph.graph import MessagesState, StateGraph
from langgraph.graph import END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Global variables to hold initialized components
llm = None
embeddings = None
vector_store = None
graph = None

def initialize_rag_pipeline():
    """Initialize all RAG pipeline components once at startup"""
    global llm, embeddings, vector_store, graph
    
    # -------------------------
    # 1) API Key Configuration
    # -------------------------
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    os.environ["LANGSMITH_TRACING"] = "true"
    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter Langsmith API Key: ")

    # ------------------------------
    # 2) Initialize LLM and Utility
    # ------------------------------
    llm = init_chat_model("gpt-4o", model_provider="openai")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # -------------------------
    # 3) Vector Store
    # -------------------------
    file_path = "data/Structured_RAG.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)

    vector_store = Chroma(embedding_function=embeddings)
    _ = vector_store.add_documents(documents=all_splits)

    # -------------------------
    # 4) Build the Pipeline
    # -------------------------
    # Define tool within the function scope
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """
        Retrieve information related to a user's query from the vector store.
        Returns serialized doc content (for internal use) and doc objects themselves.
        """
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(state: MessagesState):
        """
        Forces a retrieve tool call every time before generating a final response.
        """
        system_message = SystemMessage(
            "You MUST use the retrieve tool before responding. Never skip retrieval."
        )
        llm_with_tools = llm.bind_tools([retrieve])
        messages = [system_message] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tools = ToolNode([retrieve])

    def generate(state: MessagesState):
        """
        Uses any retrieved tool messages to form final answer (type=ai).
        Only the final 'ai' message is to be shown to the user.
        """
        # Find the last 1-2 retrieval messages if needed
        tool_messages = [msg for msg in state["messages"] if msg.type == "tool"]
        docs_content = "\n\n".join(msg.content for msg in tool_messages[-2:])

        system_prompt = (
            """
            As a moderator familiar with religious and ethnic relationships in Bangladesh, assess whether this text represents the country's broader societal values. Consider if it could be perceived as hate speech, hurtful, or culturally insensitive to members of any community. Provide feedback by highlighting potential issues and suggesting ways to foster respectful and inclusive dialogue. Answer briefly and translate that in the Bengali language before responding."""
            f"{docs_content}\n\nUser query: {state['messages'][-1].content}"
        )

        prompt = [SystemMessage(system_prompt)] + state["messages"]
        response = llm.invoke(prompt)
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", generate)

    # Force the sequence: user query -> retrieval -> final response
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_edge("query_or_respond", "tools")
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    # Enable memory for conversation tracking
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    print("RAG pipeline initialized successfully")

def run_rag_pipeline(user_message: str, thread_id: str):
    """
    Runs the RAG pipeline given a user message and a thread session ID.
    Returns a dict with keys:
      'ai_response' -> final LLM text
      'retrieval_used' -> boolean indicating if retrieval happened
    """
    global graph
    
    # If pipeline isn't initialized yet, do it now
    if graph is None:
        initialize_rag_pipeline()
    
    # Prepare a config that sets the thread id for storing conversation
    config = {"thread_id": thread_id}

    # Streaming mode returns each step in a generator. Collect them all:
    ai_response_text = ""
    retrieval_used = False

    for step in graph.stream(
        {"messages": [{"role": "user", "content": user_message}]}, 
        stream_mode="values",
        config=config
    ):
        last_msg = step["messages"][-1]
        if last_msg.type == "tool":
            retrieval_used = True
        elif last_msg.type == "ai":
            ai_response_text = last_msg.content

    return {"ai_response": ai_response_text, "retrieval_used": retrieval_used}

# Initialize the pipeline when the module is imported
initialize_rag_pipeline()
