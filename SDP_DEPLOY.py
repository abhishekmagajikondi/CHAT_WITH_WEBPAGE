import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from langchain_core.output_parsers import StrOutputParser
import time
from langchain_community.vectorstores import FAISS


# Set environment variables for LangChain and Hugging Face API keys
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_hAfWmGdtFaYncVGkfakEMWrWysYquTYZGq"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"

# Define model repository IDs for each model option
model_repo_ids = {
    "mixtral-8x22b": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemini": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-3-8B": "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft-phi": "mistralai/Mistral-7B-Instruct-v0.2",
   
}

def get_vectorstore_from_url(url):
    # Load and split the document from the URL
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # Create vector store from document chunks
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    return vector_store

def get_context_retriever_chain(vector_store, model):
    # Dynamically set the LLM using the chosen model
    repo_id = model_repo_ids[model]
    llm = HuggingFaceEndpoint(repo_id=repo_id,temperature=0.7)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get the information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain                         

def get_conversational_rag_chain(retriever_chain, model):
    # Dynamically set the LLM using the chosen model
    repo_id = model_repo_ids[model]
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7)
    
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "Answer every user's questions in few lines based on the below context, Your task is to answer precisely for each and every  given Query:\n\n{context}"),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("user", "{input}"),
    # ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer each question concisely based on the provided context from the webpage. Aim for precise, relevant answers, and keep responses to one line when possible:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

        
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)                  

def get_response_1(user_query, model):
    # Create conversation chain with the chosen model
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store, model)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain, model)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']

def generate_display_questions(question):
    
    # Template Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    
    # LLM
    # Load a LLM
     # Dynamically set the LLM using the chosen model
    repo_id = model_repo_ids[model]
    llm = HuggingFaceEndpoint(repo_id=repo_id,temperature=0.7)

    # Chain
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    # Run
    questions = generate_queries_decomposition.invoke({"question":question})

    # Display the questions
    for i, q in enumerate(questions[1:], start=1):
        print(f"Sub-question {i}: {q}")

    return generate_queries_decomposition

def retrieve_and_rag(vector_store,question,sub_question_generator_chain):
    """RAG on each sub-question"""
    from langchain import hub
    # RAG prompt
    prompt_rag = hub.pull("rlm/rag-prompt")

    # Use our decomposition / 
    sub_questions = sub_question_generator_chain.invoke({"question":question})
    
    # Initialize a list to hold RAG chain results
    rag_results = []
    
    # Load a LLM
     # Dynamically set the LLM using the chosen model
    repo_id = model_repo_ids[model]
    llm = HuggingFaceEndpoint(repo_id=repo_id,temperature=0.7)
    
    retriever = vector_store.as_retriever()
    
    for sub_question in sub_questions:
        
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)
        
        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                "question": sub_question})
        rag_results.append(answer)
    
    return rag_results,sub_questions

def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

def get_final_rag_chain(user_query , questions , answers , model):
    context = format_qa_pairs(questions, answers)
    
    # Load a LLM
    # Dynamically set the LLM using the chosen model
    repo_id = model_repo_ids[model]
    llm = HuggingFaceEndpoint(repo_id=repo_id,temperature=0.8)
    
    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = final_rag_chain.invoke({"context":context,"question":user_query})
    
    return answer

def get_response(vector_store , user_query, model):
    
    generate_queries_decomposition = generate_display_questions(user_query)
    
    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions = retrieve_and_rag(vector_store , user_query, generate_queries_decomposition)
    
    answer = get_final_rag_chain(user_query , questions, answers , model)
    
    return answer[8:]


# Typing effect function
def display_typing_effect(content, delay=0.001):
    placeholder = st.empty()
    displayed_text = ""
    for char in content:
        displayed_text += char
        placeholder.text(displayed_text)
        time.sleep(delay)
        

gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
</style>
<div class="gradient-text">Chat with webpage</div>
"""

# App Configuration
st.markdown(gradient_text_html, unsafe_allow_html=True)
st.caption("Talk your way through data")

# Define model options for user selection
model_options = {
    
    "mixtral-8x22b": "Mistral",
    "gemini": "Gemini",
    "llama-3-8B": "Llama",
    "microsoft-phi": "Microsoft Phi",
   
}
# User selects the model from a radio button
model = st.radio(
    "Choose your AI Model:",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=0,
    horizontal=True,
)

if model is "mixtral-8x22b":
    avatar = r"mistral_icon.png"

elif model is "gemini":
    avatar = r"google-gemini-icon.png"
    
elif model is "llama-3-8B" :
    avatar = r"Llama_Icon-removebg-preview.png"
    
elif model is "microsoft-phi":
    avatar = r"microsoft.png"

# Sidebar for website URL input
with st.sidebar:
    st.header("SDP Project")
    website_url = st.text_input("Enter webpage URL:")
    submit_button = st.button("Submit") 
    st.markdown("------")
    st.subheader("Team Details")
    st.write("Abhishek Magajikondi")
    st.write("Sampath G Meti")
    st.write("Sachin Prakash Aigali")
    st.write("Sam Nathan Swarna")

# Load vector store if URL is provided
if submit_button and website_url is None or website_url == "":
    st.info("Please enter a webpage URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I help you today?"),
        ]
    
    if "vector_store" not in st.session_state:  
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        with st.sidebar:
            st.success("Vector store created successfully")

    # User input and response generation
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(st.session_state.vector_store , user_query, model)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    # Loop through chat history and display each message with the appropriate avatar
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar=avatar):
                # Apply typing effect only for the most recent AI message
                if i == len(st.session_state.chat_history) - 1:
                    display_typing_effect(message.content)  # Ensure this function is defined to handle typing effect
                else:
                    st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar=r"human.png"):
                st.write(message.content)
