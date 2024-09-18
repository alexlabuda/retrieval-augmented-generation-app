from shiny import App, ui, reactive
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from operator import itemgetter
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import ValidationError

# # # Load environment variables from .env file
# # load_dotenv()

# # Create FAISS index and retriever
# faiss_index = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Define LLM and prompts
system_prompt_template = (
    "You're a helpful AI assistant. Given a user question about county codes and some documents, "
    "answer the user question. If none of the documents can answer the question, just say you don't know.\n\n"
    "Here are the documents:{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("human", "{question}")
    ]
)

# Define helper functions and chain
def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string."""
    formatted = [
        f"Page Number: {doc.metadata.get('page_number', 'Unknown')}\nPage Content: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)

format = itemgetter("docs") | RunnableLambda(format_docs)

# app_ui = ui.page_sidebar(
#     ui.sidebar(
#         ui.h5("Chatbot Settings"),
#         ui.accordion(
#             ui.accordion_panel(
#                 "API Key",
#                 ui.div(
#                     ui.input_password("api_key", "Enter your OpenAI API Key", ""),  # Password field for API key
#                     style="font-size: 14px;"
#                 ),
#                 value="api_key",
#                 open=True  # Set to True to ensure it's open
#             ),
#             ui.accordion_panel(
#                 "Click to Upload Your PDF",
#                 ui.div(
#                     ui.input_file("upload_pdf", "Ask questions about your PDF", accept=".pdf", button_label="Upload PDF", placeholder="No file selected"),
#                     style="height: 10%; padding: 0px; margin-top: 5px; margin-bottom: -10px; font-size: 14px;"
#                 ),
#                 ui.div(
#                     ui.input_text("context_input", "What are your documents about?", "county codes"),
#                     style="font-size: 14px;"
#                 ),
#                 value="upload_context",
#                 open=True  # Set to True to ensure it's open
#             ),
#             ui.accordion_panel(
#                 "Document Retrieval Settings",
#                 ui.div(
#                     ui.input_slider("k", "Number of relevant documents to retrieve based on your question", min=1, max=5, value=2),
#                     style="margin-top: 0px; margin-bottom: 0px; font-size: 14px;"
#                 ),
#                 ui.div(
#                     ui.input_slider("score_threshold", "Adjust relevance threshold a document must meet to retrieve", min=0.1, max=0.8, value=0.4, step=0.1),
#                     style="margin-top: 0px; margin-bottom: 0px; font-size: 14px;"
#                 ),
#                 value="document_retrieval",
#                 open=False
#             ),
#             id="accordion",
#             multiple=True  # Ensure multiple panels can be open
#         ),
#         ui.p("Tips for better results:", style="font-size: 14px; font-weight: bold; margin-bottom: -10px;"),
#         ui.p("Lower the similarity score threshold to allow the model to retrieve a more diverse set of documents.", style="font-size: 13px; margin-bottom: -15px;"),
#         ui.p("Increase the number of relevant documents to allow the model to use more documents for context.", style="font-size: 13px;"),
#         width=340,
#     ),
#     ui.chat_ui("chat"),
#     ui.p(
#         "Powered by OpenAI's GPT-4o-mini | Created by ",
#         ui.a("Alex Labuda", href="https://www.linkedin.com/in/alex-labuda/", style="text-decoration: underline; color: inherit;"),
#         style="font-size: 13px; text-align: center;"
#     ),
#     title=ui.div(
#         ui.div("AI Chatbot For Your Documents", style="display: inline-block; font-size: 24px;"),
#         ui.div(ui.input_dark_mode(), style="display: inline-block; float: right;"),
#         style="width: 100%;"
#     ),
#     fillable_mobile=True,
# )


# # Create a welcome message
# welcome = ui.markdown(
#     """
#     I use `GPT-4o-Mini` to answer questions regarding the documents you upload. 
    
#     Enter you OpenAI API key to the left and ask me a question about your documents!
#     """
# )

# def server(input, output, session):
#     chat = ui.Chat(id="chat", messages=[welcome])
    
#     text_value = reactive.Value("")
#     docs_value = reactive.Value("")
    
#     # Define a callback to run when the user submits a message
#     @chat.on_user_submit
#     async def _():
#         try:
#             # Get the user's API key
#             api_key = input.api_key()
            
#             if not api_key:
#                 raise ValueError("Please provide your OpenAI API Key.")
            
#             # Initialize the LLM with the provided API key
#             llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            
#             # Get the user's input
#             question = chat.user_input()
            
#             # Update the retriever with dynamic k and score_threshold
#             retriever = faiss_index.as_retriever(
#                 search_type="similarity_score_threshold",
#                 search_kwargs={"k": input.k(), "score_threshold": input.score_threshold()},
#             )
            
#             # Set up the chain with the new retriever
#             answer = prompt | llm | StrOutputParser()
#             chain = (
#                 RunnableParallel(question=RunnablePassthrough(), docs=retriever)
#                 .assign(context=format)
#                 .assign(answer=answer)
#                 .pick(["answer", "docs"])
#             )
            
#             result = chain.invoke(question)
#             text_value.set(result['answer'])
            
#             # Format the documents
#             if result['docs']:
#                 formatted_docs = []
#                 for i, doc in enumerate(result['docs'][:input.k()]):
#                     formatted_doc = f"Document {i+1}:\nPage Number: {doc.metadata.get('page', 'Unknown')}\n\nPage Content:\n{doc.page_content}"
#                     formatted_docs.append(formatted_doc)
#                 docs_value.set("\n\n".join(formatted_docs))
#             else:
#                 docs_value.set("No relevant documents found.")
            
#             # Append a response to the chat
#             await chat.append_message_stream(
#                 f"{text_value.get()}<br><br><strong><span style='color: inherit;'>---------- RELEVANT DOCUMENTS FROM YOUR SOURCE ----------</span></strong><br><br>{docs_value.get()}"
#             )
#         except Exception as e:
#             await chat.append_message_stream(f"An error occurred: {str(e)}")

# app = App(app_ui, server)


#### LAST WORKABLE CODE #### --------------------------------------------------------------------------------------------
from shiny import App, ui, reactive
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS, DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file if needed
# load_dotenv()

# app_ui = ui.page_sidebar(
#     ui.sidebar(
#         ui.h5("Chatbot Settings"),
#         ui.accordion(
#             ui.accordion_panel(
#                 "Click to Upload Your PDF",
#                 ui.div(
#                     ui.input_select(
#                         "select_file_type",
#                         "Select the file type of your document",
#                         {"pdf": "PDF", "csv": "CSV"},
#                     )
#                 ),
#                 ui.panel_conditional(
#                     "input.select_file_type === 'pdf'",
#                     ui.div(
#                         ui.input_file("upload_pdf", "Ask questions about your PDF", accept=".pdf", button_label="Upload PDF", placeholder="No file selected"),
#                         style="height: 10%; padding: 0px; margin-top: 5px; margin-bottom: -10px; font-size: 14px;"
#                     ),
#                 ),
#                 ui.panel_conditional(
#                     "input.select_file_type === 'csv'",
#                     ui.div(
#                         ui.input_file("upload_csv", "Ask questions about your CSV", accept=".csv", button_label="Upload CSV", placeholder="No file selected"),
#                         style="height: 10%; padding: 0px; margin-top: 5px; margin-bottom: -10px; font-size: 14px;"
#                     ),
#                 ),
#                 value="upload_context",
#                 open=True
#             ),
#             ui.accordion_panel(
#                 "API Key",
#                 ui.div(
#                     ui.input_password("api_key", "Enter your OpenAI API Key", ""),  # Password field for API key
#                     style="font-size: 14px;"
#                 ),
#                 value="api_key",
#                 open=True
#             ),
#             ui.accordion_panel(
#                 "Document Retrieval Settings",
#                 ui.div(
#                     ui.input_slider("k", "Number of relevant documents to retrieve based on your question", min=1, max=5, value=2),
#                     style="margin-top: 0px; margin-bottom: 0px; font-size: 14px;"
#                 ),
#                 ui.div(
#                     ui.input_slider("score_threshold", "Adjust relevance threshold a document must meet to retrieve", min=0.1, max=0.8, value=0.4, step=0.1),
#                     style="margin-top: 0px; margin-bottom: 0px; font-size: 14px;"
#                 ),
#                 value="document_retrieval",
#                 open=False
#             ),
#             id="accordion",
#             multiple=True
#         ),
#         ui.p("Tips for better results:", style="font-size: 14px; font-weight: bold; margin-bottom: -10px;"),
#         ui.p("Lower the similarity score threshold to allow the model to retrieve a more diverse set of documents.", style="font-size: 13px; margin-bottom: -15px;"),
#         ui.p("Increase the number of relevant documents to allow the model to use more documents for context.", style="font-size: 13px;"),
#         width=340,
#     ),
#     ui.chat_ui("chat"),
#     ui.p(
#         "Powered by OpenAI's GPT-4o-mini | Created by ",
#         ui.a("Alex Labuda", href="https://www.linkedin.com/in/alex-labuda/", style="text-decoration: underline; color: inherit;"),
#         style="font-size: 14px; text-align: center;"
#     ),
#     title=ui.div(
#         ui.div("AI Chatbot For Your Documents", style="display: inline-block; font-size: 24px;"),
#         ui.div(ui.input_dark_mode(), style="display: inline-block; float: right;"),
#         style="width: 100%;"
#     ),
#     fillable_mobile=True,
# )

def save_file(datapath):
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    file_path = f'./{folder}/{os.path.basename(datapath)}'
    with open(file_path, 'wb') as f:
        with open(datapath, 'rb') as uploaded_file:
            f.write(uploaded_file.read())
    return file_path

import asyncio
from shiny import App, ui, reactive

async def handle_user_query(chat, qa_chain, user_query):
    result = qa_chain.invoke({"question": user_query})
    await chat.append_message_stream(
        f"{result['answer']}<br><br><strong>---------- RELEVANT DOCUMENTS ----------</strong><br><br>{format_docs(result['source_documents'])}"
    )

# def server(input, output, session):
#     chat = ui.Chat(id="chat", messages=[])
    
#     # State variables to track if a document and API key have been provided
#     doc_uploaded = reactive.Value(False)
#     api_key_provided = reactive.Value(False)
#     file_path = reactive.Value(None)  # Reactive value to hold the file path
#     last_processed_query = reactive.Value(None)  # Track the last processed query

#     @reactive.Effect
#     def handle_document_upload():
#         uploaded_file = input.upload_pdf()

#         if uploaded_file:
#             file_path.set(save_file(uploaded_file[0]['datapath']))
#             doc_uploaded.set(True)

#             # Notify the user that the document has been uploaded
#             asyncio.create_task(chat.append_message_stream(
#                 "Document uploaded successfully. You can now ask a question."
#             ))

#             # Reset the last processed query to ensure the next query is fresh
#             last_processed_query.set(None)

#     @reactive.Effect
#     def handle_api_key_input():
#         api_key = input.api_key()

#         # Check if the API key is provided and set the state
#         if api_key and not api_key_provided.get():
#             api_key_provided.set(True)
#             asyncio.create_task(chat.append_message_stream(
#                 "API Key provided. You can now ask a question."
#             ))

#             # Reset the last processed query to ensure the next query is fresh
#             last_processed_query.set(None)

#     async def process_user_query(chat, qa_chain, user_query):
#         # Check if the query is meaningful and not just a greeting
#         if any(word in user_query.lower() for word in ["hi", "hello", "hey"]):
#             await chat.append_message_stream("Please ask a specific question related to your document.")
#             return

#         # Process the query with the document and API key
#         result = qa_chain.invoke({"question": user_query})
#         await chat.append_message_stream(
#             f"{result['answer']}<br><br><strong>---------- RELEVANT DOCUMENTS ----------</strong><br><br>{format_docs(result['source_documents'])}"
#         )

#     @reactive.Effect
#     def handle_user_query_input():
#         user_query = chat.user_input()
#         if user_query and user_query.strip():  # Ensure the input is not empty or just whitespace
#             if not doc_uploaded.get():
#                 asyncio.create_task(chat.append_message_stream(
#                     "Please upload a document before asking a question."
#                 ))
#                 last_processed_query.set(user_query)  # Mark the query so it isn't reprocessed
#             elif not api_key_provided.get():
#                 asyncio.create_task(chat.append_message_stream(
#                     "Please provide an API key before asking a question."
#                 ))
#                 last_processed_query.set(user_query)  # Mark the query so it isn't reprocessed
#             elif user_query != last_processed_query.get():  # Avoid processing the same query again
#                 # Only process meaningful queries
#                 if any(word in user_query.lower() for word in ["hi", "hello", "hey"]):
#                     asyncio.create_task(chat.append_message_stream(
#                         "Please ask a specific question related to your document."
#                     ))
#                 else:
#                     # Proceed with processing the query now that both are provided
#                     loader = PyPDFLoader(file_path.get())  # Use the reactive value
#                     docs = loader.load()

#                     text_splitter = RecursiveCharacterTextSplitter(
#                         chunk_size=1000,
#                         chunk_overlap=200
#                     )
#                     splits = text_splitter.split_documents(docs)

#                     vectordb = DocArrayInMemorySearch.from_documents(splits, OpenAIEmbeddings())

#                     retriever = vectordb.as_retriever(
#                         search_type='mmr',
#                         search_kwargs={'k': input.k(), 'fetch_k': 4}
#                     )

#                     memory = ConversationBufferMemory(
#                         memory_key='chat_history',
#                         output_key='answer',
#                         return_messages=True
#                     )

#                     qa_chain = ConversationalRetrievalChain.from_llm(
#                         llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=input.api_key()),
#                         retriever=retriever,
#                         memory=memory,
#                         return_source_documents=True,
#                         verbose=False
#                     )

#                     # Use create_task to run the coroutine in the existing event loop
#                     asyncio.create_task(process_user_query(chat, qa_chain, user_query))

#                     # Mark the query as processed
#                     last_processed_query.set(user_query)

# app = App(app_ui, server)

# def server(input, output, session):
#     chat = ui.Chat(id="chat", messages=[])
    
#     # State variables to track if a document and API key have been provided
#     doc_uploaded = reactive.Value(False)
#     api_key_provided = reactive.Value(False)
#     file_path = reactive.Value(None)  # Reactive value to hold the file path
#     last_processed_query = reactive.Value(None)  # Track the last processed query

#     @reactive.Effect
#     def handle_document_upload():
#         uploaded_file = input.upload_pdf()

#         if uploaded_file:
#             file_path.set(save_file(uploaded_file[0]['datapath']))
#             doc_uploaded.set(True)

#             # Notify the user that the document has been uploaded
#             asyncio.create_task(chat.append_message_stream(
#                 "Document uploaded successfully. You can now ask a question."
#             ))

#     @reactive.Effect
#     def handle_api_key_input():
#         api_key = input.api_key()

#         # Check if the API key is provided and set the state
#         if api_key and not api_key_provided.get():
#             api_key_provided.set(True)
#             asyncio.create_task(chat.append_message_stream(
#                 "API Key provided. You can now ask a question."
#             ))

#     async def process_user_query(chat, qa_chain, user_query):
#         result = qa_chain.invoke({"question": user_query})
#         await chat.append_message_stream(
#             f"{result['answer']}<br><br><strong>---------- RELEVANT DOCUMENTS ----------</strong><br><br>{format_docs(result['source_documents'])}"
#         )

#     @reactive.Effect
#     def handle_user_query_input():
#         user_query = chat.user_input()
#         if user_query and user_query.strip():  # Ensure the input is not empty or just whitespace
#             if user_query != last_processed_query.get():  # Avoid processing the same query again
#                 if not doc_uploaded.get():
#                     asyncio.create_task(chat.append_message_stream(
#                         "Please upload a document before asking a question."
#                     ))
#                 elif not api_key_provided.get():
#                     asyncio.create_task(chat.append_message_stream(
#                         "Please provide an API key before asking a question."
#                     ))
#                 else:
#                     # Proceed with processing the query now that both are provided
#                     loader = PyPDFLoader(file_path.get())  # Use the reactive value
#                     docs = loader.load()

#                     text_splitter = RecursiveCharacterTextSplitter(
#                         chunk_size=1000,
#                         chunk_overlap=200
#                     )
#                     splits = text_splitter.split_documents(docs)

#                     vectordb = DocArrayInMemorySearch.from_documents(splits, OpenAIEmbeddings())

#                     retriever = vectordb.as_retriever(
#                         search_type='mmr',
#                         search_kwargs={'k': input.k(), 'fetch_k': 4}
#                     )

#                     memory = ConversationBufferMemory(
#                         memory_key='chat_history',
#                         output_key='answer',
#                         return_messages=True
#                     )

#                     qa_chain = ConversationalRetrievalChain.from_llm(
#                         llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=input.api_key()),
#                         retriever=retriever,
#                         memory=memory,
#                         return_source_documents=True,
#                         verbose=False
#                     )

#                     # Use create_task to run the coroutine in the existing event loop
#                     asyncio.create_task(process_user_query(chat, qa_chain, user_query))

#                     # Mark the query as processed
#                     last_processed_query.set(user_query)

# app = App(app_ui, server)
#### LAST WORKABLE CODE #### --------------------------------------------------------------------------------------------

from shiny import App, ui, reactive
import os
from sqlalchemy import create_engine
import pandas as pd
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent

# Define helper functions for file handling
def save_file(datapath):
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = f'./{folder}/{os.path.basename(datapath)}'
    with open(file_path, 'wb') as f:
        with open(datapath, 'rb') as uploaded_file:
            f.write(uploaded_file.read())
    return file_path

# Process PDF Queries
async def process_pdf_query(chat, file_path, user_query, api_key, pdf_memory, input):
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create a retriever for the PDF content
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': input.retriever_k(),
            'similarity_score_threshold': input.similarity_threshold()
        }
    )

    # Initialize the QA chain with memory integration for PDF
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key),
        retriever=retriever,
        memory=pdf_memory,  # Use PDF-specific memory
        return_source_documents=True
    )

    # Process the user query with the PDF data and memory
    result = qa_chain({"question": user_query})

    # Display the result
    await chat.append_message_stream(f"Answer: {result['answer']}")
    return result

# Process CSV Queries
async def process_csv_query(chat, db, llm, user_query, csv_memory):
    # Create an SQL agent for CSV queries
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=None,
        db=db,
        verbose=True,
        agent_type="zero-shot-react-description",
    )

    # Extract the chat history from memory (if any)
    chat_history = csv_memory.load_memory_variables({}).get('history', [])

    try:
        # Process the user query with the CSV data and memory
        result = agent_executor.run(user_query)

        # Save the user query and the model's response to CSV-specific memory
        csv_memory.save_context({"question": user_query}, {"answer": result})

        # Display the result
        await chat.append_message_stream(f"Query Result: {result}")
    except Exception as e:
        await chat.append_message_stream(f"Error processing your question: {str(e)}")

# Initialize UI for CSV and PDF App
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("Document Chatbot"),
        ui.accordion(
            ui.accordion_panel(
                "Click to Upload Your Document",
                ui.div(
                    ui.input_select(
                        "select_file_type",
                        "Select the file type",
                        {"pdf": "PDF", "csv": "CSV"}
                    )
                ),
                ui.panel_conditional(
                    "input.select_file_type === 'pdf'",
                    ui.div(
                        ui.input_file(
                            "upload_pdf",
                            "Upload PDF",
                            accept=[".pdf"],
                            button_label="Upload PDF",
                            placeholder="No file selected"
                        ),
                        style="margin-top: 5px; font-size: 14px;"
                    )
                ),
                ui.panel_conditional(
                    "input.select_file_type === 'csv'",
                    ui.div(
                        ui.input_file(
                            "upload_csv",
                            "Upload CSV for SQL queries",
                            accept=[".csv"],
                            button_label="Upload CSV",
                            placeholder="No file selected"
                        ),
                        style="margin-top: 5px; font-size: 14px;"
                    )
                ),
                value="upload_context",
                open=True
            ),
            ui.accordion_panel(
                "Retriever Settings",
                ui.input_slider(
                    "retriever_k",
                    "Number of Documents (k)",
                    min=1,
                    max=10,
                    value=5,
                    step=1
                ),
                ui.input_slider(
                    "similarity_threshold",
                    "Similarity Score Threshold",
                    min=0.0,
                    max=1.0,
                    value=0.5,
                    step=0.05
                ),
                value="retriever_settings",
                open=False
            ),
            ui.accordion_panel(
                "API Key",
                ui.div(
                    ui.input_password(
                        "api_key",
                        "Enter your OpenAI API Key",
                        ""
                    ),
                    style="font-size: 14px;"
                ),
                value="api_key",
                open=True
            ),
            id="accordion",
            multiple=True
        ),
        ui.p("Tips for better results:", style="font-size: 14px; font-weight: bold; margin-bottom: -10px;"),
        ui.p("Lower the similarity score threshold to allow the model to retrieve a more diverse set of documents.", style="font-size: 13px; margin-bottom: -15px;"),
        ui.p("Increase the number of relevant documents to allow the model to use more documents for context.", style="font-size: 13px;"),
        width=340,
    ),
    ui.chat_ui("chat"),
    ui.p(
        "Powered by OpenAI's GPT-4o-mini | Created by ",
        ui.a("Alex Labuda", href="https://www.linkedin.com/in/alex-labuda/", style="text-decoration: underline; color: inherit;"),
        style="font-size: 13px; text-align: center;"
    ),
    title=ui.div(
        ui.div("AI Chatbot For Your Documents", style="display: inline-block; font-size: 24px;"),
        ui.div(ui.input_dark_mode(), style="display: inline-block; float: right;"),
        style="width: 100%;"
    ),
    fillable_mobile=True,
)

def server(input, output, session):
    chat = ui.Chat("chat")
    doc_uploaded = reactive.Value(False)
    csv_uploaded = reactive.Value(False)
    api_key_provided = reactive.Value(False)
    file_path = reactive.Value(None)

    # Initialize separate memory for PDF and CSV conversations
    pdf_memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'  # Specify the output key to store in memory
    )
    csv_memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'  # Specify the output key to store in memory
    )

    # Reactive values for engine and db
    engine = reactive.Value(None)
    db = reactive.Value(None)

    @reactive.Effect
    async def handle_document_upload():
        if input.select_file_type() == 'pdf' and input.upload_pdf():
            # Handle PDF upload
            uploaded_file = input.upload_pdf()
            file_path.set(save_file(uploaded_file[0]['datapath']))
            doc_uploaded.set(True)
            csv_uploaded.set(False)  # Reset CSV upload state
            await chat.append_message_stream("PDF uploaded successfully. You can now ask questions about your PDF.")
        elif input.select_file_type() == 'csv' and input.upload_csv():
            # Handle CSV upload
            uploaded_file = input.upload_csv()
            csv_file_path = save_file(uploaded_file[0]['datapath'])
            # Load CSV into SQLite database
            df = pd.read_csv(csv_file_path)
            engine.set(create_engine("sqlite:///uploaded_data.db"))
            df.to_sql('uploaded_data', engine.get(), index=False, if_exists='replace')
            csv_uploaded.set(True)
            doc_uploaded.set(False)  # Reset PDF upload state
            db.set(SQLDatabase(engine=engine.get()))
            await chat.append_message_stream(
                "CSV uploaded successfully and loaded into the table 'uploaded_data'. "
                "You can now ask questions about your CSV data in natural language."
            )

    @reactive.Effect
    def handle_api_key_input():
        api_key = input.api_key()
        if api_key and not api_key_provided.get():
            api_key_provided.set(True)
            asyncio.create_task(chat.append_message_stream("API Key provided. You can now ask a question."))

    @reactive.Effect
    async def handle_user_query_input():
        user_query = chat.user_input()
        if user_query and user_query.strip():
            if not api_key_provided.get():
                await chat.append_message_stream("Please enter your OpenAI API Key before asking a question.")
                return
            if doc_uploaded.get():
                # Process PDF-based questions
                await process_pdf_query(
                    chat,
                    file_path.get(),
                    user_query,
                    input.api_key(),
                    pdf_memory,
                    input
                )
            elif csv_uploaded.get():
                # Process CSV-based questions
                llm = ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=input.api_key()
                )
                await process_csv_query(
                    chat,
                    db.get(),
                    llm,
                    user_query,
                    csv_memory
                )
            else:
                await chat.append_message_stream("Please upload a document before asking a question.")

app = App(app_ui, server)

