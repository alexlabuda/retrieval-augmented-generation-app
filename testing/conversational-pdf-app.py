from shiny import App, ui, reactive
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import asyncio

# Define helper functions
def save_file(datapath):
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    file_path = f'./{folder}/{os.path.basename(datapath)}'
    with open(file_path, 'wb') as f:
        with open(datapath, 'rb') as uploaded_file:
            f.write(uploaded_file.read())
    return file_path

async def process_pdf_query(chat, file_path, user_query, api_key, memory):
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create a retriever for the PDF content
    vectordb = DocArrayInMemorySearch.from_documents(splits, OpenAIEmbeddings())
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 5})

    # Initialize the QA chain with memory (No custom prompt needed)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key),
        retriever=retriever,
        memory=memory,  # Memory stores chat history
        return_source_documents=True
    )

    # Process the user query with the PDF data and memory
    result = qa_chain.invoke({"question": user_query})

    # Save the user query and the model's response to memory
    memory.save_context({"question": user_query}, {"answer": result['answer']})

    # Display the result
    await chat.append_message_stream(f"Answer: {result['answer']}")
    return result

# Define UI for PDF App
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("Chatbot Settings"),
        ui.accordion(
            ui.accordion_panel(
                "Click to Upload Your PDF",
                ui.div(
                    ui.input_file("upload_pdf", "Ask questions about your PDF", accept=".pdf", button_label="Upload PDF", placeholder="No file selected"),
                    style="height: 10%; padding: 0px; margin-top: 5px; margin-bottom: -10px; font-size: 14px;"
                ),
                value="upload_context",
                open=True
            ),
            ui.accordion_panel(
                "API Key",
                ui.div(
                    ui.input_password("api_key", "Enter your OpenAI API Key", ""),
                    style="font-size: 14px;"
                ),
                value="api_key",
                open=True
            ),
            id="accordion",
            multiple=True
        ),
        width=340,
    ),
    ui.chat_ui("chat"),
    title=ui.div(
        ui.div("AI Chatbot For Your PDFs", style="display: inline-block; font-size: 24px;"),
        ui.div(ui.input_dark_mode(), style="display: inline-block; float: right;"),
        style="width: 100%;"
    ),
    fillable_mobile=True,
)

def server(input, output, session):
    chat = ui.Chat(id="chat", messages=[])
    doc_uploaded = reactive.Value(False)
    api_key_provided = reactive.Value(False)
    file_path = reactive.Value(None)
    
    # Initialize memory for conversation history
    memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True)
    
    @reactive.Effect
    async def handle_document_upload():
        # Reset states to ensure proper handling
        doc_uploaded.set(False)
        file_path.set(None)
        
        if input.upload_pdf():
            # Handle PDF Upload
            uploaded_file = input.upload_pdf()
            file_path.set(save_file(uploaded_file[0]['datapath']))
            doc_uploaded.set(True)
            await chat.append_message_stream("PDF uploaded successfully. You can now ask questions about your PDF.")

    @reactive.Effect
    def handle_api_key_input():
        api_key = input.api_key()
        if api_key and not api_key_provided.get():
            api_key_provided.set(True)
            asyncio.create_task(chat.append_message_stream("API Key provided. You can now ask a question."))

    @reactive.Effect
    def handle_user_query_input():
        user_query = chat.user_input()
        if user_query and user_query.strip():
            if doc_uploaded.get():
                # Process PDF-based questions with memory
                asyncio.create_task(process_pdf_query(chat, file_path.get(), user_query, input.api_key(), memory))
            else:
                asyncio.create_task(chat.append_message_stream("Please upload a PDF before asking a question."))

app = App(app_ui, server)
