import getpass
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from operator import itemgetter
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from shiny import ui, render, App, reactive

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load and split PDF
# file_path = "data/codes-greenville-ny.pdf"
# loader = PyPDFLoader(file_path)
# pages = loader.load_and_split()

# Create FAISS index and retriever
faiss_index = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization = True)
retriever = faiss_index.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)

# Define LLM and prompts
llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)

system_prompt = (
    "You're a helpful AI assistant. Given a user question about county codes and some county code documents, answer the user question. If none of the documents can answer the question, just say you don't know.\n\nHere are the county code documents:{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
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
answer = prompt | llm | StrOutputParser()

chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format)
    .assign(answer=answer)
    .pick(["answer", "docs"])
)

# Define the UI layout
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_dark_mode(),
        ui.h2("Ask a Question"),
        ui.p("This app uses GPT-4o-mini to return information about the town codes in your area."),
        ui.input_text_area("textarea", "Input question", "", height='400px'),
        ui.input_action_button("action_button", "Ask Question"),
        width=420
    ),
    ui.layout_column_wrap(
        ui.panel_well(
            ui.h4("Answer:"),
            ui.output_text_verbatim("value")
        )
    ),
    ui.layout_column_wrap(
        ui.panel_well(
            ui.h4("Relevant Documents:"),
            ui.output_text_verbatim("docs")
        ),
    ),
    title="Town Codes for [MY TOWN] - Retreival Augmented Generation App"
)

# Define server logic to handle user actions
def server(input, output, session):
    # Create reactive values to store the output
    text_value = reactive.Value("")
    docs_value = reactive.Value("")

    # Define a reactive effect to update the values when the button is clicked
    @reactive.Effect
    @reactive.event(input.action_button)
    def _():
        question = input.textarea()
        result = chain.invoke(question)
        text_value.set(result['answer'])
        
        # Format only the first document
        # if result['docs']:
        #     first_doc = result['docs'][0]
        #     formatted_doc = f"Page Number: {first_doc.metadata.get('page_number', 'Unknown')}\n\nPage Content:\n{first_doc.page_content}"
        #     docs_value.set(formatted_doc)
        # else:
        #     docs_value.set("No relevant documents found.")
        # Format up to three documents
        if result['docs']:
            formatted_docs = []
            for i, doc in enumerate(result['docs'][:3]):
                formatted_doc = f"Document {i+1}:\nPage Number: {doc.metadata.get('page_number', 'Unknown')}\n\nPage Content:\n{doc.page_content}"
                formatted_docs.append(formatted_doc)
            docs_value.set("\n\n".join(formatted_docs))
        else:
            docs_value.set("No relevant documents found.")

    # Define the output text verbatim for the answer
    @output()
    @render.text
    def value():
        return text_value.get()

    # Define the output text verbatim for the docs
    @output()
    @render.text
    def docs():
        return docs_value.get()

# Create and run the Shiny app
app = App(app_ui, server)
