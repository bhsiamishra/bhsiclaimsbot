from typing import List
import PyPDF2
from io import BytesIO
from langchain_community.embeddings import OllamaEmbeddings, AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.docstore.document import Document
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
import os
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)



OPENAI_API_TYPE = "Azure"
OPENAI_API_VERSION = "2023-12-01-preview"
OPENAI_API_BASE = "https://documentapi.openai.azure.com/"
OPENAI_API_KEY = "31aa5c79cfad4f77809c9b439a583ff9"
DEPLOYMENT_NAME = "gpt-35-turbo"

from dotenv import load_dotenv

os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["DEPLOYMENT_NAME"] = DEPLOYMENT_NAME


load_dotenv()


prompt_template = """ Answer the question truthfully based solely on given document. If document do not contain the answer then say I don't know. 
 ----
{context}
----
"""
general_user_template = "Question:```{question}```"
messages = [
            SystemMessagePromptTemplate.from_template(prompt_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages( messages )


   
llm = AzureChatOpenAI(
    deployment_name = "gpt-35-turbo",
    model_name="gpt-35-turbo", 
    temperature=0
    
)

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment = "text-embedding-ada-002")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, prompt_template):

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    #python -m chainlit run rag.py
    # Construct the final prompt using prompt_template and user message
    # prompt = prompt_template.format(doc="", input=message_history)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': qa_prompt},
    )
    return conversation_chain

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


@cl.on_chat_start
async def on_chat_start():

    
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    # print(file)

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # file = files[0]
    # Read the PDF file
        
    #pdf_stream = BytesIO(content)
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
        

    # get the text chunks
    text_chunks = get_text_chunks(pdf_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    chain = get_conversation_chain(vectorstore, qa_prompt)

    # Let the user know that the system is ready
    msg.content = f" `{file.name}` is Uploaded. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         # Create the text element referenced in the message
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

    #     if source_names:
    #         answer += f"\nSources: {', '.join(source_names)}"
    #     else:
    #         answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
