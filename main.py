from typing import Any

import bs4
import uvicorn
from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.base import AsyncCallbackHandler
from fastapi import WebSocket, FastAPI

app = FastAPI()

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

embed_model = OllamaEmbeddings(base_url='http://localhost:11434',
                               model="mxbai-embed-large")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class LLMCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.websocket.send_json({"message": token})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    callback_manager = AsyncCallbackManager(
        [LLMCallbackHandler(websocket)])

    llm = ChatOllama(
        base_url='http://localhost:11434',
        model="llama3",
        temperature=0,
        streaming=True,
        callback_manager=callback_manager
    )

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    while True:
        query = await websocket.receive_text()
        await rag_chain.ainvoke(query)


@app.get("/predict")
def predict(query: str):
    llm = ChatOllama(
        base_url='http://localhost:11434',
        model="llama3",
        temperature=0,
    )

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain.invoke(query)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
