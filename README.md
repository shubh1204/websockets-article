# [Websocket based Streaming with Fast API and Local LLAMA 3](https://medium.com/@shubham.mdsk/websocket-based-streaming-with-fast-api-and-local-llama-3-46f88eda71a2)

Large Language Models (LLMs) may require a significant amount of time to generate complete response to queries depending on the response size and token generation rate.
Delivering the response from the API only after the LLM has fully generated it could result in a suboptimal user experience, as users would be left waiting on the interface with a loading screen for an extended duration.

Solution
In addressing this challenge, one effective solution is the utilization of StreamingResponse methodology. This can be implemented through the application of HTTP-based StreamingResponse https://fastapi.tiangolo.com/advanced/custom-response/ or by utilizing Websockets. In this discussion, we will focus on the implementation of the Websockets approach.

For a concise comparison between HTTP and Websockets, readers can refer to the insights provided at https://ably.com/topic/websockets-vs-http

The focus of this article will be on developing a straightforward API based on RAG (Retrieval-Augmented Generation) that exposes a Websocket-dependent endpoint capable of streaming responses from Large Language Models (LLMs). For this particular application, we will work with a localized instance of LLAMA 3 deployed using Ollama technology.

This article assumes that readers have basic understanding of LLMs, REST API and have basic programming skills. With that said, Let’s begin:

Implementation
Step 1: Deploying Local LLM

When deploying large language models (LLMs) on local machines or clusters, the choice of frameworks and libraries depends on the specific use case and infrastructure setup. Here are some examples of frameworks and libraries that can be used:

llama.cpp: A framework designed for efficient deployment of LLMs. It provides tools and utilities for managing memory, optimizing inference, and handling large-scale language models.
vllm: A lightweight library that focuses on fast inference for LLMs. It aims to minimize resource usage while maintaining high performance.
GPT4all: A versatile library that supports various LLM architectures, including GPT-3 and beyond. It offers pre-trained models, fine-tuning capabilities, and efficient deployment options.
Hugging Face: A popular open-source platform that provides pre-trained LLMs, fine-tuning tools, and model serving solutions. It supports a wide range of LLM architectures and is widely used in research and production.
Remember that the choice of framework/library should align with your specific requirements, such as model size, latency, and scalability. Each of these options has its strengths and trade-offs, so it’s essential to evaluate them based on your project’s needs.

For this article we will use Ollama with llama3–7b model. Ollama is available as an executable on Mac, Linux and Windows https://ollama.com/download as well as, as docker image https://hub.docker.com/r/ollama/ollama

Once installed, Open a terminal and enter below command to pull llama3 model and mxbai-embed-large embedding model . By default this pulls the 7b model.

ollama pull llama3

ollama pull mxbai-embed-large
Step 2: Defining a simple RAG use-case with local Embedding model and LLM

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI
import bs4
import uvicorn

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


@app.get("/predict")
def predict(query: str):
    return rag_chain.invoke(query)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
Query:

curl -X 'GET' \                    
  'http://127.0.0.1:8000/predict?query=What is Task Decomposition?'
Output:

Task Decomposition is a technique that involves breaking down complex tasks into smaller, simpler steps. This is done by instructing the model to "think step by step" and utilize more test-time computation to decompose hard tasks. The goal is to transform big tasks into multiple manageable tasks, providing insight into the model's thinking process.
Step 3: Creating a Websocket endpoint and adding callback managers

The current /predict endpoint returns entire response at once, which may take time depending on retrievals, model size, token generation rate etc. Hence we want to update the service to return streaming response in Chat GPT style.
We can do it by using HTTP (GET or POST) based StreamingResponse as well, but here we will discuss about Websocket based approach.
To read more about websockets using FAST API, visit this link. https://fastapi.tiangolo.com/advanced/websockets/

Step 3.1: Define a Callback handler which inherits from Langchain’s AsyncCallbackHandler with on_llm_new_token

For details on callback and see other available methods, visit this link: https://python.langchain.com/v0.1/docs/modules/callbacks/

class LLMCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.websocket.send_json({"message": token})
Above we define a custom callback handler which send token in json format whenever a new token is received from LLM.

Step 3.2: Update LLM definition

Update LLM Definition and add Streaming=True and Async Callback

llm = ChatOllama(
    base_url='http://localhost:11434',
    model="llama3",
    temperature=0,
    streaming=True,
    callback_manager=self.callback_manager
)
Step 3.3: Add Websocket endpoint

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
You can then query this endpoint using Terminal, Postman or any other library / UI to get streaming response as soon as tokens are generated.

Conclusion
This article shows how to use websockets in conjunction with callbacks to return streaming response. Full code can be found here on Git: https://github.com/shubh1204/websockets-article.git

Next Steps
From here you can learn how to implement same using HTTP based streaming. You can also dig more into callbacks and amazing thing / transformations you can do by using them.
