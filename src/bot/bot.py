import streamlit as st
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output import GenerationChunk
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough
import os
import json
from langchain_gigachat import GigaChat

class HybridSearcher:
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL = "Qdrant/bm25"

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient("http://host.docker.internal:6333")
        self.qdrant_client.set_model(self.DENSE_MODEL)
        self.qdrant_client.set_sparse_model(self.SPARSE_MODEL)

    def search(self, text: str):
        search_result = self.qdrant_client.query(
            collection_name=self.collection_name,
            query_text=text,
            query_filter=None,
            limit=5,
        )
        metadata = [hit.metadata for hit in search_result]
        return metadata
    
searcher = HybridSearcher("beaty")

st.logo("./logo.png", size="large")
st.title("Помощник по подбору косметики")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.full_response += token
        self.container.markdown(self.full_response)

def analyze_image(image_bytes):
    pass
    # response = openai.ChatCompletion.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "Опиши это изображение и определи косметические продукты на лице человека"},
    #                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"}}
    #             ]
    #         }
    #     ],
    #     max_tokens=300,
    # )
    # return response.choices[0].message.content

def analyze_image_v2(image_bytes):

    credentials = """"""

    llm = GigaChat(
        credentials=credentials,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        model="GigaChat-2-Max",
        temperature=1,
        callbacks=[StreamlitCallbackHandler(st.empty())]
    )

    file = llm.upload_file(open(image_bytes, "rb"))

    content = llm.invoke([
        HumanMessage(
            content="Опиши и определи косметические продукты на лице человека",
            additional_kwargs={"attachments": [file.id_]},
        )
    ]).content
    return content

def extract_products(description):
    prompt = ChatPromptTemplate.from_template(
        "Извлеки названия косметических продуктов из этого описания: {description}"
    )

    credentials = """"""

    llm = GigaChat(
        credentials=credentials,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        model="GigaChat-2-Max",
        temperature=1,
        streaming=True,
        callbacks=[StreamlitCallbackHandler(st.empty())]
    )

    chain = prompt | llm
    return chain.invoke({"description": description}).content.split(", ")

def rag_pipeline(query):
    return '\n '.join([json.dumps(x, ensure_ascii=False, indent=2) for x in searcher.search(text=query)])

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Ты консультант по косметике. На основе следующих данных:
    
Анализ изображения: {description}
Рекомендуемые продукты: {context}
Запрос пользователя {query}
    
Составь дружелюбный ответ с пояснениями в виде маркированного списка с ценой. 
Для каждого продукта укажи ссылку, если она доступна, и объясни, почему он подходит."""
)

def extract_products_from_query(query):
    prompt = ChatPromptTemplate.from_template(
        "Извлеки названия косметических продуктов или категории из этого запроса: {query} \n Если их нет, то подбери подходящие продукты на основе запроса пользовател. Ответ дай в виде списка через запятую"
    )

    credentials = """"""

    llm = GigaChat(
        credentials=credentials,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        model="GigaChat-2-Max",
        temperature=1,
        streaming=True,
        callbacks=[StreamlitCallbackHandler(st.empty())]
    )

    chain = prompt | llm
    return chain.invoke({"query": query}).content.split(", ")

def get_image_path(img):
    file_path = f"data/uploadedImages/{img.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as img_file:
        img_file.write(img.getbuffer())
    return file_path

def process_input(input_text, image_bytes=None):

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    
    if image_bytes:
        description = analyze_image_v2(image_bytes)
        products = extract_products(description)
        context = "\n\n".join([rag_pipeline(product) for product in products])
    else:
        description = "Изображение не загружено"
        product_names = extract_products_from_query(input_text)
        context = "\n\n".join([rag_pipeline(product) for product in product_names])
    
    # llm = ChatOpenAI(
    #     model="o3-mini",
    #     temperature=1,
    #     streaming=True,
    #     callbacks=[StreamlitCallbackHandler(st.empty())]
    # )

    credentials = """"""

    llm = GigaChat(
        credentials=credentials,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        model="GigaChat-2-Max",
        temperature=1,
        streaming=True,
        callbacks=[StreamlitCallbackHandler(st.empty())]
    )
    
    chain = (
        {
            "query": RunnablePassthrough(),
            "history": RunnablePassthrough(),
            "description": RunnablePassthrough(),
            "context": RunnablePassthrough()
        }
        | FINAL_ANSWER_PROMPT 
        | llm
    )
    
    return chain.stream({
        "query": input_text,
        "history": history_str,
        "description": description,
        "context": context
        
    })

with st.sidebar:
    image = st.file_uploader("Загрузите фото желаемого образа", type=["jpg", "png"])
    #image = None
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.user_input = ""
        st.rerun()

input_text = st.chat_input("Задайте вопрос или загрузите фото")

if input_text or image:
    if image:
        st.session_state.messages.append({"role": "user", "content": "Загружено изображение"})
        with st.chat_message("user"):
            st.image(image, caption="Ваше фото")
    else:
        st.session_state.messages.append({"role": "user", "content": input_text})
        with st.chat_message("user"):
            st.markdown(input_text)
    
    with st.chat_message("assistant"):
        stream_container = st.empty()
        callback_handler = StreamlitCallbackHandler(stream_container)
        
        if image:
            #image_bytes = base64.b64encode(image.read()).decode("utf-8")
            image_path = get_image_path(image)
            response_generator = process_input(None, image_path)
        else:
            response_generator = process_input(input_text)
        
        final_response = ""
        for chunk in response_generator:
            if isinstance(chunk, GenerationChunk):
                final_response += chunk.content
                callback_handler.on_llm_new_token(chunk.content)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_response
        })