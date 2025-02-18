import os
import streamlit as st
from transformers import AutoModel, AutoTokenizer
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from proxy_llm import ProxyLLM, init_chain_proxy, init_knowledge_vector_store
from utils import (
    get_abs_path,
    tf_limit_memory
)
from configs.global_config import (
    MODEL_DIR,
    EMBEDDING_MODEL_DIR
)

models_folder = get_abs_path(MODEL_DIR)

MAX_CONTEXT = 720

st.set_page_config(
    page_title="测试",
    page_icon=":robot:",
    menu_items={"about": '''
                Author: FrostMiKu & Robin.Wang

                Model: ChatGLM-6B-INT4
                '''}
)


@st.cache_resource
def get_model():
    tf_limit_memory()
    tokenizer = AutoTokenizer.from_pretrained(
        models_folder, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        models_folder, trust_remote_code=True).half().cuda()
    model = model.eval()
    embeddings = HuggingFaceEmbeddings(model_name=get_abs_path(EMBEDDING_MODEL_DIR))
    # embeddings.client = sentence_transformers.SentenceTransformer(
    #     embeddings.model_name, device="cuda")
    return tokenizer, model, embeddings


@st.cache_resource
def get_vector_store(_embeddings_instance):
    vector_store = init_knowledge_vector_store(_embeddings_instance)
    return vector_store


if 'first_run' not in st.session_state:
    st.session_state.first_run = True
if 'history' not in st.session_state:
    st.session_state.history = []
if 'ctx' not in st.session_state:
    st.session_state.ctx = []

tokenizer, model, embeddings = get_model()
if 'vecdb' not in st.session_state:
    st.session_state.vecdb = get_vector_store(embeddings)
proxy_chain = init_chain_proxy(ProxyLLM(), st.session_state.vecdb, 5)

st.title("# 测试👋")
ctx_dom = st.empty()
question_dom = st.markdown(
    ">  回答由 AI 检索文件后生成，不保证准确率，仅供参考学习！"
)
md_dom = st.empty()
st.write("")


def display_ctx(history=None):
    if history != None:
        text = ""
        for index, (q, a) in enumerate(history):
            text += ":face_with_cowboy_hat:\n\n{}\n\n---\n{}\n\n---\n".format(
                q, a)
            ctx_dom.markdown(text)


def check_ctx_len(history):
    total = 0
    for q, a in history:
        total = total + len(q) + len(a)
    return total <= (MAX_CONTEXT + 10)


def predict(input, source, history=None):
    response = ""
    if history is None:
        history = []

    while not check_ctx_len(history):
        print("Free Context!")
        history.pop(0)

    for resp, history in model.stream_chat(tokenizer, input, history, max_length=4096, top_p=0.8,
                                           temperature=0.9):
        print(f"回答--->{resp}")
        print(f"历史--->{history}")
        md_dom.markdown(resp + source)
        response = resp + source
    q, _ = st.session_state.history.pop()
    st.session_state.history.append((q, response))
    history.pop()
    history.append(st.session_state.history[-1])
    return history


with st.form("form", True):
    # create a prompt text for the text generation
    prompt_text = st.text_area(label=":thinking_face: 咨询点什么？",
                               height=100,
                               max_chars=MAX_CONTEXT,
                               placeholder="支持使用 Markdown 格式书写")
    col1, col2 = st.columns([1, 1])
    with col1:
        btn_send = st.form_submit_button(
            "发送", use_container_width=True, type="primary")
    with col2:
        btn_clear = st.form_submit_button("清除历史记录", use_container_width=True)

    if btn_send and prompt_text != "":
        display_ctx(st.session_state.history)
        question_dom.markdown(
            ":face_with_cowboy_hat:\n\n{}\n\n---\n".format(prompt_text))
        q = proxy_chain(prompt_text)
        st.session_state.history.append((prompt_text, ''))
        # print(f"返回--->>>:{q}")
        seen_sources = set()
        for i, doc in enumerate(q["source_documents"]):
            source_name = os.path.split(doc.metadata['source'])[-1]
            if source_name in seen_sources:
                continue
            seen_sources.add(source_name)
        source = "\n\n"
        source += "".join(
            [
                f"""> *出处[{i + 1}] {name}*\n\n"""
                for i, name in
                enumerate(seen_sources)])
        st.session_state.ctx = predict(q['result'], source, st.session_state.ctx)
        if st.session_state.first_run:
            st.session_state.first_run = False
            st.balloons()

    if btn_clear:
        ctx_dom.empty()
        st.session_state.history = []
        st.session_state.ctx = []
