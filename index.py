# track_2
import json
import os
from transformers import AutoModel, AutoTokenizer
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


def get_model():
    tf_limit_memory()
    tokenizer = AutoTokenizer.from_pretrained(
        models_folder, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        models_folder, trust_remote_code=True).half().cuda()
    model = model.eval()
    embeddings = HuggingFaceEmbeddings(model_name=get_abs_path(EMBEDDING_MODEL_DIR))
    return tokenizer, model, embeddings


def get_vector_store(_embeddings_instance):
    vector_store = init_knowledge_vector_store(_embeddings_instance)
    return vector_store


tokenizer, model, embeddings = get_model()
vecdb = get_vector_store(embeddings)
proxy_chain = init_chain_proxy(ProxyLLM(), vecdb, 5)


def predict(input, history=None):
    print(f"预测--->{input}")
    for resp, history in model.stream_chat(tokenizer, input, history, max_length=4096, top_p=0.8,
                                           temperature=0.9):
        print(f"回答--->{resp}")
        return resp


def invoke(questions_path):
    with open(questions_path, 'r', encoding='UTF-8') as file:
        data = file.readlines()
        questions = []
        for line in data:
            questions.append(json.loads(line))
        # 测试用
        print(len(questions))

    # use your finetuned model to do inference

    NUM_OF_QUESTIONS = 1
    results = {}
    for i in range(NUM_OF_QUESTIONS):
        # 问题
        question = f"{questions[i]}"
        q = proxy_chain(question)
        print(f"返回--->>>:{q}", flush=True)
        seen_sources = set()
        for x, doc in enumerate(q["source_documents"]):
            source_name = os.path.split(doc.metadata['source'])[-1]
            if source_name in seen_sources:
                continue
            seen_sources.add(source_name)
        source = "\n\n"
        source += "".join(
            [
                f"""出处[{i + 1}] {name}"""
                for i, name in
                enumerate(seen_sources)])
        # 提示词
        prompt = f"{q['result']}"
        # 回答
        answer = predict(q['result'])
        # 相关文件title
        reference = source
        d = {}
        d[prompt] = [answer, reference]
        results[question] = d

    return results


if __name__ == "__main__":
    results = invoke('questions')
    # print(results)
