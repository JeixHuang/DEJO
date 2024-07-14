from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from .load_data import save_embeddings

def generate_embeddings(sentences, model_name='/data/model/llama-2-7b-chat-hf'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 添加 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        # 提取最后一层的平均池化表示
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)

    embeddings = np.vstack(embeddings)
    return embeddings

# 示例文本，可以替换为实际数据
sentences = [
    "This is a harmless sentence.",
    "This is another harmless sentence.",
    "This is a harmful sentence.",
    "This is another harmful sentence."
]

# 生成并保存嵌入
embeddings = generate_embeddings(sentences, model_name='/data/model/llama-2-7b-chat-hf')
save_embeddings(embeddings, 'data/embeddings.npy')
