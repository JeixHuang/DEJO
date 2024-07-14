from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from load_data import save_embeddings

def generate_embeddings(sentences, model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        # 对于大多数模型，取最后一层的第一个 token (通常是 [CLS] token) 的向量表示
        embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())

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
embeddings = generate_embeddings(sentences, model_name='bert-base-uncased')
save_embeddings(embeddings, 'data/embeddings.npy')
