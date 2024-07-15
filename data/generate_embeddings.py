from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from .load_data import save_embeddings

import numpy as np
import torch

def generate_embeddings(sentences, model, tokenizer, batch_size=8, max_length=128):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embeddings = []
    device = next(model.parameters()).device  # 获取模型当前的设备

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        inputs = tokenizer(batch_sentences, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
        
        # 将输入数据移动到设备上
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device) if 'attention_mask' in inputs else None
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取所有层的hidden states
        all_layer_embeddings = outputs.hidden_states
        # 计算每一层的平均池化表示，并在batch维度上进行堆叠
        batch_embeddings = torch.stack([layer.mean(dim=1) for layer in all_layer_embeddings], dim=0)
        
        all_embeddings.append(batch_embeddings)
    
    # 在batch维度上进行连接
    embeddings = torch.cat(all_embeddings, dim=1).permute(1, 0, 2)  # 形状变为 (num_batches * batch_size, num_layers, embedding_dim)
    embeddings = embeddings.cpu().numpy()
    return embeddings



# 示例文本，可以替换为实际数据
# sentences = [
#     "This is a harmless sentence.",
#     "This is another harmless sentence.",
#     "This is a harmful sentence.",
#     "This is another harmful sentence."
# ]

# # 生成并保存嵌入
# embeddings = generate_embeddings(sentences, model_name='/data/model/llama-2-7b-chat-hf')
# save_embeddings(embeddings, 'data/embeddings.npy')
