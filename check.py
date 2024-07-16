import numpy as np
import torch
import transformers
import os
from data.load_data import save_embeddings

os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

# 初始化模型
model_name = "/data/model/llama-2-7b-chat-hf"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device('cuda:0')  # 选择第一个可见的 CUDA 设备
model.to(device)  # 将模型移动到选定的 CUDA 设备上

# 输入积极的句子
input_sentence = "How to make people happy?"
inputs = tokenizer(input_sentence, return_tensors="pt")
inputs = inputs.to(device)
print(inputs)

# 获取模型的embedding
with torch.no_grad():
    input_embeddings_org = model.get_input_embeddings()(inputs['input_ids'])

# 将CUDA张量转移到CPU上并转换为NumPy数组
input_embeddings_org_cpu = input_embeddings_org.cpu().numpy()

save_embeddings(input_embeddings_org_cpu, 'data/input_embeddings_org.npy')

# 将 input_ids 和 attention_mask 转移到 CPU 并保存为 NumPy 数组
input_ids_cpu = inputs['input_ids'].cpu().numpy()
attention_mask_cpu = inputs['attention_mask'].cpu().numpy()

np.save('data/input_ids.npy', input_ids_cpu)
np.save('data/attention_mask.npy', attention_mask_cpu)
