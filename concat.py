import numpy as np
import torch
import transformers
import os
from data.load_data import save_embeddings
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

device = torch.device('cuda:0')  # 选择第一个可见的 CUDA 设备

model_name = "/data/model/llama-2-7b-chat-hf"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)  # 将模型移动到选定的 CUDA 设备上

input_ids = np.load('data/input_ids.npy')
attention_mask = np.load('data/attention_mask.npy')
input_embeddings_org=np.load('data/input_embeddings_org.npy')

input_ids = torch.tensor(input_ids).to(device)
attention_mask = torch.tensor(attention_mask).to(device)
input_embeddings=torch.tensor(input_embeddings_org,device=device)

inputs = {
    'input_ids': input_ids,
    'attention_mask': attention_mask
}
# 加载common_component.npy文件
common_component = np.load('data/common_components.npy')
print(common_component.shape)
# 将NumPy数组转换为PyTorch张量
common_component_tensor = torch.tensor(common_component,device=device)
# 选择common_component第二个维度的第33层 (注意索引从0开始，因此33层是index 32)
selected_layer = common_component_tensor[:, 32, :]
selected_layer.to(device)
# 将第一个维度压缩为一个平均值，得到 (1, 4096) 的张量
compressed_tensor = torch.mean(selected_layer, dim=0, keepdim=True)
compressed_tensor.to(device)
# 确保compressed_tensor和input_embeddings的形状一致
# 这里假设input_embeddings的形状为 (batch_size, sequence_length, hidden_size)
batch_size, sequence_length, hidden_size = input_embeddings.shape
print(input_embeddings.shape)
expanded_compressed_tensor = compressed_tensor.expand(batch_size, sequence_length, -1)
print(expanded_compressed_tensor.shape)
# 将输入句子的embedding与compressed_tensor相加
input_embeddings.to(device)
expanded_compressed_tensor.to(device)
combined_embeddings = input_embeddings + expanded_compressed_tensor
# combined_embeddings = input_embeddings
combined_embeddings.to(device)
# 使用模型生成新的句子
# 我们不能直接传入combined_embeddings给generate函数，因此需要特殊处理
model.get_input_embeddings().weight.data[:combined_embeddings.shape[1], :] = combined_embeddings[0, :, :]

outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

# 解码生成的句子
output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(output_sentence)
