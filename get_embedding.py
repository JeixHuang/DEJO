import os
import numpy as np
import torch
from data.load_data import load_embeddings, save_embeddings
from models.linear_transform import train_linear_transform
from data.generate_embeddings import generate_embeddings
from models.gan import train_gan
from utils.evaluation import evaluate_similarity, evaluate_harmfulness
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 设置可见的 CUDA 设备为 cuda:0

def load_model_from_json(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    model_name = config["model_name"]
    padding_side = config.get("padding_side", "left")  # 如果 JSON 文件中没有指定，默认使用 'left'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = padding_side
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    return tokenizer, model

# Function to load sentences from text files
def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]
    return sentences

json_file = "model_config.json"

# 从 JSON 文件加载模型和 tokenizer
tokenizer, model = load_model_from_json(json_file)
device = torch.device('cuda:0')  # 选择第一个可见的 CUDA 设备
model.to(device)  # 将模型移动到选定的 CUDA 设备上

# Load positive and negative sentences from text files
positive_sentences = load_sentences('data/positive.txt')
negative_sentences = load_sentences('data/negative.txt')

# Generate embeddings if they don't already exist
if not (os.path.exists('data/embeddings_positive.npy') and os.path.exists('data/embeddings_negative.npy')):
    embeddings_po = generate_embeddings(positive_sentences, model=model, tokenizer=tokenizer)
    embeddings_ne = generate_embeddings(negative_sentences, model=model, tokenizer=tokenizer)
    save_embeddings(embeddings_po, 'data/embeddings_positive.npy')
    save_embeddings(embeddings_ne, 'data/embeddings_negative.npy')

# Load embeddings
embeddings_po = load_embeddings('data/embeddings_positive.npy')
embeddings_ne = load_embeddings('data/embeddings_negative.npy')
print(f"{embeddings_po.shape}")
print(f"{embeddings_ne.shape}")

# 每个(a, b)对的common component
def compute_common_components(A, B):
    return B - A

# 计算 common components，并保存所有层的结果
common_components = compute_common_components(embeddings_po, embeddings_ne)
print(f'Common Components: {common_components}')
print(f'Common Components shape: {common_components.shape}')

np.save('data/A.npy', embeddings_po)
np.save('data/B.npy', embeddings_ne)
np.save('data/common_components.npy', common_components)

print(f'A saved with shape: {embeddings_po.shape}')
print(f'B saved with shape: {embeddings_ne.shape}')
print(f'Common components saved with shape: {common_components.shape}')

# # 针对每一层分别进行处理
# num_layers = embeddings_po.shape[1]
# similarity_linear_list = []
# similarity_gan_list = []
# harmfulness_linear_list = []
# harmfulness_gan_list = []

# for layer in range(num_layers):
#     embeddings_po_layer = embeddings_po[:, layer, :]
#     embeddings_ne_layer = embeddings_ne[:, layer, :]

#     # Train linear transformation model
#     linear_model = train_linear_transform(embeddings_po_layer, embeddings_ne_layer)

#     # Generate transformed embeddings using the linear model
#     A_tensor = torch.tensor(embeddings_po_layer, dtype=torch.float32)
#     B_pred_linear = linear_model(A_tensor).detach().numpy()

#     # Train GAN model
#     gan_generator = train_gan(embeddings_po_layer, embeddings_ne_layer)

#     # Generate embeddings using the GAN model
#     B_pred_gan = gan_generator(A_tensor).detach().numpy()

#     # Evaluate similarity
#     similarity_linear = evaluate_similarity(B_pred_linear, embeddings_ne_layer)
#     similarity_gan = evaluate_similarity(B_pred_gan, embeddings_ne_layer)
#     print(f'Linear Model Average Similarity for Layer {layer}: {similarity_linear:.4f}')
#     print(f'GAN Model Average Similarity for Layer {layer}: {similarity_gan:.4f}')

#     # Store similarity results
#     similarity_linear_list.append(similarity_linear)
#     similarity_gan_list.append(similarity_gan)

#     # Evaluate harmfulness (placeholder model for harmfulness evaluation)
#     class HarmfulnessModel:
#         def predict(self, vectors):
#             # Placeholder harmfulness scores
#             return np.random.rand(len(vectors))

#     harmfulness_model = HarmfulnessModel()
#     harmfulness_linear = evaluate_harmfulness(B_pred_linear, harmfulness_model)
#     harmfulness_gan = evaluate_harmfulness(B_pred_gan, harmfulness_model)
#     print(f'Linear Model Average Harmfulness for Layer {layer}: {harmfulness_linear:.4f}')
#     print(f'GAN Model Average Harmfulness for Layer {layer}: {harmfulness_gan:.4f}')

#     # Store harmfulness results
#     harmfulness_linear_list.append(harmfulness_linear)
#     harmfulness_gan_list.append(harmfulness_gan)

# # Print overall results
# print('Overall Linear Model Similarity per Layer:', similarity_linear_list)
# print('Overall GAN Model Similarity per Layer:', similarity_gan_list)
# print('Overall Linear Model Harmfulness per Layer:', harmfulness_linear_list)
# print('Overall GAN Model Harmfulness per Layer:', harmfulness_gan_list)
