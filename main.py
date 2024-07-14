import os
import torch
import numpy as np
from data.load_data import load_embeddings, save_embeddings
from data.generate_embeddings import generate_embeddings
from models.linear_transform import train_linear_transform
from models.gan import train_gan
from utils.evaluation import evaluate_similarity, evaluate_harmfulness

# 检查嵌入文件是否存在
if not os.path.exists('data/embeddings.npy'):
    from generate_embeddings import generate_embeddings, save_embeddings
    # 示例文本，可以替换为实际数据
    sentences = [
        "This is a harmless sentence.",
        "This is another harmless sentence.",
        "This is a harmful sentence.",
        "This is another harmful sentence."
    ]
    # 生成并保存嵌入
    embeddings = generate_embeddings(sentences)
    save_embeddings(embeddings, 'data/embeddings.npy')

# Load embeddings
embeddings = load_embeddings('data/embeddings.npy')

# 假设前半部分是无害文本，后半部分是有害文本
A = embeddings[:len(embeddings)//2]
B = embeddings[len(embeddings)//2:]

# Train linear transformation model
linear_model = train_linear_transform(A, B)

# Generate transformed embeddings using the linear model
A_tensor = torch.tensor(A, dtype=torch.float32)
B_pred_linear = linear_model(A_tensor).detach().numpy()

# Train GAN model
gan_generator = train_gan(A, B)

# Generate embeddings using the GAN model
B_pred_gan = gan_generator(A_tensor).detach().numpy()

# Evaluate similarity
similarity_linear = evaluate_similarity(B_pred_linear, B)
similarity_gan = evaluate_similarity(B_pred_gan, B)
print(f'Linear Model Average Similarity: {similarity_linear:.4f}')
print(f'GAN Model Average Similarity: {similarity_gan:.4f}')

# Evaluate harmfulness (placeholder model for harmfulness evaluation)
class HarmfulnessModel:
    def predict(self, vectors):
        # Placeholder harmfulness scores
        return np.random.rand(len(vectors))

harmfulness_model = HarmfulnessModel()
harmfulness_linear = evaluate_harmfulness(B_pred_linear, harmfulness_model)
harmfulness_gan = evaluate_harmfulness(B_pred_gan, harmfulness_model)
print(f'Linear Model Average Harmfulness: {harmfulness_linear:.4f}')
print(f'GAN Model Average Harmfulness: {harmfulness_gan:.4f}')
