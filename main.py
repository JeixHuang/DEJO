import os
import numpy as np
import torch
from data.load_data import load_embeddings, save_embeddings
from models.linear_transform import train_linear_transform
from data.generate_embeddings import generate_embeddings
from models.gan import train_gan
from utils.evaluation import evaluate_similarity, evaluate_harmfulness
from transformers import AutoTokenizer, AutoModelForCausalLM

# 检查嵌入文件是否存在
if not os.path.exists('data/embeddings.npy'):
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

# Compute common component
def compute_common_component(A, B):
    A_mean = np.mean(A, axis=0)
    B_mean = np.mean(B, axis=0)
    common_component = B_mean - A_mean
    return common_component

common_component = compute_common_component(A, B)
print(f'Common Component: {common_component}')

# Unembed common component to generate text
def unembed_common_component(common_component, model_name='/data/model/llama-2-7b-chat-hf', max_length=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 将通用分量转换为模型输入
    input_ids = torch.tensor(common_component).unsqueeze(0)
    input_ids = input_ids.to(torch.long)  # 确保数据类型为 Long

    # 生成文本
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=1
    )
    print(output_sequences)
    print('\n')
    # 解码生成的文本
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

# Unembed common component to generate text
generated_text = unembed_common_component(common_component)
print(f'Generated Text from Common Component: {generated_text}')
