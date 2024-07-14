import torch
import torch.nn as nn
import torch.optim as optim

class LinearTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)

def train_linear_transform(A, B, num_epochs=100, learning_rate=0.01):
    A_tensor = torch.tensor(A, dtype=torch.float32)
    B_tensor = torch.tensor(B, dtype=torch.float32)

    model = LinearTransform(A.shape[1], B.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        B_pred = model(A_tensor)
        loss = criterion(B_pred, B_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model
