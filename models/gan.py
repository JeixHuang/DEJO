import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(True),
            nn.Linear(512, 768),
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)

def train_gan(A, B, num_epochs=1000, learning_rate=0.001):
    A_tensor = torch.tensor(A, dtype=torch.float32)
    B_tensor = torch.tensor(B, dtype=torch.float32)

    G = Generator()
    D = Discriminator()

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(D.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(A_tensor.size(0), 1)
        fake_labels = torch.zeros(A_tensor.size(0), 1)

        outputs = D(B_tensor)
        d_loss_real = criterion(outputs, real_labels)

        z = A_tensor
        fake_vectors = G(z)
        outputs = D(fake_vectors.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        outputs = D(fake_vectors)
        g_loss = criterion(outputs, real_labels)

        g_loss.backward()
        optimizer_G.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    
    return G
