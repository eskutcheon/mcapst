import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['training']['learning_rate']
        )
        self.criterion = torch.nn.MSELoss()  # Example loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_dataset):
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            for content, style in dataloader:
                content = content.to(self.device)
                style = style.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(content, style)
                loss = self.criterion(output, content)  # Example loss computation
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{self.config["training"]["epochs"]}], Loss: {loss.item():.4f}')
