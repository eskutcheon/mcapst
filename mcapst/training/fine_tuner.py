import torch

class FineTuner:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['training']['learning_rate']
        )
        self.criterion = torch.nn.MSELoss()  # Example loss function

    def fine_tune(self, train_dataset, freeze_layers=True):
        if freeze_layers:
            for name, param in self.model.named_parameters():
                if "decoder" not in name:  # Example: Freeze encoder layers
                    param.requires_grad = False

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
        )
        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            for content, style in dataloader:
                content = content.to(self.device)
                style = style.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(content, style)
                loss = self.criterion(output, content)  # Fine-tuning loss
                loss.backward()
                self.optimizer.step()
            print(f"Epoch [{epoch + 1}/{self.config['training']['epochs']}], Loss: {loss.item():.4f}")
