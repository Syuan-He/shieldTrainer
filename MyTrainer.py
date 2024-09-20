import tqdm
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from utils.DevConf import DevConf
from utils.AttnBlocksConf import AttnBlocksConf
from model.CombinationModel import CombinationModel

DEV_CONF = DevConf(device='cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        condition = df['is_run']==True
        self.df = df.dropna()[condition]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = torch.tensor(
            [[self.df.iloc[idx][i[1]], self.df.iloc[idx][i[0]]] for i in [
                    ("No Dangerous Content Positive", "No Dangerous Content Negative"),
                    ("No Harassment Positive", "No Harassment Negative"),
                    ("No Hate Speech Content Positive", "No Hate Speech Content Negative"),
                    ("No Sexually Explicit Information Content Positive", "No Sexually Explicit Information Content Negative")
                ]]
        )
        return text, label

class MyTrainer:
    def __init__(self, class_count: int, attention_config: AttnBlocksConf=AttnBlocksConf(768, 12, nKVHead=6), new_arch=False) -> None:
        self.class_count = class_count
        self.model = CombinationModel(class_count, attention_config, devConf=DEV_CONF, new_arch=new_arch)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-multilingual-cased", cache_dir='./cache/tokenizer')
        self.train_loader = None
        self.test_loader = None

    # dataset
    def set_dataset(self, train_data_path: str):
        train_data = pd.read_csv(train_data_path)

        dataset = MyDataset(train_data)

        datasize = len(dataset)
        splitIndex = int(datasize * 0.2)
        trainDataSize = datasize - splitIndex

        train_dataset, test_dataset = random_split(dataset, [trainDataSize, splitIndex])

        def collect_fn(batch):
            texts, labels = zip(*batch)
            return self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device=DEV_CONF.device), torch.stack(labels).to(DEV_CONF.device)
        self.train_loader = DataLoader(
            train_dataset, collate_fn=collect_fn, batch_size=8, shuffle=True,
            )
        self.test_loader = DataLoader(
            test_dataset, collate_fn=collect_fn, batch_size=1, shuffle=True,
            )

        print(trainDataSize, splitIndex)

    # Train
    def train(self, epochs: int = 1, lr: float=1e-5, log:bool = False):
        loss_fn = nn.KLDivLoss(reduction="mean")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        def train_fn(model, train_loader, loss_fn, optimizer, epochs, log):
            model.train()
            writer: SummaryWriter = None
            if log:
                writer = SummaryWriter()
            for epoch in range(epochs):
                for i, (data, label) in tqdm(enumerate(train_loader)):
                    optimizer.zero_grad()
                    output = model(**data)
                    output = torch.log(output)
                    loss = loss_fn(output, label.float())
                    loss.backward()
                    optimizer.step()
                    if i % 100 == 99:
                        print(f"Epoch {epoch+1}/{epochs} - Batch {i+1}/{len(train_loader)} - Loss: {loss.item()}")
                        if log:
                            writer.add_scalar('Loss/train', loss.item(), i + 1)
            if log:
                writer.flush()
                writer.close()

        self.model.train()
        train_fn(self.model, self.train_loader, loss_fn, optimizer, epochs, log)
        print("Done")

    # Eval
    def eval(self):
        self.model.eval()

        def test(model, test_loader):
            acc = [[[0, 0], [0, 0]] for _ in range(self.class_count)]
            for (data, label) in test_loader:
                output = torch.argmax(model(**data), dim=2).squeeze().cpu().numpy()
                ans = torch.argmax(label, dim=2).squeeze().cpu().numpy()
                for i in range(self.class_count):
                    acc[i][output[i]][ans[i]] += 1
            return acc

        confuse_matrix = test(self.model, self.test_loader)

        print("Confusion Matrix:")
        for matrix in confuse_matrix:
            print(matrix)

        microacc = [[0, 0], [0, 0]]
        for i in range(self.class_count):
            for j in range(2):
                for k in range(2):
                    microacc[j][k] += confuse_matrix[i][j][k]

        print("Microaverage:")
        print(microacc)

        macro = [{"Recall" : 0, "Precision" : 0, "F1" : 0, "Acc" : 0} for _ in range(len(confuse_matrix))]
        for id, matrix in enumerate(confuse_matrix):
            macro[id]["Precision"] = matrix[1][1] / (matrix[1][1] + matrix[1][0]) if matrix[1][1] + matrix[1][0] != 0 else 0
            macro[id]["Recall"] = matrix[1][1] / (matrix[1][1] + matrix[0][1]) if matrix[1][1] + matrix[0][1] != 0 else 0
            macro[id]["F1"] = 2 * macro[id]["Recall"] * macro[id]["Precision"] / (macro[id]["Recall"] + macro[id]["Precision"]) if macro[id]["Recall"] + macro[id]["Precision"] != 0 else 0
            macro[id]["Acc"] = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[1][1] + matrix[0][1] + matrix[1][0])

        total_macroaverage = {"Recall" : 0, "Precision" : 0, "F1" : 0, "Acc" : 0}

        print("Score:")
        for i in macro:
            print(i)
            total_macroaverage["Precision"] += i["Recall"] / len(macro)
            total_macroaverage["Recall"] += i["Precision"] / len(macro)
            total_macroaverage["F1"] += i["F1"] / len(macro)
            total_macroaverage["Acc"] += i["Acc"] / len(macro)

        print(f"macroaverage: {total_macroaverage}")

    def inference(self, text: str) -> list:
        self.model.eval()
        with torch.no_grad():
            data = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device=DEV_CONF.device)
            output = self.model(**data).squeeze()
            return output[:, 1].tolist()

    def save(self, path: str = 'myModel.pth'):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
