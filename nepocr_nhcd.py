import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset as HFDataset, DatasetDict
import matplotlib.pyplot as plt


csv_path = "dataset/labels_comma.csv"
images_folder = "dataset/crops"

# Load CSV with full image paths
df = pd.read_csv(csv_path, engine="python", encoding="utf-8")
df["image_path"] = df["image_file"].apply(lambda x: os.path.join(images_folder, x))

# Split 80/20 train/validation
train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
val_df = df.drop(train_df.index).reset_index(drop=True)


train_ds = HFDataset.from_pandas(train_df)
val_ds = HFDataset.from_pandas(val_df)
dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})


class NepaliOCRDataset(Dataset):
    def __init__(self, hf_dataset, vocab_file):
        self.dataset = hf_dataset
        self.charset = self._load_vocab(vocab_file)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}

    def _load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f.readlines()]
        return chars

    def encode_text(self, text):
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode_prediction(self, preds):
        return ''.join([self.idx_to_char.get(p, '') for p in preds])

    def ctc_greedy_decoder(self, preds):
        preds = preds.permute(1, 0, 2)  # (batch, seq_len, nclass)
        results = []
        for pred in preds:
            pred_indices = torch.argmax(pred, dim=1).cpu().numpy()
            decoded = []
            prev_idx = None
            for idx in pred_indices:
                if idx != 0 and idx != prev_idx:
                    decoded.append(self.idx_to_char.get(idx, ''))
                prev_idx = idx
            results.append(''.join(decoded))
        return results

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 32))
        image = image.astype('float32') / 255.0
        image = torch.tensor(image).unsqueeze(0)  # (1, 32, 128)
        target = torch.tensor(self.encode_text(row['text']), dtype=torch.long)
        return image, target, len(target)

# 3. CRNN Model Definition
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
        )
        self.rnn = nn.LSTM(input_size=512, hidden_size=nh, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2)  # Remove height dim (1)
        conv = conv.permute(2, 0, 1)  # [width, batch, channels]
        recurrent, _ = self.rnn(conv)
        output = self.fc(recurrent)
        return output

# 4. Collate function for DataLoader
def collate_fn(batch):
    images, targets, lengths = zip(*batch)
    images = torch.stack(images)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths)
    return images, targets, lengths

# 5. Training Function
def train_model(dataset_dict, vocab_file, finetune=True, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NepaliOCRDataset(dataset_dict["train"], vocab_file)
    val_dataset = NepaliOCRDataset(dataset_dict["validation"], vocab_file)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = CRNN(32, 1, len(train_dataset.charset), nh=256).to(device)

    if finetune and os.path.exists("devanagari.pth"):
        print("Loading pretrained Devanagari CNN weights from EasyOCR...")
        weights = torch.load("devanagari.pth", map_location=device)
        model.load_state_dict(weights, strict=False)

    lr = 0.0001 if finetune else 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "tension.pth"

    for epoch in range(100):
        model.train()
        total_loss = 0
        for images, targets, lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            preds_log = F.log_softmax(preds, dim=2)
            input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(device)
            loss = criterion(preds_log, targets, input_lengths, lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        # Validation loss calculation 
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets, lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                preds_log = F.log_softmax(preds, dim=2)
                input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(device)
                loss = criterion(preds_log, targets, input_lengths, lengths)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"  Validation Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print("  ↳ Validation loss improved. Saving model.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    print(f"\nLoading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, train_dataset, val_dataset

# 6. Prediction helper on a single image path
def predict_on_image(model, image_path, dataset):
    device = next(model.parameters()).device
    model.eval()

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 32))
    image = image.astype('float32') / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred_text = dataset.ctc_greedy_decoder(output)[0]
    return pred_text

# 7. Example visualization 
def show_prediction(model, dataset, index=0):
    image, target, _ = dataset[index]
    image_np = image.squeeze(0).numpy()
    gt_text = ''.join([dataset.idx_to_char[c.item()] for c in target])
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(next(model.parameters()).device))
        pred_text = dataset.ctc_greedy_decoder(pred)[0]

    plt.imshow(image_np, cmap='gray')
    plt.title(f"GT: {gt_text}\nPred: {pred_text}")
    plt.axis('off')
    plt.show()

# 8. Main script
if __name__ == "__main__":
    vocab_file = "vocab.txt"
    model, train_dataset, val_dataset = train_model(dataset_dict, vocab_file, finetune=True, patience=8)

    # Evaluate some predictions on validation set
    for i in range(5):
        show_prediction(model, val_dataset, i)

    # Predict on new unseen image example:
    example_image = "dataset/crops/crop_000.png"
    pred = predict_on_image(model, example_image, val_dataset)
    print(f"Prediction for {example_image}: {pred}")
