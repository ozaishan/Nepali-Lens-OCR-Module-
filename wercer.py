import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset as HFDataset, DatasetDict
import editdistance 
from jiwer import wer 
import matplotlib.pyplot as plt

# -------------------------
# CRNN Model Definition
# -------------------------
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
        conv = conv.squeeze(2)  # remove height dim (H=1)
        conv = conv.permute(2, 0, 1)  # (W, batch, channel)
        recurrent, _ = self.rnn(conv)
        output = self.fc(recurrent)
        return output

# -------------------------
# Dataset Wrapper
# -------------------------
class NepaliOCRDataset(Dataset):
    def __init__(self, hf_dataset, charset_file):
        self.dataset = hf_dataset
        self.charset = self._load_charset(charset_file)
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.idx_to_char = {i: c for i, c in enumerate(self.charset)}

    def _load_charset(self, charset_file):
        with open(charset_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]

    def encode_text(self, text):
        
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def ctc_greedy_decoder(self, preds):
        
        preds = preds.permute(1, 0, 2)  # (batch, seq_len, nclass)
        decoded_texts = []
        for pred in preds:
            pred_indices = torch.argmax(pred, dim=1).cpu().numpy()
            decoded = []
            prev = None
            for idx in pred_indices:
                if idx != 0 and idx != prev:
                    decoded.append(self.idx_to_char.get(idx, ''))
                prev = idx
            decoded_texts.append(''.join(decoded))
        return decoded_texts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        
        image_path = row['image_path']
        text = row['text']

        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 32))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).unsqueeze(0)  # (1, 32, 128)

        # Encode text labels
        target = torch.tensor(self.encode_text(text), dtype=torch.long)
        target_len = len(target)

        return img_tensor, target, target_len

# -------------------------
# Collate function for DataLoader
# -------------------------
def collate_fn(batch):
    images, targets, lengths = zip(*batch)
    images = torch.stack(images)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths)
    return images, targets, lengths

# -------------------------
# Metrics for OCR
# -------------------------
def cer(ref, hyp):
    # Character Error Rate using editdistance package
    return editdistance.eval(ref, hyp) / max(len(ref), 1)

def compute_metrics(gt_texts, pred_texts):
    total_cer = 0
    total_wer = 0
    n = len(gt_texts)

    for gt, pred in zip(gt_texts, pred_texts):
        total_cer += cer(gt, pred)
        total_wer += wer(gt, pred)

    avg_cer = total_cer / n
    avg_wer = total_wer / n
    return avg_cer, avg_wer



if __name__ == "__main__":
    # Load CSV and prepare HuggingFace dataset
    csv_path = "dataset/labels_comma.csv"
    images_folder = "dataset/crops"
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['image_path'] = df['image_file'].apply(lambda x: os.path.join(images_folder, x))

    # Split 80/20 train/validation
    train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    val_df = df.drop(train_df.index).reset_index(drop=True)

    hf_train_ds = HFDataset.from_pandas(train_df)
    hf_val_ds = HFDataset.from_pandas(val_df)
    hf_datasets = DatasetDict({"train": hf_train_ds, "validation": hf_val_ds})

    # Load charset (vocab)
    charset_file = "vocab.txt"

    # Create PyTorch Dataset and DataLoader
    train_dataset = NepaliOCRDataset(hf_datasets['train'], charset_file)
    val_dataset = NepaliOCRDataset(hf_datasets['validation'], charset_file)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Instantiate model
    nclass = len(train_dataset.charset)
    model = CRNN(imgH=32, nc=1, nclass=nclass, nh=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load pretrained weights 
    weights_path = "tension.pth"
    if os.path.exists(weights_path):
        print(f"Loading pretrained weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("No pretrained weights found. Please train the model first.")

    model.eval()

    # Prepare to save results
    output_file = "validation_results.txt"
    gt_texts = []
    pred_texts = []

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Ground Truth\tPrediction\n")
        with torch.no_grad():
            for images, targets, lengths in val_loader:
                images = images.to(device)
                outputs = model(images)  
                preds = train_dataset.ctc_greedy_decoder(outputs.cpu())

                for t, pred in zip(targets, preds):
                    gt = ''.join([train_dataset.idx_to_char[idx.item()] for idx in t if idx.item() != 0])
                    f.write(f"{gt}\t{pred}\n")
                    gt_texts.append(gt)
                    pred_texts.append(pred)

    avg_cer, avg_wer = compute_metrics(gt_texts, pred_texts)
    print(f"Validation CER: {avg_cer:.4f}, WER: {avg_wer:.4f}")
    print(f"Saved detailed results to {output_file}")
