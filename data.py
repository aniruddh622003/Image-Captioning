import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import nltk
from collections import Counter

# print(os.listdir('Dataset/Images'))

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def denormalize(image):
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return image
    

class MyDataset(Dataset):
    def __init__(self, img_path, captions_path, transform=None):
        self.img_paths = os.listdir(img_path)
        self.caption_path = captions_path
        self.captions = pd.read_csv(captions_path)
        self.transform = transform

        # Generate vocab
        self.vocab = self.generate_vocab()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        image = Image.open(os.path.join('Dataset/Images', self.captions.iloc[index, 0]))
        if self.transform:
            image = self.transform(image)
        caption = self.captions.iloc[index, 1]
        caption = self.generate_tensor(caption)
        return image, caption
    
    def generate_vocab(self):
        captions = self.captions['caption'].values
        tokens = [nltk.tokenize.word_tokenize(caption.lower()) for caption in captions]

        # Generate word counts
        word_counts = Counter([token for caption_tokens in tokens for token in caption_tokens])

        # Generate vocab
        vocab = ['<pad>', '<start>', '<end>', '<unk>'] + [word for word, count in word_counts.items() if count >= 5]
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        
        return word2idx
    
    def generate_tensor(self, caption, max_len=50):
        caption = caption.split()

        pad_idx = self.vocab['<pad>']
        start_idx = self.vocab['<start>']
        end_idx = self.vocab['<end>']

        seq = [self.vocab.get(word, self.vocab['<unk>']) for word in caption]

        caption = [start_idx] + [self.vocab.get(word, self.vocab['<unk>']) for word in caption] + [end_idx]
        caption = caption + [pad_idx] * (max_len - len(caption)) if len(caption) < max_len else caption[:max_len]
        return torch.tensor(caption)
