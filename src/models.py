# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import Counter

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.Wa = nn.Linear(encoder_dim, attention_dim)
        self.Ua = nn.Linear(decoder_dim, attention_dim)
        self.Va = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, hidden):
        # encoder_out: (B, 49, encoder_dim)
        # hidden: (B, decoder_dim)
        scores = self.Va(torch.tanh(
            self.Wa(encoder_out) + self.Ua(hidden).unsqueeze(1)
        ))                              # (B, 49, 1)
        weights = F.softmax(scores, dim=1)
        # weights: attention map (for visualization to show where the attention is looking at when it generated each word)
        # weights that sum to 1.0 over 49 regions on the image, with the highest weighted region means the highest score -> signal the focus on the image

        context = (weights * encoder_out).sum(dim=1)  # (B, encoder_dim)
        # context: attended summary of the image
        # each of the 49 encoder region vectors is multiplied by its attention weight

        return context, weights

class Encoder(nn.Module):
    def __init__(self, encoded_size, embed_dim, init_dim, drop_out):
        super().__init__()
        # 1. Load pretrained ResNet-18
        resnet = models.resnet18(weights="IMAGENET1K_V1") # pretrained weights = "IMAGENET1K_V1"

        # 2. Remove last 2 layers avgpool + fc, keep spatial map for our task
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.init_dim = init_dim

        # self.attention_dim = 256 # typically the same as embed_dim
        # self.hidden_dim = 512 # typically x2 embed_dim
        # self.drop_out = 0.5 # regularization
        # 3. Freeze all weights - no retraining on ResNet weights
        # freeze the first 16 layers closer to the images (use it, no change) - useful for our task
        # just train the last 2 images

        for p in self.resnet.parameters():
            p.requires_grad = False

        # ResNet contains CNN layers at the end
        # Add another CNN for my task: conv -> activation (ReLU) -> pool -> fc -> dropout
        '''
        self.conv = nn.Conv2d( // not deleted the conv layer above
            in_channels=3, #grayscale input
            out_channels=3, #3 filters (feature maps)
            kernel_size=3, #5×5 filters
            stride=2, #move 1 pixel at a time
            padding=0 #no padding
        )'''

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size)) # removed the pool and fc above, train new pool and fc here
        self.fc = nn.Linear(init_dim, embed_dim) # init_dim = 512, projected down to embed_dim = 256
        self.drop_out = nn.Dropout(drop_out) # medium drop out, prevent overfitting
        # drop out 30% of the neurons (experiment, (not to focus too much on specific regions?) )


    def forward(self, images):
        '''
        Returns a tensor of shape (B, 49, 256). Each of the 49 rows is a 256-dimensional vector representing one spatial region of the image.
        The decoder's attention mechanism will use these 49 vectors to decide which part of the image to focus on when generating each word.
        '''
        with torch.no_grad(): # not calc the radiant for the resnet (above in the constructor)
            features = self.resnet(images)           # (B, 512,  7,  7)

        features = self.pool(features)                   # (B, 512,  7,  7) — already 7x7
        features = features.permute(0, 2, 3, 1)          # (B,  7,   7, 512)
        features = features.view(features.size(0), -1, self.init_dim)  # (B, 49,      512)
        features = self.drop_out(self.relu(
               self.fc(features)))                   # (B, 49,      256)

        return features

class Decoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size,
                 encoder_dim, attention_dim, dropout):
        super().__init__()
        self.attention = BahdanauAttention(
            encoder_dim, decoder_dim, attention_dim) # attention lives in the Decoder
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.init_h = nn.Linear(encoder_dim, decoder_dim) # hidden layers/ hidden state in the lstm model
        self.init_c = nn.Linear(encoder_dim, decoder_dim) # cell state:
        self.vocab_size = vocab_size

    def init_hidden(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        return (torch.tanh(self.init_h(mean)),
                torch.tanh(self.init_c(mean)))

    def forward(self, encoder_out, captions):
        B = encoder_out.size(0)
        vocab_size = self.fc.out_features
        seq_len = captions.size(1) # Corrected: should be captions.size(1), not captions.size(1) - 1
        preds = torch.zeros(B, seq_len, vocab_size).to(device)
        h, c = self.init_hidden(encoder_out)

        for t in range(seq_len):
            emb = self.embedding(captions[:, t])
            context, weights = self.attention(encoder_out, h)
            h, c = self.lstm_cell(torch.cat([emb, context], dim=1), (h, c))
            preds[:, t, :] = self.fc(self.dropout(h))

        return preds
    
class Vocabulary:
    def __init__(self):
        self.start = "<START>" # start of sentence, tell LTSM to begin generating
        self.end = "<END>" # end of sentence, tell LTSM to stop generating
        self.pad = "<PAD>" # padding, fill empty space at the end of short captions
        self.unk = "<UNK>" # unknow words, for words not in keys
        self.freq_threshold = 4

        self.wordToIdx = {
            self.start: 0,
            self.end: 1,
            self.pad: 2,
            self.unk: 3
        }

        self.idxToWord = {
            0: self.start,
            1: self.end,
            2: self.pad,
            3: self.unk
        }

    def build(self, captions):
        """
        Build vocabulary: word - index dictionary based on all captions
        Build the counter of frequencies of all unique words in all the captions
        Only save words that appear more than freq_threshold times
        """
        counter = Counter()
        for caption in captions:
            counter.update(caption.lower().split())

        idx = 4 # 0-3 used for start, end, pad, unk
        for word, count in counter.items():
            if count >= self.freq_threshold:
                self.wordToIdx[word] = idx
                self.idxToWord[idx] = word
                idx += 1

        print(f"Vocabulary size: {len(self.wordToIdx)}")
        print(f"Word To Index dictionary: {self.wordToIdx}")
        return (self.wordToIdx)

    def encode(self, caption):
        """
        Convert a caption into a list of integers
        If a word in the caption is not in the dictionary wordToIndx, place it as <unk>
        """
        encoded_caption = [self.wordToIdx.get(word, self.wordToIdx[self.unk]) for word in caption.lower().split()]
        # print(f"encoded_caption: {encoded_caption}")
        return encoded_caption

    def decode(self, indices):
        """
        Convert a list of integers/ indices into a caption
        Check for start and end
        """
        words = []
        idxStart = self.wordToIdx[self.start]
        idxEnd = self.wordToIdx[self.end]
        idxPad = self.wordToIdx[self.pad]

        for idx in indices:
            if idx == idxEnd:
                break
            if idx not in (idxStart, idxPad):
                words.append(self.idxToWord.get(idx, self.unk))

        return " ".join(words)
