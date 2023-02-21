---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/QuadV/ImplementingPapers/blob/main/SequenceToSequenceInNeuralNetworks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

```python id="38FlHHZi1jyY"
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
from tqdm import tqdm
```

```python id="UlAT8ZKvOBqJ" outputId="6de48f4c-42ae-4d30-89e1-2fd8dfbb31fb" colab={"base_uri": "https://localhost:8080/", "height": 496}
! python -m spacy download de
```

```python id="wU_Xf8c5XHMe"
spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')
```

```python id="cvSww3vTNkJj"
def tokenizer_ger(text):
  """ Hello my name -> ['Hello', 'my', 'name']"""
  return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
  return [tok.text for tok in spacy_eng.tokenizer(text)]
```

```python id="LaFf1ipaOj8S"
german = Field(tokenize=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos>')
```

```python id="8C7yQeQUPLlc"
train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)
```

```python id="YI97muI0R7JV"
class Encoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
    super(Encoder, self).__init__()
    self.num_layers = num_layers
    self.hidden_size= hidden_size

    self.dropout = nn.Dropout(p)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

  def forward(self, x):
    # x shape = (seq_length, N) # seq_length of words in N batches

    embedding = self.dropout(self.embedding(x))
    # embedding shape: (seq_len, N, embedding_size)
    output, (hidden, cell) = self.rnn(embedding) 
    return hidden, cell

```

```python id="Z-7tPMuyR_pi"
class Decoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, output_size,
               num_layers, p): # input_size=output_size coz it will be prob of word in vocab 10000
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.dropout = nn.Dropout(p)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden, cell):
    # shape of x: (N) but we want (1, N) # 1 word at a time in N batches
    x = x.unsqueeze(0)

    embedding = self.dropout(self.embedding(x))
    # embedding shape: (1, N, embedding_size)
    outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
    # shape of outputs: (1, N, hidden_size)

    predictions = self.fc(outputs)
    # shape of predictions: (1, N, length_of_vocab)
    predictions = predictions.squeeze(0)  # add ouput from decoder one step at a time. hence adding is simplified in this shape
    return predictions, hidden, cell
```

```python id="UadE0VBhSCR0"
class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, source, target, teacher_force_ratio=0.5): # sometimes the prediction, sometimes the actual word when training
    batch_size = source.shape[1]
    # source: (trg_len, N)
    target_len = target.shape[0]
    target_vocab_size = len(english.vocab)

    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

    hidden, cell = self.encoder(source)
    # grab start token
    x = target[0]

    for t in range(target_len):
      output, hidden, cell = self.decoder(x, hidden, cell)

      outputs[t] = output
      # output: (N, eng_vocab_size) - argmax along 1st dimension to get the best guess of word perdicted
      best_guess = output.argmax(1)

      x = target[t] if random.random() < teacher_force_ratio else best_guess
    return outputs
```

```python id="gw0PVcH_1g5n"
def load_checkpoint(checkpoint, model, optimizer):
  print(f"Loading checkpoint...")
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])

def save_checkpoint(state, filename='model_checkpoint.pth.tar'):
  print(f"Saving checkpoint: {filename}")
  torch.save(state, filename)
```

```python id="UTIbhaTW1fhS"
def translate_sentence(model, sentence, german, english, max_length, device):
  tokenizer_ger = spacy.load('de')

  if type(sentence) == str:
    tokens = [tok.text.lower() for tok in tokenizer_ger(sentence)]
  else:
    tokens = [tok.lower() for tok in sentence]

  tokens.insert(0, german.init_token)
  tokens.append(german.eos_token)
  token_indices = [german.vocab.stoi[tok] for tok in tokens]

  sentence_tensor = torch.LongTensor(token_indices).unsqueeze(1).to(device)

  with torch.no_grad():
    hidden, cell = model.encoder(sentence_tensor)

  outputs = [german.vocab.stoi['<sos>']]

  for _ in range(max_length):
    previous_word = torch.LongTensor([outputs[-1]]).to(device)

    with torch.no_grad():
      output, hidden, cell = model.decoder(previous_word, hidden, cell)
      best_guess = output.argmax(1).item()

    outputs.append(best_guess)

    if best_guess == english.vocab.stoi['<eos>']:
      break

  translated_sentence = [english.vocab.itos[idx] for idx in outputs]

  return translated_sentence[1:]
  
```

```python id="vryTxg6SatN8" outputId="d8666919-6b92-4b00-fc9b-92e1117d869e" colab={"base_uri": "https://localhost:8080/", "height": 1000}
# Training

# training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# model hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
enc_dropout = 0.5
dec_dropout = 0.5
num_layers = 2

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size = batch_size,
    sort_within_batch=True,
    sort_key = lambda x: len(x.src), # sorts examples with similar length in batch. this saves on compute
    device = device
)

encoder_net = Encoder(input_size=input_size_encoder, embedding_size=encoder_embedding_size, 
                      hidden_size=hidden_size, num_layers=num_layers, p=enc_dropout).to(device)
decoder_net = Decoder(input_size=input_size_decoder, embedding_size=decoder_embedding_size, 
                      hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, p=dec_dropout).to(device)
model = Seq2Seq(encoder=encoder_net, decoder=decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # ignore padding index

if load_model:
  load_checkpoint(torch.load('model_checkpoint.pth.tar'), model, optimizer)

sentence = 'Ein Boot wurde von einem großen Team von Pferden gezogen'

for epoch in range(num_epochs):
  print(f'Epoch {epoch} / {num_epochs}')

  checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
  save_checkpoint(checkpoint)

  model.eval()

  translated_sentence = translate_sentence(model, sentence, german, english, max_length=50, device=device)
  print(f"\nTranslated sentence: {' '.join(translated_sentence)}")

  model.train()

  for batch_idx, batch in enumerate(tqdm(train_iterator)):
    inp_data = batch.src.to(device)
    target = batch.trg.to(device)

    output = model(inp_data, target)
    # output shape: (trg_len, batch_size, output_dim)

    output = output[1:].reshape(-1, output.shape[2]) # keep vocab lengt and combine all other dimensions
    target = target[1:].reshape(-1)

    optimizer.zero_grad()
    loss = criterion(output, target)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    # Plot to tensorboard
    writer.add_scalar("Training loss", loss, global_step=step)
    step += 1
    
```

<!-- #region id="neEHbHOZhO9b" -->
The translation of input sentence: 

'Ein Boot wurde von einem großen Team von Pferden gezogen'
=>
'A boat was pulled by a large team of horses'

The model outputed '' after training of just 20 epochs.

Need to train for more epochs to get better results.

' a a boat is being pulled by a large large . <eos>' has been the best ouput so far
<!-- #endregion -->
