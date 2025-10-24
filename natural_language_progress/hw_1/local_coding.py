import os
import requests
import zipfile
import nltk
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook

from sklearn.manifold import TSNE
from sklearn.preprocessing import scale


def setting():
    # 下载数据集
    print("下载数据集...")
    url = "https://drive.google.com/uc?export=download&id=1ERtxpdWOgGQ3HOigqAMHTJjmOE_tWvoF"
    response = requests.get(url)
    with open("quora.zip", "wb") as f:
        f.write(response.content)
    
    
    # 下载 NLTK 数据
    print("下载 NLTK 数据...")
    import nltk
    nltk.download('punkt_tab')
    nltk.download('punkt')
    
    # 设置随机种子
    np.random.seed(42)
    
    # 后续处理代码...
    # 在这里添加您的作业处理代码
    
    print("处理完成！")

def load_data():
    quora_data = pd.read_csv('train.csv')

    quora_data.question1 = quora_data.question1.replace(np.nan, '', regex=True)
    quora_data.question2 = quora_data.question2.replace(np.nan, '', regex=True)

    texts = list(pd.concat([quora_data.question1, quora_data.question2]).unique())
    texts = texts[:50000] # Accelerated operation
    print(len(texts))

    tokenized_texts = [word_tokenize(text.lower()) for text in tqdm(texts)]

    assert len(tokenized_texts) == len(texts)
    assert isinstance(tokenized_texts[0], list)
    assert isinstance(tokenized_texts[0][0], str)
    return quora_data, tokenized_texts

def build_vocab(tokenized_texts):
    from collections import Counter

    MIN_COUNT = 5

    words_counter = Counter(token for tokens in tokenized_texts for token in tokens)
    word2index = {
        '<unk>': 0
    }

    for word, count in words_counter.most_common():
        if count < MIN_COUNT: # 频次过低的词不加入词典
            break

        word2index[word] = len(word2index)

    index2word = [word for word, _ in sorted(word2index.items(), key=lambda x: x[1])]

    print('Vocabulary size:', len(word2index))
    print('Tokens count:', sum(len(tokens) for tokens in tokenized_texts))
    print('Unknown tokens appeared:', sum(1 for tokens in tokenized_texts for token in tokens if token not in word2index))
    print('Most freq words:', index2word[1:21])
    return word2index, index2word

def build_contexts(tokenized_texts, window_size):
    contexts = []
    for tokens in tokenized_texts:
        for i in range(len(tokens)):
            central_word = tokens[i]
            context = [tokens[i + delta] for delta in range(-window_size, window_size + 1)
                       if delta != 0 and i + delta >= 0 and i + delta < len(tokens)]

            contexts.append((central_word, context))

    return contexts

def make_cbow_batches_iter(contexts, window_size, batch_size):
    # 过滤有效样本
    valid_contexts = [
        (word, context) for word, context in contexts 
        if len(context) == 2 * window_size and word != 0
    ]
    
    central_words = np.array([word for word, context in valid_contexts])
    context_words = np.array([context for word, context in valid_contexts])
    
    batches_count = int(math.ceil(len(context_words) / batch_size))
    print(f'Initializing batches generator with {batches_count} batches per epoch')
    
    indices = np.arange(len(context_words))
    np.random.shuffle(indices)
    
    for i in range(batches_count):
        batch_begin = i * batch_size
        batch_end = min((i + 1) * batch_size, len(context_words))
        batch_indices = indices[batch_begin:batch_end]
        
        # 获取当前批次的上下文和中心词
        batch_context = context_words[batch_indices]
        batch_central = central_words[batch_indices] 
        batch_context_tensor = torch.LongTensor(batch_context)
        batch_central_tensor = torch.LongTensor(batch_central)
        yield batch_context_tensor, batch_central_tensor

 

def trainer():
    # Here are the hyperparameters you can adjust
    embedding_dim = 32
    learning_rate = 0.001
    epoch_num = 4
    batch_size = 128
    batched_losses = []
    # Initialization Model
    model = CBoWModel(len(word2index),embedding_dim)
    # Getting model to GPU
    model.cuda()
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_every_nsteps = 3000
    total_loss = 0
    start_time = time.time()
    global_step = 0

    for ep in range(epoch_num):
        for step, batch in enumerate(make_cbow_batches_iter(contexts, window_size=2, batch_size=batch_size)):
            global_step += 1
            output = model(batch[0].cuda())
            loss = criterion(output, batch[1].cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batched_losses.append(loss.item())
            if global_step != 0 and global_step % loss_every_nsteps == 0:
                print("Epoch = {}, Step = {}, Avg Loss = {:.4f}, Time = {:.2f}s".format(ep, step, total_loss / loss_every_nsteps,
                                                                        time.time() - start_time))
                total_loss = 0
                start_time = time.time()
    return model, losses



def most_similar(embeddings, index2word, word2index, word):
    word_emb = embeddings[word2index[word]]

    similarities = cosine_similarity([word_emb], embeddings)[0]
    top10 = np.argsort(similarities)[-10:]

    return [index2word[index] for index in reversed(top10)]

def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    output_notebook()

    if isinstance(color, str):
        color = [color] * len(x)
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show:
        pl.show(fig)
    return fig


def get_tsne_projection(word_vectors):
    tsne = TSNE(n_components=2, verbose=1)
    return scale(tsne.fit_transform(word_vectors))


def visualize_embeddings(embeddings, index2word, word_count):
    word_vectors = embeddings[1: word_count + 1]
    words = index2word[1: word_count + 1]

    word_tsne = get_tsne_projection(word_vectors)
    draw_vectors(word_tsne[:, 0], word_tsne[:, 1], color='blue', token=words)

if __name__ == "__main__":
    # setting()
    quora_data, tokenized_texts = load_data()
    word2index, index2word = build_vocab(tokenized_texts)
    contexts = build_contexts(tokenized_texts, window_size=2)
    contexts = [(word2index.get(central_word, 0), [word2index.get(word, 0) for word in context])
                for central_word, context in contexts]
    # model = CBoWModel(vocab_size=len(word2index), embedding_dim=32).cuda()

    model, losses = trainer()
    embeddings = model.embeddings.weight.data.cpu().numpy()
    Top10_similarity = most_similar(embeddings, index2word, word2index, 'my')
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('CBOW Training Loss')
    plt.show()

