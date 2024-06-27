import os.path
import warnings
import torch
import numpy
import sys
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.manifold import TSNE
from tqdm import tqdm
from movie_reviews.auxiliary.split_sample import split_sample
from .models_and_data import load_model_tokenizer_dataset_collatefn
from .settings import num_training_epochs, base_dir, Config
    
def tsne(all_hidden_states, dataset):
    all_cls_outputs = []
    # layer[:, 0, :]表示提取每个隐藏层输出的第一个token(也就是[CLS]token)的特征向量。
    # all_hidden_states中存储的是多个句子的隐藏层，for hidden_state in all_hidden_states取出单个句子的隐藏层
    for hidden_state in all_hidden_states:
        # cls_outputs列表【存储的是每个隐藏层（，）中每一块【的cls向量【】】】
        cls_outputs = [layer[:, 0, :] for layer in hidden_state]# [:, 0, :]表示13个隐藏层中每个隐藏层的每一块的第一个向量【CLS】
        # 将cls张量转为向量，维度是(隐藏层数13,块数n, 输出维度768)
        cls_vector = [cls_output.detach().cpu().numpy() for cls_output in cls_outputs]
        all_cls_outputs.append(cls_outputs)# 列表【每句话（，）的每层【中每块【的【cls

    # 得到每个句子的每个隐藏层的每个块的cls数组,【句子【隐藏层【块的cls[]】】】】
    all_cls_outputs_array = numpy.array(all_cls_outputs)

    X = numpy.array([])
    for sentence in range(len(all_cls_outputs_array)):
        # x存储一个句子每一层的cls（假设取第一个块）【【cls】，【cls】，【】，13个，，】
        x = numpy.array([layer[1] for layer in sentence])
        numpy.append(X, [x])
    y = numpy.array(dataset[:15, 1])

    # '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, perplexity=2, init='pca', random_state=520)
    for x in X:
        X_tsne = tsne.fit_transform(x)
        # X.shape返回X的【行数，列数】,所以X.shape[-1]表示X矩阵的列数，即向量的维度。
        # 输出降维前后的维度
        print("Org data dimension is {}. Embedded data dimension is {}".format(x.shape[-1], X_tsne.shape[-1]))
 
        # '''嵌入空间可视化'''
        # 得到X_tsne二维嵌入向量在每一列的最小值最大值，组成向量
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)# 假如是10和100
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化，（X_tsne-10）/90
        plt.figure(figsize=(8, 8))# 画图大小为8 英寸 x 8 英寸。
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                fontdict={'weight': 'bold', 'size': 9})
        # 隐藏 x 轴和 y 轴的刻度标签
        plt.xticks([])
        plt.yticks([])
        plt.show()
        return plt
    

def get_hidden_stages(model, tokenizer, dataset, name="get_hidden_stages"):
    # print(f"\nCalculating accuracy on {name.lower()} set:")

    bar = tqdm(desc="Evaluating... Acc: ", total=15, position=0, leave=True, file=sys.stdout)# total=len(dataset)
    model.eval()
    
    all_hidden_states = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for i in range(15):#len(dataset)
            sample = dataset[i]
            bar.update(1)

            # Split sample into smaller parts if it is too long to be processed at once
            words, num_splits, split_counts = split_sample(tokenizer, sample[0])
            # sample=(0, num_splits)之间的随机整数
            # sample = 0
            # 完成分词，"tokens"连接起来，组成列表
            # splits_words = ["".join(w["tokens"]) for w in words if sample in w["splits"]]
            splits_words = [["".join(w["tokens"]) for w in words if j in w["splits"]] for j in range(num_splits)]

            sample_tokenized = tokenizer(splits_words, return_tensors='pt', truncation=False, is_split_into_words=True, padding=True).to("cuda")

            # 输出隐藏层的cls
            outputs = model(**sample_tokenized,  output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # print(hidden_states)
            all_hidden_states.append(hidden_states)

    bar.close()
    # 输出是列表【包含每个句子（13个隐藏层的张量（的列表【块【向量【768维】】】）），（），（）】
    return all_hidden_states

def get_cls_np(all_hidden_states):
    all_cls_outputs = []
    # layer[:, 0, :]表示提取每个隐藏层输出的第一个token(也就是[CLS]token)的特征向量。
    # all_hidden_states中存储的是多个句子的隐藏层，for hidden_state in all_hidden_states取出单个句子的隐藏层
    for hidden_state in all_hidden_states:
        # cls_outputs列表【存储的是每个隐藏层（，）中每一块【的cls向量【】】】
        cls_outputs = [layer[:, 0, :] for layer in hidden_state]# [:, 0, :]表示13个隐藏层中每个隐藏层的每一块的第一个向量【CLS】
        # 将cls张量转为向量，维度是(隐藏层数13,块数n, 输出维度768)
        cls_vector = [cls_output.detach().cpu().numpy() for cls_output in cls_outputs]
        all_cls_outputs.append(cls_vector)# 列表【每句话（，）的每层【中每块【的【cls

    # 得到每个句子的每个隐藏层的每个块的cls数组,【句子【隐藏层【块的cls[]】】】】
    all_cls_outputs_array = numpy.array(all_cls_outputs)

    return all_cls_outputs_array
