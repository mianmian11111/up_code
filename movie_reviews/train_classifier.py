import json
import os.path
import warnings
import pandas as pd
import torch, gc
import sys
from tqdm import tqdm
from .auxiliary.split_sample import split_sample
from .models_and_data import load_model_tokenizer_dataset_collatefn
from transformers import get_constant_schedule_with_warmup
from .settings import num_training_epochs, base_dir, Config
import numpy
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.manifold import TSNE

def update_loss_avg(new_loss, average):
    if average is None:
        average = new_loss
    else:
        average = 0.9*average + 0.1*new_loss
    return average

def evaluate_model(model, tokenizer, dataset, name="evaluation"):
    # print(f"\nCalculating accuracy on {name.lower()} set:")

    bar = tqdm(desc="Evaluating... Acc: ", total=len(dataset), position=0, leave=True, file=sys.stdout)
    model.eval()
    num_predictions = 0
    num_correct = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            bar.update(1)

            # Split sample into smaller parts if it is too long to be processed at once
            words, num_splits, split_counts = split_sample(tokenizer, sample[0])
            splits_words = [["".join(w["tokens"]) for w in words if j in w["splits"]] for j in range(num_splits)]

            sample_tokenized = tokenizer(splits_words, return_tensors='pt', truncation=False, is_split_into_words=True, padding=True).to("cuda")


            # The overall prediction is the mean of all predictions on the separate parts
            prediction = torch.mean(torch.sigmoid(model(**sample_tokenized,  output_hidden_states=True, output_attentions = True)["logits"]), dim=0)
            prediction_discrete = (prediction.detach().cpu().numpy() > 0.5).astype("int64")

            num_predictions += 1
            num_correct += 1 if prediction_discrete[0] == sample[1] else 0

            bar.desc = f"Evaluating... Acc: {num_correct/num_predictions:<.3f}"


    bar.close()
    acc = num_correct/num_predictions
    print(f"{name} Accuracy: {acc:<.3f}")
    return acc 


def get_hidden_stages_and_attention_score(model, tokenizer, dataset, name="get_hidden_stages_and_attention_score", path = '/home/cjm/PythonProject/MaRC/Explainability-master/movie_reviews/data/hidden_stage_data.csv'):
    # print(f"\nCalculating accuracy on {name.lower()} set:")

    bar = tqdm(desc="get_hidden_stages_and_attention_score: ", total=21, position=0, leave=True, file=sys.stdout)# total=len(dataset)
    model.eval()
    
    all_hidden_states = []
    #attention_scores = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for i in range(21):#len(dataset)
            sample = dataset[i]
            bar.update(1)

            # 将文本分块，使每一块的token数量不大于512
            words, num_splits, split_counts = split_sample(tokenizer, sample[0])# sample[0]是文本，sample[1]是标签

            # 完成分词，"tokens"连接起来，组成单词列表（分块，有num_splits块，预测是取每个块结果的平均）
            splits_words = [["".join(w["tokens"]) for w in words if j in w["splits"]] for j in range(num_splits)]

            sample_tokenized = tokenizer(splits_words, return_tensors='pt', truncation=False, is_split_into_words=True, padding=True).to("cuda")

            # 运行模型，输出每一层的cls和每一层的注意力分数
            with torch.no_grad():
                outputs = model(**sample_tokenized,  output_hidden_states=True)#, output_attentions = True
            # 得到隐藏层张量，每一层的是一个元组，包含（tensor（[[[]]]），device='cuda:0'）
            t_hidden_states = outputs.hidden_states
            #print(t_hidden_states)
            # 将张量转为数组
            hidden_states = []
            # 取出单个隐藏层的张量，转为数组
            for i in t_hidden_states:
                numpy_array = i.detach().cpu().numpy()
                #print(numpy_array)
                hidden_states.append(numpy_array)# 四层【隐藏层，块，token向量】
            # 存入所有句子的hidden_states，句子，13层，块，512个嵌入向量
            all_hidden_states.append(hidden_states)
            # print(type(hidden_states[0]))
            # attention_scores.append(attention_score)
            # 定时清内存
            gc.collect()
            torch.cuda.empty_cache()
        
    bar.close()   
    #将所有隐藏层向量存入hidden_stage_data.csv文件
    #df = pd.DataFrame(all_hidden_states)
    #df.to_csv(path, index=False, header=False)
    return all_hidden_states

    # attention_scores尺寸：(sentences, hidden_layer, batch_size, num_heads, sequence_length, sequence_length)
    # hidden_stages是列表：(sentences, hidden_layer, parts, inputs_token_vectors_and_cls_sep, sequence_length),【包含每个句子（13个隐藏层的张量（的列表【每个块的【所有token的特征向量和【cls】【sep】】】）），（），（）】
    # hidden_stages_and_attention_score = {'all_hidden_states': all_hidden_states, 'attention_scores':attention_scores}
    
# 在隐藏层中提取所有cls向量(每个句子每一层每一块的cls)
def get_cls(all_hidden_states, path = '/home/cjm/PythonProject/MaRC/Explainability-master/movie_reviews/data/cls_data.csv'):
    # 列表存放每段文本的cls
    all_cls_outputs = []

    gc.collect()
    torch.cuda.empty_cache()
    # all_hidden_states中存储的是多个句子的隐藏层，for hidden_state in all_hidden_states取出单个句子的隐藏层
    for hidden_state in all_hidden_states:
        # layer[:, 0, :]表示提取隐藏层每个块的第一个向量，也就是[CLS]
        cls_outputs = [layer[:, 0, :] for layer in hidden_state]
        all_cls_outputs.append(cls_outputs)
        # 定时清内存
        gc.collect()
        torch.cuda.empty_cache()
    #print(all_cls_outputs)
        
    # 将所有cls向量存入cls_data.csv文件
    #df = pd.DataFrame(all_cls_outputs)
    #df.to_csv(path, index=False, header=False)


    # 得到每个句子的每个隐藏层的每个块的cls数组,【句子【隐藏层【块的cls[]】】】】
    # all_cls_outputs_array = numpy.array(all_cls_outputs)
    # torch.save(all_cls_outputs, os.path.join(base_dir, f"saved_models/cls_hidden_stages.txt"))
    # 返回每个句子每一层（每一块）的cls向量列表(sentences, hidden_layer, parts, sequence_length)
    return all_cls_outputs


# 第a条数据每个隐藏层cls的tsne图
def cls_tsne(all_cls_outputs, dataset, a, attacked = False):
    if attacked == True and dataset[a][2] != "Successful":
        w = "Sorry,this data is not be attacked."
        print(w)
        return w
    # 得到每个句子的每个隐藏层的每个块的cls数组,【句子【隐藏层【块的cls[]】】】】
    # all_cls_outputs_array = numpy.array(all_cls_outputs)

    X = []
    for sentence in all_cls_outputs:
        # x列表存储一个句子每一层的cls（假设取第一个块0）【【cls】，【cls】，【】，13个，，】
        x = numpy.array([layer[0] for layer in sentence])
        X.append(x)
    numpy.array(X)
    y = numpy.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
    '''for i in range(3):
        sample = dataset[i]
        y[i] = sample[1]'''

    # '''t-SNE'''
    x = X[a]
    tsne = manifold.TSNE(n_components=2, perplexity=2, init='pca', random_state=520)
    # 对x降维
    X_tsne = tsne.fit_transform(x)
    # x.shape返回X的【行数，列数】,所以x.shape[-1]表示X矩阵的列数，即向量的维度。
    # 输出降维前后的维度
    print("Org data dimension is {}. Embedded data dimension is {}".format(len(x[0]), len(X_tsne[0])))
 
    # '''嵌入空间可视化'''
    # 得到X_tsne二维嵌入向量在每一列的最小值最大值，组成向量
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)# 假如是10和100
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化，（X_tsne-10）/90
    plt.figure(figsize=(8, 8))# 画图大小为8 英寸 x 8 英寸。
    for i in range(len(X_norm)):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
            fontdict={'weight': 'bold', 'size': 9})
    # 隐藏 x 轴和 y 轴的刻度标签
    plt.xticks([])
    plt.yticks([])
    if attacked == True:
        plt.title(f"CLS for data{a} (attacked) ,{dataset[a][3]*100:.2f}% -> {dataset[a][4]*100:.2f}%")
    else :
        plt.title(f"CLS for data{a}")
    plt.show()
    return plt


def train():
    model, tokenizer, dataset, train_dataloader = load_model_tokenizer_dataset_collatefn(load_weights=False, parallel=True)

    print("Start training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_dataloader.dataset.set_split("train")
    dataset.set_split("val")
    max_acc = evaluate_model(model, tokenizer, dataset)

    evals_without_improvement = -3
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        for epoch in range(num_training_epochs):
            print(f"\n\nEpoch {epoch}:")
            model.train()
            loss_avg = None
            bar = tqdm(desc="Loss: None", total=len(train_dataloader), position=0, leave=True, file=sys.stdout)

            for idx, batch in enumerate(train_dataloader):
                bar.update(1)
                prediction = model(**batch)

                loss = loss_fn(prediction["logits"], batch["labels"])
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loss_avg = update_loss_avg(loss, loss_avg).detach().cpu().numpy()
                bar.desc = f"Loss: {loss_avg:<.3f}"

            bar.close()
            print()

            acc = evaluate_model(model, tokenizer, dataset)

            evals_without_improvement += 1
            if acc >= max_acc:
                print("Model saved.")
                torch.save(model.state_dict(), os.path.join(base_dir, f"saved_models/clf_{Config.save_name}.pkl"))
                max_acc = acc
                evals_without_improvement = (min(evals_without_improvement, 0))

            if evals_without_improvement == 4:
                break