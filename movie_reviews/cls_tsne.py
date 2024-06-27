import pandas as pd
from movie_reviews.attack_dataset import get_attacked_data
from movie_reviews.train_classifier import cls_tsne, get_cls, get_hidden_stages_and_attention_score
from movie_reviews.attack_dataset import get_dataset_text_lable,  get_attacked_data




def tsne_for_cls(number_of_data, model, tokenizer, dataset,attack = False):
    # 得到text和lable列表
        # get_dataset_text_lable()
        
        if attack == False:
            # 原始数据每条原始文本的隐藏层，存入hidden_stage_data.csv文件
            all_hidden_states = get_hidden_stages_and_attention_score(model, tokenizer, dataset,'/home/cjm/PythonProject/MaRC/Explainability-master/movie_reviews/data/hidden_stage_data.csv')
            # 原始数据攻击前的cls,存入cls_data.cls文件
            all_cls = get_cls(all_hidden_states, path = '/home/cjm/PythonProject/MaRC/Explainability-master/movie_reviews/data/cls_data.csv')
            # 原始数据用cls画tsne图（第三个参数代表第n条数据）
            cls_lable_tsne = cls_tsne(all_cls, dataset, number_of_data, attacked = False)
        else:
            # 得到攻击后的数据，存入attacked_result.csv文件
            attacked_data = get_attacked_data()
            # 攻击后文本的隐藏层，存入attacked_hidden_stage.csv, attacted_data目前只存放了100条数据的被攻击文本
            attacked_hidden_states = get_hidden_stages_and_attention_score(model, tokenizer, attacked_data, path = '/home/cjm/PythonProject/MaRC/Explainability-master/movie_reviews/data/attacked_hidden_stage.csv')
            # 攻击后的cls,存入attacked_cls.cls文件
            attacked_cls = get_cls(attacked_hidden_states, path = '/home/cjm/PythonProject/MaRC/Explainability-master/movie_reviews/data/attacked_cls.csv')
            # 攻击后的数据用cls画tsne图（第三个参数代表第n条数据）
            cls_lable_tsne = cls_tsne(attacked_cls, attacked_data, number_of_data, attacked = True)

        
        
        
        # 得到原始数据的cls
        #df = pd.read_csv('movie_reviews/data/cls_data.csv', header=None)
        #cls_data = df.values.tolist()
        #cls_lable_tsne = cls_tsne(cls_data, dataset, number_of_data, attacked = False)
        #print(type(cls_data[0][0]))
        # 得到攻击之后数据的cls
        #df = pd.read_csv('movie_reviews/data/cls_data.csv', header=None)
        #attacked_cls_data = df.values.tolist()

        # 得到每条文本每一层的注意力分数s
        # attention_scores = output["attention_scores"]      

    # attention_scores的形状（sentence_size, layer, batch_size, num_heads, sequence_length, sequence_length）
    # 每一层注意力分数的形状为 (batch_size, num_heads, sequence_length, sequence_length)
    # print(f"每一层注意力分数的形状: {attention_scores[0][0].shape}")

    
    # print(all_hidden_states)
    # print("all_cls_numpy:", all_cls)
    # print("accuracy = ", accuracy)  
    # print("attention_score:", attention_scores)  