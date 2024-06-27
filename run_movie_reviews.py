import os
import sys
import numpy as np
import warnings
from movie_reviews.cls_tsne import tsne_for_cls
from movie_reviews.models_and_data import load_model_tokenizer_dataset_collatefn
import matplotlib.pyplot as plt
from sklearn import manifold, datasets



def run_experiment():

    model, tokenizer, dataset, train_dataloader = load_model_tokenizer_dataset_collatefn(load_weights=True, parallel=True)
    train_dataloader.dataset.set_split("test") 
    model.eval()
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # 画出n条数据攻击前后cls的tsne图(两张图)
        tsne_for_cls(10, model, tokenizer, dataset, attack = True)
        
if __name__ == '__main__':
    run_experiment()