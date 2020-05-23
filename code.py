import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import transformers as ppb
import torch
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("https://raw.githubusercontent.com/BeTris/fake5star/f7c9048a7750897b2db4c85653886707be64cfd0/articles(1).csv",delimiter='\t',header=None)
print(data)
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
