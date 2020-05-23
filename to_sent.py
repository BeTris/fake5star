import pandas as pd
import csv

from nltk.tokenize import sent_tokenize

reader=csv.reader(open('original.csv', 'r'), delimiter= "\t")
data_sent=pd.DataFrame()
for line in reader:
    data_sent=data_sent.append(sent_tokenize(str(line)))
data_sent.to_csv("data_sent.csv")
