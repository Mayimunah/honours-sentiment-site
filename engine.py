"""Inference using numeric parameters recovered from the original saved models."""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin
ROOT=Path(__file__).parent

def load():
    arrays=np.load(ROOT/'models/parameters.npz',allow_pickle=False)
    vocab=json.loads((ROOT/'models/vocabulary.json').read_text())
    vectorizer=TfidfVectorizer(vocabulary=vocab)
    vectorizer.idf_=arrays['idf']
    stops=set((ROOT/'models/stopwords.txt').read_text().splitlines())
    return arrays,vectorizer,stops

def analyze(texts):
    arrays,v,stops=load()
    cleaned=[' '.join(w.lower() for w in re.sub('user','',re.sub(r'[^A-Za-z\s]','',t)).split() if w.lower() not in stops) for t in texts]
    x=v.transform(cleaned)
    valid=np.asarray(x.getnnz(axis=1)>0)
    ap=pairwise_distances_argmin(x@arrays['svd'].T,arrays['ap'])
    km=pairwise_distances_argmin(x,arrays['km'])
    return pd.DataFrame({'Text':texts,'Cleaned text':cleaned,'AP cluster':[str(i) if ok else 'Unassigned' for i,ok in zip(ap,valid)],'K-means cluster':[str(i) if ok else 'Unassigned' for i,ok in zip(km,valid)],'Status':['Assigned' if ok else 'No recognised vocabulary' for ok in valid]})
