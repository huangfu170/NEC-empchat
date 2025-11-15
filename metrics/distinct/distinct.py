import logging
from typing import List
from collections import defaultdict,Counter
import torch
from nltk import word_tokenize

class Distinct:
    def __init__(self, n):
        self.n = n

    def plot(self,n,n_gram,epoch,counter):
        from matplotlib import pyplot as plt
        mc=counter.most_common(n)
        x=[' '.join(i[0]) for i in mc]
        print(x)
        y=[i[1] for i in mc]
        print(y)
        plt.figure(dpi=300,figsize=(10,4))
        plt.barh(x,y,height=0.4)
        plt.savefig(f'{epoch}-{n_gram}-gram.png',dpi=300)
        
    def compute(self, pred: List[str],epoch=0,plot=False):
        """
        Args:
            input_ids: list of generated text

        Returns: the macro_distinct-k and the micro_distinct-k

        """
        k = self.n
        total_texts_num= len(pred)
        macro_distinct=[0]*k #文档级distinct
        micro_distinct=[0]*k #句子级distinct的平均数
        for ngram in range(k):
            dict_=Counter()
            ngram_total=1e-8
            ngram_distinct_count=0
            for s in pred:
                sen_dict_=Counter()
                s=[ch for ch in s if ch not in [',','.','?','!']]
                for i in range(len(s)-ngram):
                    dict_[tuple(s[i:i+ngram+1])]+=1
                    sen_dict_[tuple(s[i:i+ngram+1])]+=1
                sen_ngram_total,sen_ngram_distinct_count=1e-8,0

                for freq in sen_dict_.values():
                    sen_ngram_total+=freq
                    sen_ngram_distinct_count+=1
                micro_distinct[ngram]+=sen_ngram_distinct_count/sen_ngram_total
            for freq in dict_.values():
                ngram_total+=freq
                ngram_distinct_count+=1
            macro_distinct[ngram]=ngram_distinct_count/ngram_total
            micro_distinct[ngram]=micro_distinct[ngram]/total_texts_num #文档级distinct
            print(f"总{ngram+1}-gram数共:{ngram_total},其中distinct-{ngram+1}-ngram共:{ngram_distinct_count}")
            if plot:
                self.plot(20,ngram,epoch,dict_)
        return {
            "macro-distinct": macro_distinct,
            "micro-distinct": micro_distinct,
        }
