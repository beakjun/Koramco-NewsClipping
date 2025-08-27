from typing import List
from transformers import PreTrainedTokenizer
import re
import math
import pandas as pd
import numpy as np
from collections import defaultdict




class BM25(object):
    """
    Corpus를 입력받아 토크나이저를 통해 토큰으로 분해하고 쿼리를 입력받아 BM25를 계산하는 객체

    Argument:
        corpus: list[str] corpus
        tokenizer: PretrainedTokenizer 사전학습모델의 토크나이저

    Example:
        from Koramco.transformers import TransFormers
        from Koramco.bm25 import BM25
            
        model_id = "klue/roberta-base"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = './cache'            
        
        tf = TransFormers(model_id, device, cache_dir = cache_dir)
        tokenizer, model = tf.from_pretrained_atuo_model()
        bm = BM25(corpus1,tokenizer)

        bm25_score = bm.get_score_mat(corpus2)
    """
    def __init__(self, corpus:List[List[str]], tokenizer:PreTrainedTokenizer):
        self.corpus = corpus
        self.tokenizer = tokenizer
        raw_ids = self.tokenizer(corpus, add_special_tokens=False)['input_ids']
        self.tokenized_corpus = [self._clean_ids(ids) for ids in raw_ids]
        self.n_docs = len(self.tokenized_corpus)
        self.avg_doc_lens = sum(len(lst) for lst in self.tokenized_corpus) / len(self.tokenized_corpus)
        self.idf = self._calculate_idf()
        self.term_freqs = self._calculate_term_freqs()

        self.preferred_source = [
                '뉴스1', '더벨', '인베스트조선','딜사이트',  '매일경제',  '조선비즈',
                '이데일리', '한국경제', '서울경제', '연합뉴스', '머니투데이'
                ]

    def _clean_ids(self, ids):
        toks = self.tokenizer.convert_ids_to_tokens(ids)
        keep = []
        for t in toks:
            # byte-level/wordpiece 접두부 정리
            t0 = t.replace('##', '').replace('▁', '').replace('Ġ', '')
          
            if not re.search(r'[A-Za-z가-힣0-9]', t0):
                # 한글/영문/숫자 중 하나도 없으면(순수 기호) 드롭
                continue
            if re.fullmatch(r'[0-9]+', t0):
                # 숫자만 있는 토큰은 드롭(원하면 <NUM> 치환으로 바꿔도 됨)
                continue
            keep.append(t)
        # 다시 id로 변환
        return self.tokenizer.convert_tokens_to_ids(keep)

    
    def _calculate_idf(self):
        idf = defaultdict(float)
        for doc in self.tokenized_corpus:
          for token_id in set(doc):
            idf[token_id] += 1
        for token_id, doc_frequency in idf.items():
          idf[token_id] = math.log(((self.n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)) + 1)
        return idf
        
    def _calculate_term_freqs(self):
        term_freqs = [defaultdict(int) for _ in range(self.n_docs)]
        for i, doc in enumerate(self.tokenized_corpus):
          for token_id in doc:
            term_freqs[i][token_id] += 1
        return term_freqs


    def get_scores(self, query: str, k1: float = 1.2, b: float = 0.75):
        # 1) 토크나이즈
        q_ids = self.tokenizer([query], add_special_tokens=False)['input_ids'][0]
        # 2) 클린업 (기호/숫자/짧은 토큰 제거 등)
        q_ids = self._clean_ids(q_ids)  # ← clean_id 메서드명이 다르면 여길 바꿔 쓰세요.
    
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not q_ids:
            return scores
    
        avg_len = self.avg_doc_lens or 1.0
    
        for q in q_ids:
            idf = self.idf.get(q, 0.0)
            if idf <= 0:
                continue
            for i, term_freq in enumerate(self.term_freqs):
                tf = term_freq.get(q, 0)
                if tf == 0:
                    continue
                doc_len = len(self.tokenized_corpus[i]) or 1
                denom = tf + k1 * (1 - b + b * (doc_len / avg_len))
                scores[i] += idf * (tf * (k1 + 1)) / denom
    
        return scores

    def get_score_mat(self, query:list[str], k1:float = 1.2, b:float=0.75):
        result = []
        for i, t in enumerate(query):
          tmp_df = pd.DataFrame(self.get_scores(t, k1, b), columns=['BM25'])
          tmp_df['corpus2_idx'] = i
          tmp_df['corpus1_idx'] = range(len(tmp_df))
          result.append(tmp_df)
        df = pd.concat(result).reset_index(drop=True)
        df['corpus1'] = df['corpus1_idx'].apply(lambda x: self.corpus[x])
        df['corpus2'] = df['corpus2_idx'].apply(lambda x: query[x])
        df['score'] = df['BM25'] / df.groupby('corpus1_idx', as_index=False)['BM25'].transform(max)
        return df[['corpus1_idx', 'corpus2_idx', 'corpus1', 'corpus2', 'score']].sort_values(['corpus1_idx','corpus2_idx'])
