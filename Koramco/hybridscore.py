import pandas as pd
import numpy as np
import re
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from Koramco.bm25 import BM25
from Koramco.embedding import Embedding
from Koramco._voca import add_tokens

import networkx as nx
from networkx.algorithms import community


class HybridScore:
    """
    (임베딩 목적) 입력받은 model_id에 해당되는 tokenizer와 분류 model을 token을 추가하여 반환하는 메서드
    
    Returns:
        bm25_matrix_df, similarity_matrix_df

     Attribute:
        model_id: str 트랜스포머 모델 ID
        device (str): CPU or GPU
        cache_dir: str 캐쉬 디렉토리
        kwargs (str): Keywords Arguments
        
    Example:
        from Koramco.transformers import TransFormers
        import torch
        
        model_id = "klue/roberta-base"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = './cache'            
        
        hs = HybridScore(corpus, model_id, device, cache_dir=cache_dir)
        bm25_df = hs.get_bm25_mat(corpus)
        sim_df = hs.get_similarity_mat(corpus)
        merge_df = hs.get_cluster_tg_data(sim_df, bm25_df) 
    """
    
    def __init__(self, corpus, model_id, device=None, **kwargs):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs

        self.__add_tokens = tuple(add_tokens)

        self._tokenizer = None
        
        tok = self.build_tokenizer_only()

        self.bm25 = BM25(corpus, self._tokenizer)
        self.embedding_obj = Embedding(corpus, self.model_id, self.device, **self.kwargs)
        

    def build_tokenizer_only(self):
        if self._tokenizer is not None:
            return self._tokenizer
        tok = AutoTokenizer.from_pretrained(self.model_id)
        if self.__add_tokens:
            vocab = tok.get_vocab()
            new_tokens = [t for t in self.__add_tokens if t not in vocab]
            if new_tokens:
                tok.add_tokens(new_tokens)
        self._tokenizer = tok
        return tok



    def get_bm25_mat(self, query:list[str], k1:float = 1.2, b:float=0.75):
        if self.bm25 is None:
            raise RuntimeError("BM25가 없습니다.")

        return prepare_mat(self.bm25.get_score_mat(query, k1=k1, b=b))
        
    def get_similarity_mat(self, query:list[str]):
        embe_array1 = self.embedding_obj.sentance_embedding.to('cpu').numpy()
        
        embedding_query = Embedding(query, self.model_id, self.device, **self.kwargs)
        embe_array2 = embedding_query.sentance_embedding.to('cpu').numpy()
        
        sim_mat = cosine_similarity(embe_array1, embe_array2)
        sim_df = pd.DataFrame(sim_mat).reset_index()
        sim_df = sim_df.rename(columns={'index':'corpus1_idx'})
        sim_df = pd.melt(sim_df, id_vars='corpus1_idx', value_name='similarity', var_name='corpus2_idx')
        sim_df.insert(2, 'corpus1', sim_df['corpus1_idx'].apply(lambda x: self.embedding_obj.corpus[x]))
        sim_df.insert(3, 'corpus2', sim_df['corpus2_idx'].apply(lambda x: embedding_query.corpus[x]))    
        return prepare_mat(sim_df.sort_values(['corpus1_idx', 'corpus2_idx']).reset_index(drop=True))

"""
Similarity와 BM25를 활용한 하이브리드 점수 후처리 모듈
"""

def get_cluster_tg_data(sim_mat:pd.DataFrame, bm25_mat:pd.DataFrame, sim_threshold:float=0.9, score_threshold:float=0.8)-> pd.DataFrame :    
    sim_mat_ = sim_mat.copy()
    sim_mat_ = sim_mat_[(sim_mat_['corpus1_idx']!=sim_mat_['corpus2_idx'])&(sim_mat_['similarity'] >= sim_threshold)]
    sim_mat_ = sim_mat_.rename(columns={'similarity':'fin_score'})
    
    
    score_mat_ = bm25_mat.copy()
    score_mat_ = score_mat_[(score_mat_['corpus1_idx']!=score_mat_['corpus2_idx'])&(score_mat_['score'] >= score_threshold)]
    score_mat_ = score_mat_.rename(columns={'score':'fin_score'})
    
    
    total_mat = pd.concat([sim_mat_, score_mat_])
    total_mat['idx_set'] = total_mat.apply(lambda x: f'{x.corpus1_idx}, {x.corpus2_idx}' if x.corpus1_idx < x.corpus2_idx else f'{x.corpus2_idx}, {x.corpus1_idx}', axis=1)
    total_mat = total_mat.drop_duplicates(subset='idx_set').reset_index(drop=True)
    
    return total_mat[['corpus1_idx', 'corpus2_idx', 'corpus1', 'corpus2', 'fin_score']]

def detect_communities(df: pd.DataFrame, source_col: str = 'corpus1_idx', target_col: str = 'corpus2_idx', weight_col: str = 'fin_score') -> pd.DataFrame:
    """
    하이브리드 스코어 기반 네트워크 그래프 생성 및 커뮤니티 탐지 함수
    Parameters:
    - df: 입력 DataFrame
    - source_col: source 노드 컬럼명
    - target_col: target 노드 컬럼명
    - weight_col: 엣지 weight 컬럼명
    - threshold: 엣지로 포함할 최소 하이브리드 스코어
    Returns:
    - node_df: 각 노드와 클러스터 ID가 포함된 DataFrame
    """
    
    links_df = df[[source_col, target_col, weight_col]].reset_index(drop=True)
    links_df.columns = ['source', 'target', 'weight']
    
    G = nx.from_pandas_edgelist(links_df, 'source', 'target', 'weight')
    
    communities = community.greedy_modularity_communities(G)
    
    result = []
    
    for i, v in enumerate(communities):
        tmp_df = pd.DataFrame([(list(v), i)], columns=['corpus_idx', 'cluster'])
        result.append(tmp_df)
    node_df = pd.concat(result).reset_index(drop=True)
    node_df = node_df.explode(column='corpus_idx').reset_index(drop=True)
    
    return node_df


def prepare_mat(df, col_idx1='corpus1_idx', col_idx2='corpus2_idx'):
    df_filtered = df[df[col_idx1] != df[col_idx2]].copy()
    df_filtered['idx_set'] = df_filtered.apply(
        lambda x: f'{min(x[col_idx1], x[col_idx2])}, {max(x[col_idx1], x[col_idx2])}', axis=1)
    df_filtered = df_filtered.drop_duplicates(subset='idx_set').reset_index(drop=True)
    return df_filtered


PREFERRED_SOURCES_DEFAULT = [
    '뉴스1','더벨','인베스트조선','딜사이트','매일경제','조선비즈',
    '이데일리','한국경제','서울경제','연합뉴스','머니투데이'
]

def add_preferred_weight(df: pd.DataFrame,
                         source_col='source',
                         out_col='pref_source',
                         preferred= PREFERRED_SOURCES_DEFAULT) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = df[source_col].isin(set(preferred)).astype(int)
    return df

def add_rrf(df, simrank_col_nm, bm25rank_col_nm, sim_col_nm, k=60, alpha=0.5):
    df_copy = df.copy()
    df_copy['rrf_score'] = (1.0 / (k + df_copy[simrank_col_nm]) + 1.0 / (k + df_copy[bm25rank_col_nm])) * alpha * df_copy[sim_col_nm]
    return df_copy


def get_top_k_rrf_results(df: pd.DataFrame, top_k: int = 5,
                          group_cols: list = ['corpus1_idx', 'corpus1'],
                          rank_col: str = 'rrf_score') -> pd.DataFrame:

    # RRF 점수를 기준으로 그룹별 순위(rank)를 매김
    df['rrf_rank'] = df.groupby(group_cols)[rank_col].rank(method='dense', ascending=False)
    
    # 상위 K개 결과만 필터링
    top_k_df = df.query(f'rrf_rank <= {top_k}')
    
    # 상위 K개 결과의 평균 점수를 계산
    agg_df = top_k_df.groupby(group_cols, as_index=False).agg({rank_col: 'mean'})
    
    # RRF 점수 내림차순으로 정렬
    result_df = agg_df.sort_values(rank_col, ascending=False).reset_index(drop=True)
    
    return result_df


def get_new_label(title:str, add_tokens:list= add_tokens) -> int:
    tokens = [t for t in add_tokens if '코람코' not in t]
    trust = '(' + ')|('.join([t for t in tokens if re.match('(.*신탁)', t)]) + ')'
    retis = '(' + ')|('.join([t for t in add_tokens if re.match('([^메]+리츠)', t)]) + ')|([^메]리츠)'
    investments = '(' + ')|('.join([t for t in tokens if re.match('(.*운용)', t)]) + ')|(마스턴운용)' 
    if re.search('코람코', title):
        return 1
    elif re.search(trust, title):
        return 1
    elif re.search(retis+'|'+investments, title):
        return 1
    else: 
        return 0

def get_final_score(rrf:pd.Series, keyword:pd.Series, g: float=3):
    rrf_mean, rrf_std = rrf.mean(), rrf.std()
    rrf_z = (rrf - rrf_mean) / rrf_std
    rrf_fin = rrf_z + 3 * keyword 
    return rrf_fin

def order_keep_all_mark_20_reps(
    df: pd.DataFrame,
    top_n_reps: int = 20,
    score_col: str = "final_score",
    cluster_col: str = "cluster",
    tiebreak_cols: tuple = ("pref_source", "rrf_score"),
    rename_map: dict | None = None
) -> pd.DataFrame:
    """
    1) 전역 점수 내림차순으로 훑는다(동점은 tiebreak_cols 내림차순).
    2) 클러스터를 처음 만나면: 클러스터 대표(해당 클러스터 내 최고 점수) + 같은 클러스터 나머지(블록) 이어붙임.
       - 대표는 클러스터 내 '점수 우선'으로 선발. (선호언론사로 대표를 바꾸지 않음)
       - 블록의 '나머지'는 선호언론사 → 점수 → 기타 tiebreak 순으로 정렬.
    3) 클러스터가 없는 기사(NaN)는 단독으로 붙이며, 대표 카운트에 포함 가능.
    4) 전체 행은 모두 남김. 단, 'check=True'는 처음 만나는 대표들(클러스터 대표 또는 단독 기사) 중 상위 top_n_reps개만 True.
    """
    if df.empty:
        return df.copy()

    tmp = df.copy()

    # tiebreak 결측 컬럼 보정
    for c in tiebreak_cols:
        if c not in tmp.columns:
            tmp[c] = 0

    # ---- 대표 선발용: (score desc, tiebreak desc) ----
    sort_in_cols = [score_col] + list(tiebreak_cols)
    ascending_in  = [False] + [False] * len(tiebreak_cols)

    clustered = tmp[tmp[cluster_col].notna()].copy()

    # 클러스터별 블록 구성:
    #  - 대표: score 우선(동점 tiebreak)
    #  - 나머지: pref_source -> score -> (기타 tiebreak) 내림차순
    cluster_blocks: dict = {}
    if not clustered.empty:
        for cid, g in clustered.groupby(cluster_col, sort=False):
            g_rep_sorted = g.sort_values(sort_in_cols, ascending=ascending_in)
            rep_idx = g_rep_sorted.index[0]  # 대표는 '점수 우선'을 보장

            rest = g.drop(index=rep_idx)
            # 나머지 정렬 키 구성: pref_source가 있으면 최우선
            pref_key = "pref_source" if "pref_source" in rest.columns else None
            rest_sort_cols = ([pref_key] if pref_key else []) + [score_col] + [c for c in tiebreak_cols if c != pref_key]
            rest_ascending = [False] * len(rest_sort_cols)  # 전부 내림차순

            if not rest.empty:
                rest_sorted = rest.sort_values(rest_sort_cols, ascending=rest_ascending)
                block_idx = [rep_idx] + rest_sorted.index.tolist()
            else:
                block_idx = [rep_idx]

            cluster_blocks[cid] = block_idx

    # ---- 전역 정렬(점수 desc, tiebreak desc) ----
    global_sorted_idx = tmp.sort_values(sort_in_cols, ascending=ascending_in).index.tolist()

    emitted: list[int] = []
    seen_clusters: set = set()
    rep_count = 0
    tmp["check"] = False

    for idx in global_sorted_idx:
        if idx in emitted:
            continue

        row = tmp.loc[idx]
        cid = row[cluster_col]

        if pd.isna(cid):
            # 단독 기사
            emitted.append(idx)
            if rep_count < top_n_reps:
                tmp.at[idx, "check"] = True
                rep_count += 1
        else:
            if cid not in seen_clusters:
                seen_clusters.add(cid)
                block = cluster_blocks.get(cid, [])
                if not block:
                    continue
                emitted.extend(block)
                rep_idx = block[0]  # 클러스터 대표
                if rep_count < top_n_reps:
                    tmp.at[rep_idx, "check"] = True
                    rep_count += 1
            # 이미 본 클러스터면 스킵(블록으로 이미 들어감)

    ordered = tmp.loc[emitted].copy()
    ordered["display_rank"] = range(1, len(ordered) + 1)

    # 출력 컬럼 (실제 컬럼명을 씀)
    base_cols = [
        "date", "source", "clean_title", "link", "category_pred",
        cluster_col, "check", "pref_source", score_col, "display_rank"
    ]
    exist_cols = [c for c in base_cols if c in ordered.columns]
    out = ordered[exist_cols].reset_index(drop=True)

    # 컬럼 한글명 매핑(기본)
    if rename_map is None:
        rename_map = {
            "date": "날짜",
            "source": "언론사",
            "category_pred": "카테고리",
            "clean_title": "기사제목",
            "link": "링크",
            cluster_col: "클러스터",
            "check": "대표기사여부",
            "pref_source": "선호언론사여부",
            score_col: "스코어",
            "display_rank": "표시순위",
        }
    out = out.rename(columns=rename_map)

    return out