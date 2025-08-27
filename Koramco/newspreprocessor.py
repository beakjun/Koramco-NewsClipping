import pandas as pd
import numpy as np
import torch
import re
from sklearn.metrics.pairwise import cosine_similarity
from ._filterWords import filter_words
from Koramco.transformers import TransFormers
from Koramco.embedding import Embedding


class NewsPreprocessor:
    
    class Filtering(object): # 중첩 클래쓰
        
        def __init__(self):
            self.__words = tuple(filter_words)
            self.words = filter_words
            self.add_words_list = []
    
        # def add_words(self, words:str|list[str]):
        #     if type(words) == list:
        #         self.words = self.words + words
        #         self.add_words_list = self.add_words_list + words
        #     elif type(words) == str:
        #         self.words.append(words)
        #         self.add_words_list.append(words)
    
        def update_words(self):
            context = "# 필터 단어 - ['포토', '속보', '특징주', '부고', '인사', '프로필', '영상', '시황', '코스피', '코스닥', '증시', '종목']"
            with open('./filterWords.py', 'w') as f:
                f.write(f'{context}\n\nfilter_words = {str(self.words)}')
    
        def clear_words(self):
            """
            추가한 단어를 초기화하는 메서드
            """
            self.words = list(self.__words)
    
        # def remove_words(self, words:str|list[str]):
        #     if type(words) == str:
        #         words = [str]
        #     for w in self.words:
        #         self.words.remove(w)
        #         self.add_words_list.remove(w)
    
        def create_filter_pat(self) -> str:
            """
            필터 단어를 참고하여 필터링 정규표현식을 생성하는 메서드
            *해당 패턴에 포함되지 않는 데이터로 필터링
            
            Return:
                필터링 정규표현식
                    예: '^\\[포토\\]|^\\[코스닥\\]|^\\[증시\\]|^\\[종목\\]'
    
            Example:
                ft = Filtering()
                ft_pat = ft.creat_filter_pat()
                clean_data = data[~data['Title'].str.contas(ft_pat)]
                
            """
            omit_pat_list = [f'^\[.*{k}.*\]' for k in self.words]
            return '|'.join(omit_pat_list)
        
        def __repr__(self):
            return f"Filtering(words: {self.words})"
        
        def __str__(self):
            return f"Filtering(words: {self.words})"


    
    def __init__(self, model_id: str = "klue/roberta-base", device=None, **kwargs):
        """
        NewsPreprocessor 클래스 초기화.
        모델과 토크나이저를 로드하고, 디바이스를 설정합니다.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.kwargs = kwargs
        # transformer_loader = TransFormers(self.model_id, self.device, **kwargs)
        # self.tokenizer, self.model = transformer_loader.from_pretrained()
        
        
        self.filtering_obj = self.Filtering()
        self.pattern = self.filtering_obj.create_filter_pat()
        self.special_char_map = {
             '\.\.\.':'…',  
            '[①②③④⑤⑥⑦⑧⑨⑩➇]': '',
            '∼': '~',
            'ㆍ': '‧',
            '(\[단독\])':'',
            '(\[더벨\])':''
        }
        
    def _create_embedding_obj(self, corpus: list[str]) -> 'Embedding':
        """
        입력 코퍼스로부터 Embedding 객체를 생성하는 내부 메서드입니다.
        """
        embedding_obj = Embedding(corpus, self.model_id, self.device, **self.kwargs)

        return embedding_obj  

    def filter_words(self, news_df: pd.DataFrame, title_col:str) -> pd.DataFrame:
        """
        뉴스 제목에서 브라켓 안에 있는 특정 키워드를 포함하는 행을 제거합니다.
        
        Notes
        -----
        필터링 패턴은 클래스 내부의 self.pattern 변수를 사용합니다.
        """
        return news_df[~news_df[title_col].str.contains(self.pattern)].reset_index(drop=True)

    def replace_symbols(self, news_df: pd.DataFrame, title_col: str) -> pd.DataFrame:
        news_df_ = news_df.copy()
        
        news_df_["clean_symbol_title"] = news_df_[title_col].replace(self.special_char_map, regex=True)
        
        return news_df_

    def remove_bracket_text_by_similarity(self, news_df: pd.DataFrame, title_col:str, threshold: float=0.6) ->pd.DataFrame:
        """
        뉴스 제목 문자열에서 대괄호([]) 안의 워딩(sub_text)이 본문(body_text)과 유사하지 않을 경우 제거합니다.
        """
        news_df_ = news_df.copy()

        # 특수기호를 기준으로 sub_text와 body_text로 컬럼 나누기
        news_df_[title_col] = news_df_[title_col].str.strip()
        bracket_pattern = r'(?<=\[)[^\[\]]+(?=\])'
        bracket_remove = r'\[[^\[\]]+\]'
        tg_data = news_df_[news_df_[title_col].str.contains(bracket_pattern)].copy().reset_index(drop=True)
        tg_data['sub_text'] = tg_data[title_col].apply(lambda x: ','.join(re.findall(bracket_pattern, x)))
        tg_data['body_text'] = tg_data[title_col].replace(bracket_remove, '', regex=True)
        tg_data = tg_data[tg_data['body_text'] != ''].reset_index(drop=True)
        
         # 클래스 내부 메서드를 사용하여 Embedding 객체 생성
        sub_text = self._create_embedding_obj(tg_data['sub_text'].to_list())
        body_text = self._create_embedding_obj(tg_data['body_text'].to_list())

        # 각 로우별로 body_text와 sub_text간의 유사도 점수 생성
        scores = []
        for i in range(len(tg_data)):
            tmp_score = cosine_similarity(
                sub_text.sentance_embedding.to('cpu').numpy()[i].reshape(1, -1),
                body_text.sentance_embedding.to('cpu').numpy()[i].reshape(1, -1)
            )
            scores.append(tmp_score[0][0])
        tg_data['score'] = scores
        # threshold를 기준으로 subtext제거 기사와 본 기사제목 선정
        tg_data.loc[tg_data['score'] <= threshold, 'fin_text'] = tg_data['body_text']
        tg_data.loc[tg_data['score'] > threshold, 'fin_text'] = tg_data[title_col]

        fin_data = pd.concat([news_df_[~news_df_[title_col].str.contains(bracket_pattern)].assign(fin_text=lambda x: x[title_col]),
                              tg_data.drop(['sub_text', 'body_text', 'score'], axis=1)]).reset_index(drop=True)
        fin_data['clean_title'] = fin_data['fin_text'].str.strip()
        
        return fin_data.drop(columns='fin_text')
    

    
    def clean_title(self, news_df: pd.DataFrame, title_col: str) -> pd.DataFrame:
        """
        뉴스 제목 전처리 파이프라인.
    
        다음과 같은 과정을 수행합니다:
        1. 중복 제거
        2. 특수문자 정리 및 통일
        3. 필터링 키워드 제거
        4. 괄호 안 워딩이 본문과 유사하지 않으면 제거

        Example:
            import torch
            from Koramco.newspreprocessor import NewsPreprocessor
            
            
            model_id = "klue/roberta-base"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cache_dir = './cache'            
            
            news_prep = NewsPreprocessor(model_id, device, cache_dir = cache_dir)
            clean_df = news_prep.clean_title(df,'title')
        """
        news_df_ = news_df.copy()
        news_df_ = news_df_.drop_duplicates(subset=[title_col])
        news_df_ = news_df_.dropna(subset=[title_col])

        # 특수기호 
        news_df_ = self.replace_symbols(news_df_, title_col)
       
        # [] 안 단어 제거
        news_df_ = self.filter_words(news_df_, "clean_symbol_title")

        news_df_ = self.remove_bracket_text_by_similarity(news_df_, 'clean_symbol_title')
        
        return news_df_.drop(columns='clean_symbol_title')
        

