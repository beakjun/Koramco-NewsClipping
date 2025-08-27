from .transformers import TransFormers
import torch

class Embedding(object):
    """
    Corpus를 입력받아 임베딩을 생성하는 객체

    Argument:
        corpus: list[str] corpus
        model_id: str 트랜스포머 모델 ID
        devic: str CPU or GPU
        kwargs
            - cache_dir: str 캐시디렉토리        
            
    Attribute:
        corpus: list[str] corpus
        model_id: str 트랜스포머 모델 ID
        devic: str CPU or GPU
        kwargs: dict kwargs
        tokenizer: 토크나이저
        model: 트랜스포머 모델
        token_ids: Pytorch.tensor 토큰인코딩
        tokens: list[str] 토큰
        model_output: Pytorch.tensor 토큰 임베딩
        sentance_embedding: Pytorch.tensor 문장임베딩

    Example:
        from Koramco.embedding import Embedding
        import torch
        
        corpus = ['공사비 리스크에 컨소시엄 꾸리는 건설사들… 정비사업 조합은 ‘불만', 
                  'GS건설, 인공지능 시대 '데이터센터 개발' 자체사업 도전장']
        model_id = "klue/roberta-base"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = './cache'
        embedding_obj = Embedding(corpus, model_id, device, cache_dir=cache_dir)

        # [출력 예시] embedding_obj.sentance_embedding
        tensor([[ 0.1619, -0.0197,  0.0695,  ..., -0.0315,  0.0289,  0.1385],
                [ 0.2112, -0.4154,  0.0449,  ..., -0.2887,  0.0290,  0.0533]],
               device='cuda:0')
    """
    def __init__(self, corpus:list[str], model_id:str, device:str, **kwargs):
        self.corpus = corpus
        self.model_id = model_id
        self.device = device
        self.kwargs = kwargs
        self._transformers = TransFormers(self.model_id, self.device, **kwargs)
        self.tokenizer, self.model = self._transformers.from_pretrained_atuo_model()
        self.token_ids = self.__token_encoding()
        self.tokens = self.__convert_ids_to_tokens()
        self.model_output = self.__get_model_output()
        self.sentance_embedding = self.__mean_pooling()

    def __token_encoding(self):
        encoding = self.tokenizer(self.corpus, return_tensors="pt", padding=True, truncation=True)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        return encoding

    def __convert_ids_to_tokens(self):
        return [self.tokenizer.convert_ids_to_tokens(_) for _ in self.token_ids['input_ids']]

    def __get_model_output(self):
        with torch.no_grad():
            outputs = self.model(**self.token_ids)
            last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

    def __mean_pooling(self):
        attention_mask = self.token_ids['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(self.model_output.size()).float()
        sum_embeddings = (self.model_output * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled

    def __repr__(self):
        return f"Embedding(model_id: {self.model_id})"
    
    def __str__(self):
        return f"Embedding(model_id: {self.model_id})"