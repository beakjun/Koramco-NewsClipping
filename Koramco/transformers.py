from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from ._voca import add_tokens
import os

class TransFormers(object):
    """
    입력받은 model_id에 해당되는 tokenizer와 model을 token을 추가하여 반환하는 메서드

    Argument:
        model_id: str 트랜스포머 모델 ID
        device: str CPU or GPU
        kwargs: str kwargs
            cache_dir: str 캐쉬 디렉토리
            num_labels: int (분류 모델만 사용 가능) 분류 모델일 경우 클래스 수

    Attribute:
        model_id: str 트랜스포머 모델 ID
        device (str): CPU or GPU
        cache_dir: str 캐쉬 디렉토리
        num_label: int 분류 모델일 경우 클래스 수
        device (str): CPU or GPU
        kwargs (str): Keywords Arguments
        add_tokens list[str]: 토크나이저에 추가할 토큰 목록

    Method:
        from_pretrained:(분류모델) 입력받은 model_id에 해당되는 tokenizer와 분류 model을 token을 추가하여 반환하는 메서드
        from_pretrained_atuo_model:(임베딩) 입력받은 model_id에 해당되는 tokenizer와 분류 model을 token을 추가하여 반환하는 메서드
        
    """
    def __init__(self, model_id:str, device:str, **kwargs):
        self.model_id = model_id
        self.device = device
        self.kwargs = kwargs
        self.add_tokens = add_tokens
        self.__add_tokens = tuple(add_tokens)

    def from_pretrained(self):
        """
        (분류모델 개발 목적) 입력받은 model_id에 해당되는 tokenizer와 분류 model을 token을 추가하여 반환하는 메서드
        
        Returns:
            tokenizer, model

        Example:
            from Koramco.transformers import TransFormers
            import torch
            
            model_id = "klue/roberta-base"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cache_dir = './cache'
            num_labels = 2
            
            tf = TransFormers(model_id, device, cache_dir = cache_dir, num_labels = num_labels)
            tokenizer, model = tf.from_pretrained()
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, **self.kwargs)
        tokenizer.add_tokens(self.add_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model.to(self.device)
        return tokenizer, model

    def from_pretrained_atuo_model(self):
        """
        (임베딩 목적) 입력받은 model_id에 해당되는 tokenizer와 분류 model을 token을 추가하여 반환하는 메서드
        
        Returns:
            tokenizer, model

        Example:
            from Koramco.transformers import TransFormers
            import torch
            
            model_id = "klue/roberta-base"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cache_dir = './cache'            
            
            tf = TransFormers(model_id, device, cache_dir = cache_dir)
            tokenizer, model = tf.from_pretrained_atuo_model()
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModel.from_pretrained(self.model_id, **self.kwargs)
        tokenizer.add_tokens(self.add_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model.to(self.device)
        return tokenizer, model
    
    def __repr__(self):
        return f"TransFormers(model_id: {self.model_id}, add_tokens: {self.add_tokens})"
    
    def __str__(self):
        return f"TransFormers(model_id: {self.model_id}, add_tokens: {self.add_tokens})"


# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from ._voca import add_tokens

# class TransFormers(object):
#     def __init__(self, model_id:str):
#         self.model_id = model_id
#         self.add_tokens = add_tokens
#         self.__add_tokens = tuple(add_tokens)

#     def from_pretrained(self):
#         """
#         입력받은 model_id에 해당되는 tokenizer와 model을 token을 추가하여 반환하는 메서드
        
#         Returns:
#             tokenizer, model

#         Example:
#             model_id = "klue/roberta-base"
#             tf = TransFormers(model_id)
#             tokenizer, model = tf.from_pretrained()
#         """
#         tokenizer = AutoTokenizer.from_pretrained(self.model_id)
#         model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
#         tokenizer.add_tokens(self.add_tokens)
#         model.resize_token_embeddings(len(tokenizer))
#         return tokenizer, model
    
#     # def add_tokens_(self, tokens:str|list[str]):
#     #     if type(tokens) == str:
#     #         self.add_tokens.append(tokens)

#     # def clear_tokens(self):
#     #     self.add_tokens = list(self.__add_tokens)
    
#     def __repr__(self):
#         return f"TransFormers(model_id: {self.model_id}, add_tokens: {self.add_tokens})"
    
#     def __str__(self):
#         return f"TransFormers(model_id: {self.model_id}, add_tokens: {self.add_tokens})"