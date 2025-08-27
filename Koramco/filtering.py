from ._filterWords import filter_words

class Filtering(object):
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