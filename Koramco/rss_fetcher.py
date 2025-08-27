import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import html

from ._rss_source import rss_sources

class Rss_Fecher(object):
    def __init__(self):
        self.rss_sources = rss_sources
        self.today = datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def normalize_date(date_str):
        try:
            # 형식 1: RFC822 스타일 ("Mon, 14 Jul 2025 16:09:28 +0900")
            dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
            
        try:
            # 형식 1: RFC822 스타일 ("Mon,14 Jul 2025 16:09:28 +0900")
            dt = datetime.strptime(date_str, "%a,%d %b %Y %H:%M:%S %z")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
            
        try:
            # 형식 2: ISO 스타일 ("2025-07-14")
            dt = datetime.fromisoformat(date_str)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    
        return None  # 실패 시 None 반환

    @staticmethod
    def get_text(tag):
        return tag.text.strip() if tag else None

    
    def crawl_source(self, source_name):
        """한 매체에 대한 뉴스 크롤링"""
        if source_name not in rss_sources:
            raise ValueError(f"{source_name} 는 sources_config에 등록되어 있지 않습니다.")
            
        source_info = self.rss_sources[source_name]
        feed_type = source_info.feed_type
        date_tag = source_info.date_tag
        title_tag = source_info.title_tag
        link_tag = source_info.link_tag

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            # 필요하다면 다른 헤더도 추가할 수 있습니다 (예: Accept-Language, Referer 등)
            # "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            # "Referer": "https://www.google.com/"
        }

        
        response = requests.get(source_info.url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            raise RuntimeError(f"요청 실패: {response.status_code} - {response.reason}")
        
        response = response.content.decode('utf-8', errors='replace')
        if source_name in ['연합인포맥스', 'EBN산업경제', '블로터', '아주경제', '보험매일', '뉴스톱', '이데일리']:
            response = html.unescape(response)
        soup = BeautifulSoup(response, "xml")
        rows = []
    
        for item in soup.find_all(feed_type):
            title = self.get_text(item.find(title_tag))
            if source_name =='이데일리':
                link = self.get_text(item.find(link_tag))
                link = link[:-4] # '=257' 제거
            else:
                link = self.get_text(item.find(link_tag))
            date = self.get_text(item.find(date_tag))
            date = self.normalize_date(date)
            short_date = date[:10] if date else None
            if short_date != self.today :
                break  # 오늘 날짜가 아니면 그 매체 수집 중단
                
            rows.append({
                    "source": source_name,
                    "title": title,
                    "link": link,
                    "date": short_date
                })
            
        return pd.DataFrame(rows)

    def crawl_all(self):
        """등록된 모든 소스를 수집"""
        all_rows = []
        tot_cnt = 0
        print(f'{self.today}일자 RSS 수집 시작')
        for source_name in self.rss_sources.keys():
            try:
                df = self.crawl_source(source_name)
                print(f"{source_name}, {len(df)} 건 수집 성공")
                all_rows.append(df)
                tot_cnt+=len(df)
            except Exception as e:
                print(f"[WARN] {source_name} 수집 실패: {e}")

        print(f"총 {tot_cnt}건 수집 완료")
        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        else:
            return pd.DataFrame(columns=["source", "title", "link", "date"])



        
        
