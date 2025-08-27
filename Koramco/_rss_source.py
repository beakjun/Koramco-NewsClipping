from dataclasses import dataclass

# dictionary 구성
@dataclass
class SourceInfo:
    url: str
    feed_type: str # "rss" 또는 "urlset"
    date_tag: str
    title_tag: str = "title"  # 기본값 "title"
    link_tag: str = "link"    # 기본값 "link"


rss_sources = {
    "뉴스1": SourceInfo(
        url="https://www.news1.kr/api/feeds/news-sitemap/section/realestate",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "더벨": SourceInfo(
        url="https://www.thebell.co.kr/newspartner/google.asp",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "조선일보 땅집고": SourceInfo(
        url="https://realty.chosun.com/site/data/rss/rss.xml",
        feed_type="item",
        date_tag="dc:date",
        title_tag="title", 
        link_tag = "link"
    ),
    "인베스트조선": SourceInfo(
        url="https://www.investchosun.com/freenews/rss/freenews.xml",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag = "link"
    ),
    "블로터": SourceInfo(
        url="https://www.bloter.net/sitemap.xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "대한금융신문": SourceInfo(
        url="https://www.kbanker.co.kr/sitemap.xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),   
    "위키리크스한국": SourceInfo(
        url="http://www.wikileaks-kr.org/rss/allArticle.xml",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag = "link"
    ),
    "쿠키뉴스": SourceInfo(
        url="https://www.kukinews.com/rss/",
        feed_type="item",
        date_tag="dc:date",
        title_tag="title", 
        link_tag = "link"
    ),
    "연합인포맥스": SourceInfo(
    url="https://news.einfomax.co.kr/sitemap.xml",
    feed_type="url",
    date_tag="news:publication_date",
    title_tag="title",
    link_tag="loc",
    ),
    "EBN산업경제": SourceInfo(
        url="https://www.ebn.co.kr/sitemap.xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "보험매일": SourceInfo(
        url="https://www.fins.co.kr/sitemap.xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "데일리안": SourceInfo(
        url="https://center.dailian.co.kr/sitemap/newssitemap.xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "한국주택경제": SourceInfo(
        url="https://www.arunews.com/sitemap.xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "매일경제": SourceInfo(
        url="https://www.mk.co.kr/rss/50300009/",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    ),
    "파이낸셜뉴스": SourceInfo(
        url="https://www.fnnews.com/rss/r20/fn_realnews_all.xml",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    ),
    "조선비즈": SourceInfo(
        url="https://biz.chosun.com/arc/outboundfeeds/news-sitemap/?outputType=xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "이데일리": SourceInfo(
        url="https://www.edaily.co.kr/sitemap/latest-article.xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "한국경제": SourceInfo(
        url="https://www.hankyung.com/sitemap/latest-article",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "서울경제": SourceInfo(
        url="https://www.sedaily.com/rss/newsall",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    ),
    "아주경제": SourceInfo(
        url="https://www.ajunews.com/sitemap.php",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "대한경제": SourceInfo(
        url="https://www.dnews.co.kr/rss/news_main.xml",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    ),
    "팍스넷뉴스": SourceInfo(
        url="http://www.paxetv.com/rss/allArticle.xml",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    ),
    "아시아경제": SourceInfo(
        url="https://view.asiae.co.kr/rss/all.htm",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    ),
    "연합뉴스": SourceInfo(
        url="https://www.yna.co.kr/rss/economy.xml",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    ),
    "뉴스톱": SourceInfo(
        url="https://www.newstof.com/sitemap.xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "머니투데이": SourceInfo(
        url="https://mt.co.kr/b2b/mt_news_google.xml",
        feed_type="url",
        date_tag="news:publication_date",
        title_tag="title",
        link_tag="loc",
    ),
    "리걸타임즈": SourceInfo(
        url="https://www.legaltimes.co.kr/rss/allArticle.xml",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    ),
    "하우징헤럴드": SourceInfo(
        url="https://www.housingherald.co.kr/rss/allArticle.xml",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    ),
    "1코노미뉴스": SourceInfo(
        url="https://cdn.1conomynews.co.kr/rss/gn_rss_allArticle.xml",
        feed_type="item",
        date_tag="pubDate",
        title_tag="title",
        link_tag="link",
    )
}
