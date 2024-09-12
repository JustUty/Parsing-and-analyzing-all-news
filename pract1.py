import requests
import json
import logging
from typing import List, Dict, Any
from transformers import pipeline
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelне)s - %(message)s'
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9"
}

keywords_vish = ["инженерная школа", "РУТ МИИТ", "ВИШ"]
keywords_high_speed = ["ВСМ", "скоростные магистрали", "высокоскоростной"]
keywords_rzd = ["РЖД", "Российские железные дороги"]

class NewsFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def fetch_news(self, query: str) -> List[Dict[str, Any]]:
        url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&apiKey={self.api_key}"
        try:
            response = self.session.get(url, headers=HEADERS)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при запросе NewsAPI: {e}")
            if response is not None:
                logging.error(f"HTTP Status Code: {response.status_code}")
                logging.error(f"Response Content: {response.text}")
            return []

        articles = response.json().get('articles', [])
        if not articles:
            logging.info(f"Для {query} не найдено статей в ответе NewsAPI.")
        
        fetched_articles = [{"title": article['title'], "link": article['url'], "published": article['publishedAt']} for article in articles]
        logging.debug(f"Извлеченные статьи: {fetched_articles}")
        return fetched_articles

    def fetch_vish_news(self) -> List[Dict[str, Any]]:
        return self.fetch_news('Высшая инженерная школа')

    def fetch_high_speed_railways(self) -> List[Dict[str, Any]]:
        return self.fetch_news('ВСМ')

    def fetch_russian_railways(self) -> List[Dict[str, Any]]:
        return self.fetch_news('РЖД Российские Железные дороги')

class NewsParser:
    def __init__(self, fetcher: NewsFetcher):
        self.fetcher = fetcher
        self.sentiment_model = pipeline('sentiment-analysis', model="blanchefort/rubert-base-cased-sentiment")

    def filter_and_sort_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(articles, key=lambda x: x['published'], reverse=True)

    def analyze_sentiment(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for article in articles:
            title = article['title']
            result = self.sentiment_model(title)[0]
            article['sentiment'] = result['label']
            article['subjectivity'] = result['score']
            logging.debug(f"Заголовок: {article['title']}, Настроение: {article['sentiment']}, Субъективность: {article['subjectivity']}")
        return articles

    def adjust_subjectivity(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for article in articles:
            if article['sentiment'] == 'NEUTRAL':
                article['subjectivity'] = 0.1
        return articles

    def create_dashboard(self, data: List[Dict[str, Any]]) -> str:
        dashboard = """
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Новостная панель</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                }
                .container {
                    max-width: 1200px;
                    margin: 20px auto;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .filters {
                    position: fixed;
                    top: 15px;
                    left: calc(50% - 600px);
                    margin-top: 45px;
                    padding: 10px;
                    background-color: #f1f1f1;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    z-index: 1000;
                    width: auto;
                    display: flex;
                    gap: 10px;
                }
                .filters label {
                    display: flex;
                    align-items: center;
                }
                .filters input[type="checkbox"] {
                    margin-right: 5px;
                }
                ul {
                    list-style-type: none;
                    padding: 0;
                    margin-top: 60px;
                }
                li {
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 4px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    background-color: #fafafa;
                }
                a {
                    text-decoration: none;
                    color: #1a0dab;
                }
                a:hover {
                    text-decoration: underline;
                }
                .sentiment-positive {
                    color: green;
                }
                .sentiment-negative {
                    color: red;
                }
                .sentiment-neutral {
                    color: gray;
                }
                .date {
                    font-size: 0.9em;
                    color: #888;
                }
                .subjectivity {
                    font-size: 0.9em;
                    color: #555;
                }
            </style>
        </head>
        <body>
            <div class="filters">
                <label><input type="checkbox" id="positive" onclick="filterNews()"> Позитивные</label>
                <label><input type="checkbox" id="negative" onclick="filterNews()"> Негативные</label>
                <label><input type="checkbox" id="neutral" onclick="filterNews()"> Нейтральные</label>
            </div>

            <div class="container">
                <h1>Новостная панель</h1>
                <ul id="news-list">
        """
        for entry in data:
            sentiment_class = {
                'POSITIVE': 'sentiment-positive',
                'NEGATIVE': 'sentiment-negative',
                'NEUTRAL': 'sentiment-neutral'
            }.get(entry['sentiment'], 'sentiment-neutral')
            dashboard += f"<li class='{sentiment_class}' data-sentiment='{entry['sentiment']}'><a href='{entry['link']}'>{entry['title']}</a> - <span class='date'>{entry['published']}</span> (Настроение: {entry['sentiment']}, Субъективность: <span class='subjectivity'>{entry['subjectivity']:.2f}</span>)</li>"
        dashboard += """
                </ul>
            </div>

            <script>
                function filterNews() {
                    const positiveChecked = document.getElementById('positive').checked;
                    const negativeChecked = document.getElementById('negative').checked;
                    const neutralChecked = document.getElementById('neutral').checked;

                    const newsItems = document.querySelectorAll('#news-list li');
                    newsItems.forEach(item => {
                        const sentiment = item.getAttribute('data-sentiment');
                        if ((sentiment === 'POSITIVE' && positiveChecked) ||
                            (sentiment === 'NEGATIVE' && negativeChecked) ||
                            (sentiment === 'NEUTRAL' && neutralChecked)) {
                            item.style.display = 'block';
                        } else {
                            item.style.display = 'none';
                        }
                    });
                }

                filterNews();
            </script>
        </body>
        </html>
        """
        return dashboard

    def save_dashboard(self, html_content: str, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def save_to_json(self, result: List[Dict[str, Any]], filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)
        logging.info(json.dumps(result, ensure_ascii=False, indent=4))

    def read_json(self) -> List[Dict[str, Any]]:
        with open("results.json", "r", encoding="utf-8") as outfile:
            data = json.load(outfile)
        return data

    def filter_articles_by_keywords(self, articles: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        filtered_articles = []
        for article in articles:
            title = article.get('title', '')
            summary = article.get('description', '')
            if any(keyword.lower() in title.lower() or keyword.lower() in summary.lower() for keyword in keywords):
                filtered_articles.append(article)
        return filtered_articles

    def main(self):
        vish_news = self.fetcher.fetch_vish_news()
        high_speed_news = self.fetcher.fetch_high_speed_railways()
        russian_railways_news = self.fetcher.fetch_russian_railways()

        all_articles = vish_news + high_speed_news + russian_railways_news

        unique_articles = {entry['link']: entry for entry in all_articles}.values()

        sorted_articles = self.filter_and_sort_articles(list(unique_articles))
        analyzed_articles = self.analyze_sentiment(sorted_articles)
        adjusted_articles = self.adjust_subjectivity(analyzed_articles)

        filtered_vish = self.filter_articles_by_keywords(adjusted_articles, keywords_vish)
        filtered_high_speed = self.filter_articles_by_keywords(adjusted_articles, keywords_high_speed)
        filtered_rzd = self.filter_articles_by_keywords(adjusted_articles, keywords_rzd)

        final_articles = filtered_vish + filtered_high_speed + filtered_rzd

        result = {
            "unique_articles": list(unique_articles),
            "analyzed_articles": analyzed_articles,
            "filtered_vish": filtered_vish,
            "filtered_high_speed": filtered_high_speed,
            "filtered_rzd": filtered_rzd,
            "final_articles": final_articles
        }

        self.save_to_json(result, 'all_articles.json')

        dashboard_content = self.create_dashboard(final_articles)
        self.save_dashboard(dashboard_content, 'news_dashboard.html')

if __name__ == "__main__":
    api_key = '87db1ec85f5645df82e8a9d425a2b911'
    fetcher = NewsFetcher(api_key)
    parser = NewsParser(fetcher)
    parser.main()
