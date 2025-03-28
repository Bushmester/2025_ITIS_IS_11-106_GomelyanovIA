import re
import argparse
import urllib.parse
from typing import List, Tuple, Set
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def crawl_page(url: str) -> bytes | None:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error occurred while fetching {url}:", e)
        return None


def extract_text(html_content: bytes) -> str:
    try:
        soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
    except:
        soup = BeautifulSoup(html_content, 'html.parser')

    # Удаление тегов
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=' ', strip=True)

    # Замена подряд идущих символов переноса строк на один
    text = '\n'.join([line for line in text.splitlines() if line])

    return text


def save_page(text: str, filename: str):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


def has_cyrillic(text: str) -> bool:
    return bool(re.search(r'[а-яёА-ЯЁ]', text[0]))


def normalize_url(url: str) -> str:
    """Нормализация URL с декодированием percent-encoding"""
    try:
        # Разбираем URL, декодируем каждый компонент и собираем обратно
        parsed = urllib.parse.urlparse(url)
        new_path = urllib.parse.unquote(parsed.path)
        new_query = urllib.parse.unquote(parsed.query)
        return urllib.parse.urlunparse(parsed._replace(path=new_path, query=new_query))
    except:
        return url


def main(urls: List[str]):
    Path("pages").mkdir(exist_ok=True)
    
    target_count = 100
    min_words = 1000
    index: List[Tuple[str, str]] = []
    visited: Set[str] = set()
    queue = urls.copy()
    
    saved = 0
    while saved < target_count and queue:
        url = queue.pop(0)
        normalized_url = normalize_url(url)
        
        if normalized_url in visited:
            continue
            
        print(f"Processing: {normalized_url}")
        visited.add(normalized_url)
        
        content = crawl_page(url)
        if not content:
            continue
            
        try:
            text = extract_text(content)
        except Exception as e:
            print(f"Extraction error: {normalized_url} - {e}")
            continue
        
        # Проверка на кириллицу
        if not has_cyrillic(text):
            print(f"Skipping (no cyrillic): {normalized_url}")
            continue
            
        # Проверка длины контента
        word_count = len(text.split())
        if word_count < min_words:
            print(f"Skipping (too short): {normalized_url}")
            continue
            
        # Сохраняем страницу
        page_num = f"{saved+1:03d}"
        filename = f"pages/{page_num}.txt"
        save_page(text, filename)
        
        # Добавляем в индекс с нормализованным URL
        index.append((page_num, normalized_url))
        saved += 1
        print(f"Saved {page_num}: {normalized_url}")
        
        # Парсим новые ссылки
        try:
            soup = BeautifulSoup(content, 'html.parser')
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('#'):
                    continue
                absolute_url = urllib.parse.urljoin(url, href)
                normalized = normalize_url(absolute_url)
                if (normalized.startswith('http') 
                    and not any(ext in normalized for ext in ['.css', '.js', '.png', '.jpg', '.gif', '.pdf', '.zip'])
                    and normalized not in visited
                    and normalized not in queue):
                    links.add(normalized)
            queue.extend(links)
        except Exception as e:
            print(f"Link parsing error: {normalized_url} - {e}")

    # Сохраняем индекс
    with open('index.txt', 'w', encoding='utf-8') as f:
        for num, url in index:
            f.write(f"{num} {url}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "urls",
        nargs="+",
        help="Список начальных веб-адресов."
    )
    args = parser.parse_args()
    
    main(args.urls)
