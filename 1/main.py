import re
import argparse
from typing import Sequence
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def crawl_page(url: str) -> bytes | None:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        print(f"Error occurred while fetching {url}:", e)
        return None


def extract_text(html_content: bytes) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    # Удаление тегов
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()

    # Замена подряд идущих символов переноса строк на один
    text = '\n'.join([line for line in text.splitlines() if line])
    return text


def save_page(text: str, filename: str):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


def main(urls: Sequence[str]):
    required_pages_count = 100
    min_required_words_count = 1000
    saved_pages_count = 0
    index: Sequence[(str, str)] = []
    visited_urls: set[str] = set()
    
    Path("pages").mkdir(exist_ok=True)
    
    while saved_pages_count < required_pages_count and urls:
        url: str = urls.pop(0)
        if url not in visited_urls:
            html_content = crawl_page(url)
            if html_content:
                visited_urls.add(url)
                text = extract_text(html_content)

                # Проверка на количество слов
                if len(text.split()) >= min_required_words_count:
                    page_number = f"{saved_pages_count + 1}".zfill(3)
                    page_filename = "pages/" + page_number + ".txt"
                    save_page(text, page_filename)
                    index.append((page_number, url))
                    saved_pages_count += 1
                    print(f"Saved page {page_number}: {url}")

                # Поиск ссылок на другие страницы
                try:
                    links = re.findall(r'href=[\'"]?([^\'" >]+)', html_content.decode('utf-8'))
                except UnicodeDecodeError:
                    print("Failed to find links:", url)
                else:
                    urls.extend(
                        link.split('#')[0] for link in links
                        if link.startswith('http') and
                        'css' not in link and
                        'json' not in link and
                        link not in visited_urls
                    )
            else:
                print("Failed to crawl:", url)
        else:
            print("Already visited:", url)

    with open('index.txt', 'w', encoding='utf-8') as index_file:
        for entry in index:
            index_file.write(f"{entry[0]} - {entry[1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "urls",
        nargs="+",
        help="Список начальных веб-адресов."
    )
    args = parser.parse_args()
    
    main(args.urls)
