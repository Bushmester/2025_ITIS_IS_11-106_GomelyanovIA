import os
import re
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem


RUSSIAN_STOPWORDS = set(stopwords.words('russian'))
mystem = Mystem()


def is_valid_word(lemma: str) -> bool:
    """Проверяет, является ли слово валидным (только буквы и допустимые дефисы)"""
    return bool(re.fullmatch(r'^[а-яёa-z-]+$', lemma, flags=re.IGNORECASE)) and lemma.strip('-')


def process_file(directory: str, file_name: str) -> Tuple[str, List[str]]:
    file_path = os.path.join(directory, file_name)
    unique_lemmas = set()  # Используем множество для уникальных лемм

    print(f'Processing file: {file_path}')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Приводим к нижнему регистру
        tokens = word_tokenize(text, language="russian")
        
        for word in tokens:
            # Лемматизируем слово и берем первую лемму (обычно это основная)
            lemmas = [lem for lem in mystem.lemmatize(word) if lem.strip()]
            if not lemmas:
                continue
                
            lemma = lemmas[0].lower().strip()
            
            # Проверяем валидность и стоп-слова
            if (is_valid_word(lemma) and 
                lemma not in RUSSIAN_STOPWORDS and
                len(lemma) > 1):  # Исключаем однобуквенные "слова"
                unique_lemmas.add(lemma)

    return file_name, sorted(unique_lemmas)


def main(source_dir: str, output_dir: str):
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with Pool(max(1, os.cpu_count() - 1)) as pool:
        results = pool.starmap(process_file, [(source_dir, f) for f in source_files])

    for file_name, lemmas in results:
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lemmas))
        print(f'Saved {len(lemmas)} unique lemmas to {output_path}')


if __name__ == "__main__":
    input_dir = '/Users/ilnargomelyanov/job/itis/2025_ITIS_IS_11-106_GomelyanovIA/1/pages'
    output_dir = '/Users/ilnargomelyanov/job/itis/2025_ITIS_IS_11-106_GomelyanovIA/2/pages'
    main(input_dir, output_dir)
