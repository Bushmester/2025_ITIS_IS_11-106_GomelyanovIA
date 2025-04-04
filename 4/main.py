import math
import os
from collections import defaultdict

import pandas as pd

PRECISION = 6


def compute_tf(text: list[str]) -> dict[str, float]:
    """
    TF (Term Frequency) - частота термина, показывает, насколько часто термин встречается в документе.
    Вычисляется как отношение числа вхождений термина к общему количеству слов в документе.
    """
    tf: defaultdict[str, int] = defaultdict(int)
    for word in text:
        tf[word] += 1
    word_count = sum(tf.values())
    return {word: round(count / word_count, PRECISION) for word, count in tf.items()}


def compute_idf(corpus: dict[str, list[str]]) -> dict[str, float]:
    """
    IDF (Inverse Document Frequency) - обратная документная частота, используется для оценки "важности" слова во всем корпусе документов.
    Вычисляется как логарифм отношения общего числа документов к числу документов, содержащих данный термин.
    """
    idf: defaultdict[str, int] = defaultdict(int)
    total_docs = len(corpus)
    for text in corpus.values():
        for word in set(text):
            idf[word] += 1
    # добавляем единицу чтобы не было никогда нуля
    return {word: round(math.log10(total_docs / count) + 1, PRECISION) for word, count in idf.items()}


def compute_tf_idf(tf: dict[str, dict[str, float]], idf: dict[str, float]) -> dict[str, dict[str, float]]:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) - произведение TF и IDF.
    Используется для определения важности слова в конкретном документе относительно других документов в корпусе.
    Большие значения TF-IDF указывают на большую важность слова.
    """
    tf_idf: dict[str, dict[str, float]] = {}
    for file_name, tf_vals in tf.items():
        tf_idf[file_name] = {word: round(tf_val * idf[word], PRECISION) for word, tf_val in tf_vals.items()}
    return tf_idf


if __name__ == '__main__':
    input_dir_path = '/Users/ilnargomelyanov/job/itis/2025_ITIS_IS_11-106_GomelyanovIA/2/pages'
    corpus: dict[str, list[str]] = {}
    for input_file_name in os.listdir(input_dir_path):
        input_file_path = os.path.join(input_dir_path, input_file_name)
        with open(input_file_path) as file:
            corpus[input_file_name] = file.read().split()

    tf = {file_name: compute_tf(text) for file_name, text in corpus.items()}
    idf = compute_idf(corpus)
    tf_idf = compute_tf_idf(tf, idf)

    with open('result.txt', mode='w') as file:
        file.write("TF (Term Frequency):\n")
        for file_name, tf_vals in sorted(tf.items()):
            file.write(f"{file_name} - {', '.join([f'{word}: {freq}' for word, freq in tf_vals.items()])}\n")

        file.write("\nIDF (Inverse Document Frequency):\n")
        for word, freq in sorted(idf.items()):
            file.write(f"{word} - {freq}\n")

        file.write("\nTF-IDF:\n")
        for file_name, tf_idf_vals in sorted(tf_idf.items()):
            file.write(f"{file_name} - {', '.join([f'{word}: {freq}' for word, freq in tf_idf_vals.items()])}\n")

    tf_df = pd.DataFrame(tf).fillna(0).round(PRECISION)
    idf_df = pd.DataFrame(idf.items(), columns=['Word', 'IDF']).set_index('Word')

    tf_idf_records: list[dict[str, str | float]] = []
    for file_name, words in tf_idf.items():
        for word, value in words.items():
            tf_idf_records.append({'Document': file_name, 'Word': word, 'TF-IDF': value})
    tf_idf_df = pd.DataFrame(tf_idf_records)
    # Преобразование в таблицу, где строки - слова, столбцы - документы, значения - TF-IDF
    tf_idf_df_pivot_df = tf_idf_df.pivot(index='Word', columns='Document', values='TF-IDF').fillna(0).round(PRECISION)

    tf_df.to_csv('tf.csv', encoding='utf-8')
    idf_df.to_csv('idf.csv', encoding='utf-8')
    tf_idf_df_pivot_df.to_csv('tf_idf.csv', encoding='utf-8')

    print("CSV файлы успешно сохранены.")
