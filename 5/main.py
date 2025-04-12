import math
import os
from collections import Counter, defaultdict
from typing import List

import pandas as pd
from pymystem3 import Mystem
from sklearn.metrics.pairwise import cosine_similarity


mystem = Mystem()

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


def query_to_vector(query: str, idf: dict[str, float], total_docs_count: int):
    """
    Преобразует запрос в вектор TF-IDF, с применением лемматизации к каждому слову.
    """
    words = mystem.lemmatize(query.lower())
    words = [word for word in words if word.strip() and word not in [' ', '\n']]
    query_length = len(words)
    words_counter = Counter(words)
    query_vector: dict[str, float] = {}

    for word, count in words_counter.items():
        tf = count / query_length
        # Если слово не в IDF, используем минимальный IDF
        word_idf = idf.get(word, math.log10(total_docs_count))
        query_vector[word] = tf * word_idf

    return query_vector


def compute_cosine_similarity(doc_vector: dict[str, float], query_vector:  dict[str, float]) -> float:
    """
    Вычисляет косинусное сходство между двумя векторами.
    """
    all_words = list(set(doc_vector.keys()).union(set(query_vector.keys())))
    doc_vector = [doc_vector.get(word, 0) for word in all_words]
    query_vector = [query_vector.get(word, 0) for word in all_words]
    return cosine_similarity([doc_vector], [query_vector])[0][0]


def vector_search_multi(
    queries: List[str], 
    tf_idf: dict[str, dict[str, float]], 
    idf: dict[str, float],
    total_docs_count: int, 
    output_csv_file: str
):
    results = []
    for query in queries:
        query_vector = query_to_vector(query, idf, total_docs_count)
        scores: dict[str, float] = {}
        for doc, doc_vector in tf_idf.items():
            scores[doc] = compute_cosine_similarity(doc_vector, query_vector)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for doc, score in sorted_scores:
            if score == 0:
                break
            results.append({"Query": query, "Document": doc, "Score": score})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_file, index=False)
    print(f"Результаты поиска сохранены в {output_csv_file}")


if __name__ == '__main__':
    path = '/Users/ilnargomelyanov/job/itis/2025_ITIS_IS_11-106_GomelyanovIA/2/pages'
    corpus = {}
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            corpus[file] = f.read().split()

    tf = {doc: compute_tf(text) for doc, text in corpus.items()}
    idf = compute_idf(corpus)
    tf_idf = compute_tf_idf(tf, idf)

    queries = ['хэмилтон', 'хэмилтон макларен', 'хэмилтон макларен кубок']
    output_txt = 'vector_search.txt'
    output_csv = 'vector_search.csv'
    vector_search_multi(queries, tf_idf, idf, len(corpus), output_txt, output_csv)
