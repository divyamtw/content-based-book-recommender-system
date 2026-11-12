import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_pickle("books.pkl")
vectors = pd.read_pickle("vectors.pkl")


def recommend(book_title, top_n=5):
    book_idx = books[books["Book-Title"] == book_title].index[0]
    sim_scores = cosine_similarity(vectors[book_idx], vectors).flatten()
    similar_indices = np.argsort(sim_scores)[::-1][1 : top_n + 1]

    return books.iloc[similar_indices]


recs = recommend("Tell Me This Isn't Happening")

book = {
    "title": recs["Book-Title"].tolist(),
    "author": recs["Book-Author"].tolist(),
    "publisher": recs["Publisher"].tolist(),
    "img": recs["Image-URL-L"].tolist(),
    "isbn": recs["ISBN"].tolist(),
}

print(
    f"Title: {book['title'][0]}\n"
    f"Author: {book['author'][0]}\n"
    f"Publisher: {book['publisher'][0]}\n"
    f"Img: {book['img'][0]}\n"
    f"Isbn: {book['isbn'][0]}"
)
