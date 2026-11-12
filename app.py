from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
books = pd.read_pickle(BASE_DIR / "books.pkl")
vectors = pd.read_pickle(BASE_DIR / "vectors.pkl")
titles = books["Book-Title"].astype(str)
normalized_titles = titles.str.casefold()

if len(books) != vectors.shape[0]:
    raise ValueError("The book catalog and recommendation vectors are misaligned.")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/suggestions")
def suggestions():
    query = request.args.get("q", "").strip().casefold()
    if len(query) < 2:
        return jsonify(suggestions=[])

    prefix_matches = titles[normalized_titles.str.startswith(query)].drop_duplicates()
    results = prefix_matches.head(8).tolist()

    if len(results) < 8:
        contains_matches = titles[
            normalized_titles.str.contains(query, regex=False, na=False)
        ].drop_duplicates()
        seen = set(results)
        results.extend(title for title in contains_matches if title not in seen)

    return jsonify(suggestions=results[:8])


@app.get("/api/recommend")
def recommend():
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify(error="Please enter a book title."), 400

    title_key = title.casefold()
    matching_positions = np.flatnonzero(normalized_titles.to_numpy() == title_key)
    if matching_positions.size == 0:
        return jsonify(error="That title is not in the book collection."), 404

    book_position = int(matching_positions[0])
    similarity_scores = cosine_similarity(vectors[book_position], vectors).ravel()
    candidate_positions = np.argsort(similarity_scores)[::-1]

    recommendations = []
    seen_titles = {title_key}
    for position in candidate_positions:
        candidate_title = titles.iloc[position]
        candidate_key = candidate_title.casefold()
        if candidate_key in seen_titles:
            continue

        seen_titles.add(candidate_key)
        book = books.iloc[position]
        recommendations.append(
            {
                "title": candidate_title,
                "author": str(book["Book-Author"]),
                "publisher": str(book["Publisher"]),
                "image_url": str(book["Image-URL-L"]),
                "isbn": str(book["ISBN"]),
            }
        )
        if len(recommendations) == 5:
            break

    return jsonify(
        success=True,
        searched_title=titles.iloc[book_position],
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run()
