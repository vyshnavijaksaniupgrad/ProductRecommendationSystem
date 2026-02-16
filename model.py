import pickle
import pandas as pd
import numpy as np

user_item_matrix = pickle.load(open('user_item_matrix.pkl', 'rb'))
item_based_predictions = pickle.load(open('item_based_predictions.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
sentiment_model = pickle.load(open('sentiment_model.pkl', 'rb'))

# -------------------------------------------------
# Item-based Collaborative Filtering Recommendation
# -------------------------------------------------

def recommend_item_based(username, user_item_matrix, item_based_predictions, n=20):
    """
    Recommend top-N products using item-based collaborative filtering
    """
    if username not in user_item_matrix.index:
        return pd.Series(dtype=float)

    user_ratings = user_item_matrix.loc[username]
    unrated_products = user_ratings[user_ratings.isna()].index

    recommendations = item_based_predictions.loc[username, unrated_products]
    return recommendations.sort_values(ascending=False).head(n)


# -------------------------------------------------
# Sentiment Scoring
# -------------------------------------------------

def get_product_sentiment(product_name, df, tfidf_vectorizer, sentiment_model):
    """
    Compute average positive sentiment score for a product
    """
    reviews = df[df["name"] == product_name]["reviews_text"]

    if reviews.empty:
        return 0

    review_tfidf = tfidf_vectorizer.transform(reviews)

    # Probability of positive sentiment
    sentiment_prob = sentiment_model.predict_proba(review_tfidf)[:, 1]

    return sentiment_prob.mean()


# -------------------------------------------------
# Hybrid Recommendation System
# -------------------------------------------------

def recommend_with_sentiment(username, df, top_n=5):
    """
    Recommend top-5 products using item-based CF + sentiment analysis
    """

    top_20_products = recommend_item_based(
        username,
        user_item_matrix,
        item_based_predictions,
        n=20
    )

    if top_20_products.empty:
        return pd.DataFrame(columns=["product_name", "sentiment_score"])

    sentiment_scores = {}

    for product in top_20_products.index:
        sentiment_scores[product] = get_product_sentiment(
            product,
            df,
            tfidf_vectorizer,
            sentiment_model
        )

    sentiment_df = (
        pd.DataFrame.from_dict(
            sentiment_scores,
            orient="index",
            columns=["sentiment_score"]
        )
        .sort_values(by="sentiment_score", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "product_name"})
    )

    return sentiment_df
