import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score

Ratings_file = "u.data"
items_file = "u.item"
random_seed = 42
test_ratings = 10
top_K = 5
neighpors = 20
similarity = "cosine"
min_rating = 20

np.random.seed(random_seed)
random.seed(random_seed)


def load_ml_100k(ratings_file=Ratings_file, items_file=items_file):
    names = ["userid", "movieid", "rating", "timestamp"]
    ratings = pd.read_csv(ratings_file, sep="\t", names=names, engine="python")
    items = pd.read_csv(items_file, sep="|", names=range(24), encoding="latin-1", engine="python")
    items = items[[0,1]]
    items.columns = ["movieid", "title"]
    ratings = ratings.merge(items, on="movieid", how="left")
    return ratings, items

def create_user_item(ratings_df):
    ui = ratings_df.pivot_table(index="userid", columns="movieid", values="rating")
    return ui



def train_test_split(ratings_df, n_holdout=test_ratings, min_ratings=min_rating):
    users = ratings_df['userid'].unique()
    train_list = []
    test_list = []
    grouped = ratings_df.groupby('userid')
    for u in users:
        user_ratings= grouped.get_group(u)
        if len(user_ratings) >= n_holdout and len(user_ratings) >= min_ratings:
            test_idx = user_ratings.sample(n=n_holdout, random_state=random_seed).index
            test_list.append(user_ratings.loc[test_idx])
            train_list.append(user_ratings.drop(test_idx))
        else:
           train_list.append(user_ratings)

    train = pd.concat(train_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True) if len(test_list) > 0 else pd.DataFrame(columns=ratings_df.columns)
    return train, test

def computer_user_similarity(user_item_matrix, metric="cosine"):
    if metric == "cosine":
        mat = user_item_matrix.fillna(0).values
        sim = cosine_similarity(mat)
        sim_df = pd.DataFrame(sim, index=user_item_matrix.index, columns=user_item_matrix.index)
        return sim_df
    elif metric == "person":
        corr = user_item_matrix.T.corr(method='person', min_periods=1)
        corr = corr.fillna(0)
        return corr
    else:
        raise ValueError("Unsupported metric: choose 'cosine' or 'person'")


def recommend_for_user(user_id, user_item_matrix, user_similarity_df, top_n=top_K, n_neighbors=neighpors):
    if user_id not in user_item_matrix.index:
        return pd.Series(dtype=float)
    sims = user_similarity_df.loc[user_id].drop(index=user_id, errors='ignore')
    top_neighbors = sims.sort_values(ascending=False).head(n_neighbors)
    if top_neighbors.sum() == 0:
        return pd.Series(dtype=float)
    
    neighbors_ratings = user_item_matrix.loc[top_neighbors.index]
    weighted_sum = neighbors_ratings.fillna(0).T.dot(top_neighbors)
    normalizer = np.abs(top_neighbors).sum()
    pred_scores = weighted_sum / (normalizer + 1e-9)
    user_rated = user_item_matrix.loc[user_id].dropna().index
    preds = pred_scores.drop(index=user_rated, errors='ignore')
    return preds.head(top_n)


def precision_at_k_for_user(recommended_movieids, test_positive_movieids, k=top_K):
    if len(recommended_movieids) == 0:
        return 0.0
    
    recommended_k = list(recommended_movieids)[:k]
    hits = sum(1 for m in recommended_k if m in test_positive_movieids)
    return hits / k

def evaluate_precision_at_k(train_df, test_df, k=top_K, metric=similarity, n_neighbors=neighpors):
    train_ui = create_user_item(train_df)
    user_similarity_df = computer_user_similarity(train_ui, metric=metric)
    test_group = test_df.groupby('userid')
    test_positives = {}
    for user, grp in test_group:
        pos = set(grp[grp['rating'] >= 4]['movieid'].tolist())
        test_positives[user] = pos

    precision = []
    users_evaluated = 0
    for user in test_positives:
       if user not in train_ui.index:
           continue
       if len(test_positives[user]) == 0:
           continue
       preds = recommend_for_user(user, train_ui, user_similarity_df, top_n=k, n_neighbors=n_neighbors)
       prec  = precision_at_k_for_user(preds.index.tolist(), test_positives[user], k=k)
       precision.append(prec)
       users_evaluated += 1

    avg_prec = np.mean(precision) if len(precision) > 0 else 0.0
    return{
        "avg_precision_at_{}".format(k): avg_prec,
        "users_evaluate": users_evaluated
    }


def main():
    print("loading data")
    ratings, items = load_ml_100k()
    print(f"Total ratings: {len(ratings)}, unique users:{ratings['userid'].nunique()}, unique movies: {ratings['movieid'].nunique}")
    print("splitting train/test (leave-out)")
    train, test = train_test_split(ratings, n_holdout=test_ratings, min_ratings= min_rating)
    print(f"train ratings: {len(train)}, test ratings: {len(test)}")
    print(f"unique users in test set: {test['userid'].nunique()}")
    print("building user-item matrix from training set")
    train_ui = create_user_item(train)
    print(f"computing user similarity using metric='{similarity}'")
    user_sim_df = computer_user_similarity(train_ui, metric=similarity)
    sample_user = train_ui.index[0]
    print(f"\nsample recommendation for user {sample_user}:")
    recs = recommend_for_user(sample_user, train_ui, user_sim_df, top_n=10, n_neighbors=neighpors)
    if recs.empty:
        print("no recommendation (cold-start or no similar users).")
    else:
        movie_titles = items.set_index('movieid')['title'].to_dict()
        for movie_id, score in recs.items():
            title = movie_titles.get(movie_id, str(movie_id))
            print(f" {title} (movie_id)) predicted_score={score:.3f}")
    print("\nevaluate precision on test users")
    eval_results = evaluate_precision_at_k(train, test, k=top_K, metric=similarity, n_neighbors=neighpors)
    print("evaluate results:", eval_results)


if __name__ == "__main__":
    main()


    