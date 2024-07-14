from sklearn.metrics.pairwise import cosine_similarity

def evaluate_similarity(vectors1, vectors2):
    similarity = cosine_similarity(vectors1, vectors2)
    average_similarity = similarity.mean()
    return average_similarity

def evaluate_harmfulness(vectors, model):
    # Placeholder function, assume model.predict returns harmfulness scores
    scores = model.predict(vectors)
    average_harmfulness = scores.mean()
    return average_harmfulness
