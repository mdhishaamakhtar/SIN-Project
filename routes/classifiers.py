from fastapi import APIRouter, HTTPException, status
from schemas.models import ReviewResponse, ReviewRequest
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pickle
import json

with open("routes/product_reviews.json") as f:
    reviews = f.read().strip().split("\n")
reviews = [json.loads(review) for review in reviews]
texts = [review['reviewText'] for review in reviews]
stop_words = set(stopwords.words("english"))
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
vectorizer.fit_transform(texts)
router = APIRouter(tags=["classifiers"])
with open("routes/DT_Classifier.pkl", "rb") as fid:
    dt_classifier = pickle.load(fid)
with open("routes/DT_Classifier.pkl", "rb") as fid:
    svm_classifier = pickle.load(fid)


@router.post("/create", status_code=status.HTTP_200_OK)
def create_post(review_texts: ReviewRequest):
    content = review_texts.review
    print(content)
    words = content.split()
    print(words)
    tokenized_text = ""
    rcount = 0
    for r in words:
        if r not in stop_words:
            tokenized_text = tokenized_text + " " + r
            rcount = rcount + 1
    if rcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Review Given"
        )
    else:
        rev_list = [tokenized_text]
        print(rev_list)
        vec = vectorizer.transform(rev_list)
        svm_pred = svm_classifier.predict(vec)
        dt_pred = dt_classifier.predict(vec)
        return ReviewResponse(
            status="Success",
            message="Ratings Found",
            svm_rating=str(svm_pred[0]),
            decision_tree_rating=str(dt_pred[0]),
        )
