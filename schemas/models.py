from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ReviewResponse(BaseModel):
    status: str
    message: str
    svm_rating: str
    decision_tree_rating: str


class ReviewRequest(BaseModel):
    review: str
