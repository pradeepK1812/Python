#from pydantic import BaseModel
#from typing import List

#class PredictionRequest(BaseModel):
 #   features: List[float]

#class PredictionResponse(BaseModel):
#    prediction: float


from pydantic import BaseModel, conlist

class PredictionRequest(BaseModel):
    features: conlist(float, min_length=2, max_length=2)

class PredictionResponse(BaseModel):
    prediction: float
