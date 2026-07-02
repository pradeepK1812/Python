#from pydantic import BaseModel
#from typing import List

#class PredictionRequest(BaseModel):
 #   features: List[float]

#class PredictionResponse(BaseModel):
#    prediction: float


#from pydantic import BaseModel, conlist,List

#class PredictionRequest(BaseModel):
    #features: conlist(float, min_length=2, max_length=2)
 #   features: List[List[float]]

#class PredictionResponse(BaseModel):
 #   prediction: float


from pydantic import BaseModel, field_validator
from typing import List


class PredictionResponse(BaseModel):
    prediction: List[float]

class PredictionRequest(BaseModel):
    features: List[List[float]]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        for row in v:
            if len(row) != 2:
                raise ValueError("Each feature vector must contain exactly 2 values")
        return v

class GenerationRequest(BaseModel):

    prompt: str

    max_tokens: int = 5

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value):

         if not value.strip():
            raise ValueError("Prompt cannot be empty")

         return value


class GenerationResponse(BaseModel):

    generated_text: str

