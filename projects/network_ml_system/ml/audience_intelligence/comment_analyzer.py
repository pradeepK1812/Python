import torch

from ml.audience_intelligence.model import SentimentModel
from ml.audience_intelligence.vocabulary import word_to_index
from ml.audience_intelligence.sample_comments import comments

# -----------------------------------
# Emotion label decoder
# -----------------------------------

index_to_label = {

    0: "appreciation",
    1: "confusion",
    2: "curiosity",
    3: "frustration",
    4: "excitement",
    5: "boredom"
}

# -----------------------------------
# Start fresh error report
# -----------------------------------

open(
    "error_analysis.txt",
    "w"
).close()

#--------------------------------------------------
# Emotion counter 
#--------------------------------------------------
emotion_counts = {

    "appreciation": 0,

    "confusion": 0,

    "curiosity": 0,

    "frustration": 0,

    "excitement": 0,

    "boredom": 0
}
# -----------------------------------
# Model configuration
# -----------------------------------

vocab_size = len(word_to_index)

embedding_dim = 64
max_seq_length = 15
# -----------------------------------
# Create model
# -----------------------------------

model = SentimentModel(
    vocab_size,
    max_seq_length,
    embedding_dim
)

# -----------------------------------
# Load trained weights
# -----------------------------------

model.load_state_dict(
    torch.load("sentiment_model.pt")
)

# -----------------------------------
# Set model to inference mode
# -----------------------------------

model.eval()

# -----------------------------------
# Convert sentence into token IDs
# -----------------------------------

def tokenize(text):

    words = text.lower().split()

   # token_ids = [] #As now we have added the CLS tocken at the start
    token_ids = [
      word_to_index["[CLS]"]
    ]

    for word in words:

        # Ignore unknown words for now
        if word in word_to_index:

            token_ids.append(
                word_to_index[word]
            )

    return token_ids

# -----------------------------------
# Input sentence
# -----------------------------------

for text,expected in comments:

      print(f"\nInput sentence is :{text} :Label: {expected} \n")
# -----------------------------------
# Convert to token IDs
# -----------------------------------

      tokens = tokenize(text)



# -----------------------------------
# Convert to tensor
# -----------------------------------

      input_tensor = torch.tensor(
        [tokens]
      )

# -----------------------------------
# Run inference
# -----------------------------------

      with torch.no_grad():

         prediction, attention_weights = model(input_tensor)


# -----------------------------------
# Convert logits into probabilities
# -----------------------------------

      probabilities = torch.softmax(
         prediction,
         dim=-1
      )

# -----------------------------------
# Get top  emotion
# -----------------------------------

      top_probs, top_indices = torch.topk(
        probabilities,
        k=1,
        dim=-1
      )

      print("\nTop Emotion Analysis:\n")



      class_index = top_indices[0][0].item()

      emotion = index_to_label[
        class_index
      ]
      print( f"Predicted Emotion: {emotion}")
      #increase the emotion count
      emotion_counts[emotion] +=1

      #update result based on comparing the predicted emotion with expected emotion
      if emotion == expected:

            result = "CORRECT"

      else:

            result = "WRONG"

            # -----------------------------------
            # Start fresh error report
            # -----------------------------------

            with open(
              "error_analysis.txt",
              "a"
            ) as f:
              f.write(
                f"Sentence : {text}\n"
                f"Expected : {expected}\n"
                f"Predicted: {emotion}\n"
                f"Result   : WRONG\n"
                f"{'-'*50}\n"
              )

      confidence = top_probs[0][0].item()
      
      print(
             f"Sentence : {text}\n"
             f"Expected : {expected}\n"
             f"Predicted: {emotion}\n"
             f"Result   : {result}"
      )

      print("-" * 50)
      print(
         f"{emotion} -> "
         f"{confidence * 100:.2f}%"
         f"(count={emotion_counts[emotion]})"
      )
"""
total_comments = len(comments)
print("\nEmotion Counts:\n")

for emotion, count in emotion_counts.items():

    print(
        f"{emotion}: {count}"
    )

print("\nAudience Emotion Distribution:\n")

for emotion, count in emotion_counts.items():

    percentage = (
        count / total_comments
    ) * 100

    print(
        f"{emotion}: "
        f"{percentage:.1f}%"
    ) """
   # -----------------------------------

