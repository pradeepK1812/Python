import torch

from model import SentimentModel
from vocabulary import word_to_index
from evaluation_test import test_sentences

# =========================================================
# Emotion label decoder
# =========================================================

index_to_label = {

    0: "appreciation",
    1: "confusion",
    2: "curiosity",
    3: "frustration",
    4: "excitement",
    5: "boredom"
}

# =========================================================
# Model configuration
# =========================================================

vocab_size = len(word_to_index)

embedding_dim = 64

max_seq_length = 15

# =========================================================
# Create model
# =========================================================

model = SentimentModel(

    vocab_size,
    max_seq_length,
    embedding_dim
)

# =========================================================
# Load trained model
# =========================================================

model.load_state_dict(

    torch.load("sentiment_model.pt")
)

# =========================================================
# Inference mode
# =========================================================

model.eval()

# =========================================================
# Tokenization
# =========================================================

def tokenize(text):

    words = text.lower().split()

    token_ids = [

        word_to_index["[CLS]"]
    ]

    for word in words:

        if word in word_to_index:

            token_ids.append(

                word_to_index[word]
            )

    return token_ids

# =========================================================
# Evaluation loop
# =========================================================

correct = 0

total = len(test_sentences)

print("\n==============================")
print("EVALUATION RESULTS")
print("==============================\n")

for text, expected_label in test_sentences:

    # -----------------------------------
    # Tokenize
    # -----------------------------------

    tokens = tokenize(text)

    input_tensor = torch.tensor(
        [tokens]
    )

    # -----------------------------------
    # Run inference
    # -----------------------------------

    with torch.no_grad():

        prediction, _ = model(
            input_tensor
        )

    # -----------------------------------
    # Convert logits to probabilities
    # -----------------------------------

    probabilities = torch.softmax(

        prediction,
        dim=-1
    )

    # -----------------------------------
    # Get predicted class
    # -----------------------------------

    predicted_index = torch.argmax(

        probabilities,
        dim=-1

    ).item()

    predicted_label = index_to_label[
        predicted_index
    ]

    # -----------------------------------
    # Compare prediction
    # -----------------------------------

    is_correct = (

        predicted_label ==
        expected_label
    )

    if is_correct:

        correct += 1

    # -----------------------------------
    # Display result
    # -----------------------------------

    print(f"Sentence : {text}")

    print(
        f"Expected : {expected_label}"
    )

    print(
        f"Predicted: {predicted_label}"
    )

    print(
        f"Result   : "
        f"{'CORRECT' if is_correct else 'WRONG'}"
    )

    print("-" * 50)

# =========================================================
# Final accuracy
# =========================================================

accuracy = (

    correct / total
) * 100

print("\n==============================")

print(
    f"FINAL ACCURACY: "
    f"{accuracy:.2f}%"
)

print("==============================\n")
