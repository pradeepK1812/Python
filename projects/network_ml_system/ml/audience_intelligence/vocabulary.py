from audience_intelligence.sample_data import training_data

# -----------------------------------
# Vocabulary dictionary
# -----------------------------------

word_to_index = {}

# -----------------------------------
# Special transformer tokens
# -----------------------------------

word_to_index["[CLS]"] = 0
# -----------------------------------
# Current word index counter
# -----------------------------------

#current_index = 0
current_index = 1 # as now CLS is at index 0

# -----------------------------------
# Build vocabulary
# -----------------------------------

for text, label in training_data:

    # Convert sentence into words
    words = text.split()

    for word in words:

        # Add unseen words to vocabulary
        if word not in word_to_index:

            word_to_index[word] = current_index

            current_index += 1

# -----------------------------------
# Print vocabulary
# -----------------------------------

if __name__ == "__main__":

    print("\nVocabulary:\n")

    for word, index in word_to_index.items():

        print(f"{word} -> {index}")

