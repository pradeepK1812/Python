import torch

from mini_bert_classifier.mini_bert_classifier  import MiniBERTClassifier

model = MiniBERTClassifier(
    vocab_size=100,
    embedding_dim=8,
    max_seq_length=10,
    num_layers=3,
    num_classes=5
)

x = torch.randint(
    0,
    100,
    (1, 5)
)

output = model(x)

print("\nOutput Shape:")
print(output.shape)
