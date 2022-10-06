# train2generalize
Use MAML on a single dataset in order to generalize instead of matching patterns

Similar paper: https://arxiv.org/pdf/2002.12455.pdf
They sample two batches. Do an update on the first batch and save the computation to the loss. Then calculate the loss of that updated network on a second randomly sampled batch from the same dataset and calculate the loss of that network towards the original weight parameters. This constitues the second loss components. The loss for both batches is then weighted with a hyperparameter.
Contrast to our approach:
1. we use a distinct generalization split of the dataset, instead of sampling both batches from the same dataset. This makes sure that the generalization ability is tested.
2. They use the loss of the first batch too. We, on the other hand, only use the second order loss.

Relations to second order optimizers such as Shampoo?
https://arxiv.org/pdf/1802.09568.pdf
Distributed shampoo: https://openreview.net/pdf?id=Sc8cY4Jpi3s

## Setup
`python3 -m pip install -r requirements.txt`

Torchviz requires graphviz: `sudo apt-get install graphviz`
