Toxicity Detection in Comments
==================================

This project aims to detect toxicity in comments using various approaches. The dataset used for evaluation is `youtoxic_english_1000.csv`, which contains comments labeled for toxicity. We explore three distinct methods to achieve this task:

1. **Fine-Tuning a Pre-trained Model for Classification**
2. **Using a Pre-trained Toxicity Detection Model**
3. **Fine-Tuning a Toxicity Detection Model on Custom Data**

### Dataset:
The dataset used in this project is `youtoxic_english_1000.csv`. This relevant information in the dataset includes the following:
- **Description**: The text of the comment.
- **IsToxic**: The boolean identifier to flag if the comment was indeed toxic.

### Approaches:

1. Fine-Tuning a Pre-trained Model for Classification

In this approach, we start with a pre-trained model originally designed for fill-masking tasks. We adapt this model for toxicity classification by fine-tuning it on the provided dataset. This involves:
- Loading a pre-trained fill-masking model.
- Modifying the model architecture for classification.
- Training the model on the training set.
- Evaluating the model's performance on the test set.

2. Using a Pre-trained Toxicity Detection Model

For this method, we leverage a pre-trained toxicity detection model available on Hugging Face. This model is already fine-tuned for toxicity detection tasks and can be used directly to classify comments. Steps include:
- Loading the pre-trained model from Hugging Face.
- Evaluating the model's performance on the test set.

3. Fine-Tuning a Toxicity Detection Model on Custom Data

In this approach, we take an existing toxicity detection model and further fine-tune it using the training data. This method aims to adapt the model specifically to our dataset. The process involves:
- Selecting a pre-trained toxicity detection model.
- Fine-tuning the model on the training set
- Evaluating the model's performance on the test set.


### Credits/Citations:
Pretrained Network:
"martin-ha/toxic-comment-model" - https://huggingface.co/martin-ha/toxic-comment-model



