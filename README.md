Zero-Shot Classification for Medical Assistance
===============================================

### Project Description

Nothing fancy here. Just trying out how accurate a small model (tasksource/deberta-small-long-nli) can categorize medical transcripts. 
We get data from kaggle then apply a logic to get multi-class classification evaluation metrics.

### Zero-shot learning vs Transfer learning

**Transfer Learning:**

- What it does: It adapts a model trained on one task to a related task using the knowledge from the original task. The pre-trained model is fine-tuned on new, often smaller, datasets to improve performance on the new task.
- Typical use: When you have a related task but not enough data to train a model from scratch.

**Zero-Shot Classification:**

- What it does: It classifies data into categories it has never seen before by using semantic descriptions or attributes of the categories. It doesn’t require new training examples for those categories.
- Typical use: When you want the model to handle new categories based on descriptions or related features.

**How they overlap:**

Transfer Learning in Zero-Shot Classification: Zero-shot classification often uses models that have been pre-trained on large, diverse datasets (transfer learning). The pre-trained model’s learned representations or features are useful for making inferences about new classes that it hasn’t been specifically trained on. The key difference is that in zero-shot classification, the model doesn’t receive new labeled data for these unseen classes; instead, it relies on semantic information to map inputs to new categories.
So, while zero-shot classification often benefits from transfer learning by using pre-trained models, it operates differently by relying on class descriptions or attributes rather than additional training examples.




