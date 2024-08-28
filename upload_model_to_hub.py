from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


# Load the best model
model_distilbert = 'distilbert-base-uncased'
model = 'results/run1/checkpoint-160'
model = AutoModelForSequenceClassification.from_pretrained(model)
# We still use the tokenizer from distilbert
tokenizer = AutoTokenizer.from_pretrained(model_distilbert)

# Instantiate the Trainer with only the model and tokenizer
trainer = Trainer(
    model=model, 
    tokenizer=tokenizer
)

trainer.create_model_card(model_name='distilbert-base-uncased-yt')
trainer.push_to_hub(commit_message="Youtube comments toxicity classification")