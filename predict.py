from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the best model
model_distilbert = 'distilbert-base-uncased'
model = 'results/run1/checkpoint-160'
model = AutoModelForSequenceClassification.from_pretrained(model)
# We still use the tokenizer from distilbert
tokenizer = AutoTokenizer.from_pretrained(model_distilbert)

def classify_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Probability of the "Toxic" class (assuming '1' is the label for "Toxic")
    toxic_prob = probabilities[0, 1].item()  # Extract probability for class 1
    
    # Determine classification based on probability threshold
    predicted_class = 1 if toxic_prob > 0.5 else 0  # Example threshold
    classification = "Toxic" if predicted_class == 1 else "Not Toxic"
    
    return classification, toxic_prob

if __name__ == "__main__":
    # Get input from the user
    text = input("Enter text to classify: ")
    
    # Classify the text
    result, prob = classify_text(text)
    
    # Print the result and probability
    print(f"Prediction: {result}")
    print(f"Toxicity Score: {prob:.4f}")