# Import necessary modules
from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load model and tokenizer
MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define the index view
def index(request):
    return render(request, 'index.html')

# Define the sentiment_analysis view
def sentiment_analysis(request):
    if request.method == 'POST':
        txt = request.POST.get('txt')

        # Perform sentiment prediction
        result = classifier(txt)[0]

        # Prepare data for template
        scores_dict = {
            'label': result['label'],  # POSITIVE or NEGATIVE
            'score': round(result['score'] * 100, 2),  # Percentage
            'txt': txt,
        }

        return render(request, 'index.html', scores_dict)
