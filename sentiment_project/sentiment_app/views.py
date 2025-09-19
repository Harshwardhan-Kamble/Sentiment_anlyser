from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_protect
from django.middleware.csrf import get_token
import json
import os
import sys
from pathlib import Path

# Add the analyzer directory to the Python path
current_dir = Path(__file__).resolve().parent
analyzer_dir = current_dir / 'analyzer'
sys.path.append(str(analyzer_dir))

from sentiment_analyser import SentimentPredictor

# Initialize the sentiment predictor with the correct model path
model_base_path = analyzer_dir
predictor = SentimentPredictor(model_name=str(model_base_path / 'twitter_sentiment'))

@ensure_csrf_cookie
def index(request):
    # Force CSRF cookie to be set
    get_token(request)
    return render(request, 'sentiment_app/index.html')

@csrf_protect
def analyze_sentiment(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        tweet = data.get('tweet', '')
        
        if not tweet:
            return JsonResponse({'error': 'No tweet provided'}, status=400)
        
        try:
            result = predictor.predict_sentiment(tweet)
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
