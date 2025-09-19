from django.urls import path
from . import views

app_name = 'sentiment_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze_sentiment, name='analyze'),
]