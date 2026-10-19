# v1.1.0
from django.urls import path
from .views import analyze_summary_file, analyze_summary_folder, health

urlpatterns = [
    path('health/', health, name='health'),
    path('analyze-folder/', analyze_summary_folder, name='analyze-folder'),
    path('analyze-file/', analyze_summary_file, name='analyze-file'),
]
