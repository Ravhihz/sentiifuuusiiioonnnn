from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Dataset Management
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/upload/', views.dataset_upload, name='dataset_upload'),
    path('datasets/<int:pk>/', views.dataset_detail, name='dataset_detail'),
    path('datasets/<int:pk>/delete/', views.dataset_delete, name='dataset_delete'),
    
    # Preprocessing
    path('datasets/<int:pk>/preprocess/', views.dataset_preprocess, name='dataset_preprocess'),
    path('datasets/<int:pk>/analysis/', views.preliminary_analysis, name='preliminary_analysis'),
    
    # Labeling
    path('datasets/<int:pk>/label/', views.dataset_label, name='dataset_label'),
    path('datasets/<int:pk>/label/save/', views.save_labels, name='save_labels'),
    
    # Training
    path('datasets/<int:pk>/train/', views.train_model, name='train_model'),
    path('datasets/<int:pk>/train/start/', views.start_training, name='start_training'),
    
    # Results & Evaluation
    path('models/<int:pk>/', views.model_detail, name='model_detail'),
    path('models/<int:pk>/evaluate/', views.model_evaluate, name='model_evaluate'),
    path('models/<int:pk>/topics/', views.model_topics, name='model_topics'),
    
    # API endpoints for async operations
    path('api/training-status/<int:pk>/', views.training_status, name='training_status'),

    # CSV Preview
    path('api/preview-csv/', views.preview_csv, name='preview_csv'),

    # Pre-analysis (before preprocessing)
    path('datasets/<int:pk>/pre-analysis/', views.pre_analysis, name='pre_analysis'),

    # Preprocessing async
    path('api/preprocessing/start/<int:pk>/', views.start_preprocessing, name='start_preprocessing'),
    path('api/preprocessing/status/<int:pk>/', views.check_preprocessing_status, name='check_preprocessing_status'),

    # Preprocessing results
    path('datasets/<int:pk>/preprocessing-results/', views.preprocessing_results, name='preprocessing_results'),

    # Model URLs
    path('models/<int:pk>/', views.model_detail, name='model_detail'),
    path('models/<int:pk>/evaluation/', views.model_evaluation, name='model_evaluation'),

    # Batch Prediction URLs
    path('datasets/<int:pk>/batch-prediction/', views.batch_prediction, name='batch_prediction'),
    path('datasets/<int:pk>/batch-prediction/start/', views.start_batch_prediction, name='start_batch_prediction'),

    # Real-time Prediction URLs
    path('prediction/', views.realtime_prediction, name='realtime_prediction'),
    path('prediction/predict/', views.predict_sentiment, name='predict_sentiment'),
    path('prediction/history/', views.prediction_history, name='prediction_history'),
    
]