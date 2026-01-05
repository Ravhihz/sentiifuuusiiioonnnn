from django.contrib import admin
from .models import (
    Dataset,
    PredictionRequest,
    Review,
    ProcessedReview,
    SentimentDictionary,
    SentimentModel,
    TopicModel,
    TrainingLog,
)


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ["name", "total_reviews", "is_preprocessed", "uploaded_at"]
    list_filter = ["is_preprocessed", "uploaded_at"]
    search_fields = ["name", "description"]


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ["id", "dataset", "label", "text_preview", "created_at"]
    list_filter = ["label", "dataset"]
    search_fields = ["text"]

    def text_preview(self, obj):
        return obj.text[:50] + "..." if len(obj.text) > 50 else obj.text

    text_preview.short_description = "Text Preview"


@admin.register(ProcessedReview)
class ProcessedReviewAdmin(admin.ModelAdmin):
    list_display = ["review", "cleaned_preview", "created_at"]
    search_fields = ["cleaned_text"]

    def cleaned_preview(self, obj):
        return obj.cleaned_text[:50] + "..." if len(obj.cleaned_text) > 50 else obj.cleaned_text

    cleaned_preview.short_description = "Cleaned Text"


@admin.register(SentimentDictionary)
class SentimentDictionaryAdmin(admin.ModelAdmin):
    list_display = ["name", "is_extended", "created_at", "updated_at"]
    list_filter = ["is_extended"]
    search_fields = ["name", "description"]


@admin.register(SentimentModel)
class SentimentModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset', 'algorithm', 'accuracy', 'f1_score', 'created_at']
    list_filter = ['algorithm', 'created_at', 'dataset']
    search_fields = ['name', 'dataset__name']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'dataset', 'algorithm')
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score')
        }),
        ('Model File', {
            'fields': ('model_file',)
        }),
        ('Parameters', {
            'fields': ('parameters',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('trained_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(TopicModel)
class TopicModelAdmin(admin.ModelAdmin):
    list_display = [
        "sentiment_model",
        "sentiment_class",
        "topic_number",
        "document_count",
        "perplexity",
    ]
    list_filter = ["sentiment_class"]


@admin.register(TrainingLog)
class TrainingLogAdmin(admin.ModelAdmin):
    list_display = ["dataset", "step", "status", "message_preview", "created_at"]
    list_filter = ["status", "created_at"]
    search_fields = ["step", "message"]

    def message_preview(self, obj):
        return obj.message[:50] + "..." if len(obj.message) > 50 else obj.message

    message_preview.short_description = "Message"

@admin.register(PredictionRequest)
class PredictionRequestAdmin(admin.ModelAdmin):
    list_display = ['id', 'predicted_sentiment', 'confidence_score', 'model', 'source', 'created_at']
    list_filter = ['predicted_sentiment', 'source', 'user_feedback', 'created_at']
    search_fields = ['text', 'cleaned_text']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Input', {
            'fields': ('text', 'cleaned_text')
        }),
        ('Prediction', {
            'fields': ('model', 'predicted_sentiment', 'confidence_score')
        }),
        ('Metadata', {
            'fields': ('source', 'ip_address')
        }),
        ('User Feedback', {
            'fields': ('user_feedback', 'corrected_sentiment'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        })
    )