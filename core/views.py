from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.db.models import Count, Q

from core.preprocessing.text_preprocessor import TextPreprocessor
from .models import Dataset, PredictionRequest, Review, ProcessedReview, SentimentModel, TopicModel
import pandas as pd
import json


def dashboard(request):
    """Enhanced dashboard with analytics"""
    from django.db.models import Count, Avg, Q
    from datetime import datetime, timedelta
    
    # Basic statistics
    total_datasets = Dataset.objects.count()
    total_reviews = Review.objects.count()
    total_models = SentimentModel.objects.count()
    total_predictions = PredictionRequest.objects.count()
    
    # Preprocessing statistics
    preprocessed_count = ProcessedReview.objects.count()
    labeled_count = Review.objects.filter(label__isnull=False).count()
    
    # Batch predictions
    batch_predicted = Review.objects.filter(predicted_label__isnull=False).count()
    
    # Recent activity (last 7 days)
    last_week = datetime.now() - timedelta(days=7)
    
    # Predictions trend (last 7 days)
    predictions_trend = []
    for i in range(7):
        date = datetime.now() - timedelta(days=6-i)
        date_str = date.strftime('%Y-%m-%d')
        count = PredictionRequest.objects.filter(
            created_at__date=date.date()
        ).count()
        predictions_trend.append({
            'date': date.strftime('%d %b'),
            'count': count
        })
    
    # Sentiment distribution (all predictions)
    sentiment_distribution = PredictionRequest.objects.values(
        'predicted_sentiment'
    ).annotate(count=Count('id'))
    
    sentiment_data = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    for item in sentiment_distribution:
        sentiment_data[item['predicted_sentiment']] = item['count']
    
    # Model performance
    top_models = SentimentModel.objects.annotate(
        prediction_count=Count('predictions')
    ).order_by('-accuracy')[:5]
    
    # Recent predictions
    recent_predictions = PredictionRequest.objects.select_related('model').order_by('-created_at')[:5]
    
    # Recent datasets
    recent_datasets = Dataset.objects.order_by('-uploaded_at')[:5]
    
    # Recent models
    recent_models = SentimentModel.objects.select_related('dataset').order_by('-created_at')[:3]
    
    # Average confidence
    avg_confidence = PredictionRequest.objects.aggregate(
        avg=Avg('confidence_score')
    )['avg'] or 0
    
    # Today's predictions
    today_predictions = PredictionRequest.objects.filter(
        created_at__date=datetime.now().date()
    ).count()
    
    context = {
        # Basic stats
        'total_datasets': total_datasets,
        'total_reviews': total_reviews,
        'total_models': total_models,
        'total_predictions': total_predictions,
        'preprocessed_count': preprocessed_count,
        'labeled_count': labeled_count,
        'batch_predicted': batch_predicted,
        'today_predictions': today_predictions,
        
        # Analytics
        'predictions_trend': predictions_trend,
        'sentiment_data': sentiment_data,
        'avg_confidence': avg_confidence,
        
        # Top performers
        'top_models': top_models,
        
        # Recent activity
        'recent_predictions': recent_predictions,
        'recent_datasets': recent_datasets,
        'recent_models': recent_models,
    }
    
    return render(request, 'dashboard.html', context)


def dataset_list(request):
    """List all datasets"""
    datasets = Dataset.objects.all().annotate(
        labeled_count=Count('reviews', filter=Q(reviews__label__isnull=False))
    )
    
    context = {
        'datasets': datasets,
    }
    
    return render(request, 'datasets/list.html', context)


def dataset_upload(request):
    """Upload new dataset"""
    if request.method == 'POST':
        try:
            # Get form data
            name = request.POST.get('name')
            description = request.POST.get('description', '')
            text_column = request.POST.get('text_column')
            label_column = request.POST.get('label_column', '')
            
            # Validation
            if not name:
                messages.error(request, 'Please provide a dataset name.')
                return redirect('core:dataset_upload')
            
            if not text_column:
                messages.error(request, 'Please select a text column from the dropdown.')
                return redirect('core:dataset_upload')
            
            # Check if file exists in session (from preview)
            if 'temp_csv_path' not in request.session:
                messages.error(request, 'Please upload and preview CSV file first.')
                return redirect('core:dataset_upload')
            
            import os
            from django.conf import settings
            
            temp_path = request.session['temp_csv_path']
            full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
            
            if not os.path.exists(full_path):
                messages.error(request, 'Temporary file not found. Please upload again.')
                del request.session['temp_csv_path']
                return redirect('core:dataset_upload')
            
            # Read CSV
            df = pd.read_csv(full_path)
            
            # Validate column exists
            if text_column not in df.columns:
                messages.error(request, f'Column "{text_column}" not found in CSV. Available columns: {", ".join(df.columns)}')
                return redirect('core:dataset_upload')
            
            # Create dataset (move temp file to permanent location)
            dataset = Dataset.objects.create(
                name=name,
                description=description,
                uploaded_by=request.user if request.user.is_authenticated else None
            )
            
            # Copy file to dataset location
            import shutil
            dataset_file_path = f'datasets/{dataset.pk}_{os.path.basename(temp_path)}'
            full_dataset_path = os.path.join(settings.MEDIA_ROOT, dataset_file_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_dataset_path), exist_ok=True)
            shutil.copy(full_path, full_dataset_path)
            
            dataset.file = dataset_file_path # type: ignore
            dataset.save()
            
            # Import reviews
            reviews_created = 0
            for idx, row in df.iterrows():
                text = str(row[text_column]).strip()
                
                if not text or text == 'nan':
                    continue
                
                label = None
                if label_column and label_column in df.columns:
                    label_value = str(row[label_column]).lower().strip()
                    if label_value in ['positive', 'negative', 'neutral']:
                        label = label_value
                
                Review.objects.create(
                    dataset=dataset,
                    text=text,
                    label=label
                )
                reviews_created += 1
            
            # Update dataset stats
            dataset.total_reviews = reviews_created
            dataset.save()
            
            # Cleanup temp file
            try:
                os.remove(full_path)
                del request.session['temp_csv_path']
            except:
                pass
            
            messages.success(
                request, 
                f'Dataset "{name}" uploaded successfully! {reviews_created} reviews imported.'
            )
            return redirect('core:dataset_detail', pk=dataset.pk)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messages.error(request, f'Error uploading dataset: {str(e)}')
            return redirect('core:dataset_upload')
    
    return render(request, 'datasets/upload.html')


def dataset_detail(request, pk):
    """Dataset detail view"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    # Get statistics
    total_reviews = dataset.reviews.count() # type: ignore
    labeled_reviews = dataset.reviews.filter(label__isnull=False).count() # type: ignore
    preprocessed_reviews = ProcessedReview.objects.filter(review__dataset=dataset).count()
    
    # Get label distribution
    label_distribution = dataset.reviews.values('label').annotate( # type: ignore
        count=Count('id')
    ).order_by('-count')
    
    # Get recent reviews with processed data
    recent_reviews_qs = dataset.reviews.all()[:10] # type: ignore
    
    # Attach processed data to each review
    recent_reviews = []
    for review in recent_reviews_qs:
        review_data = {
            'id': review.id,
            'text': review.text,
            'label': review.label,
            'cleaned_text': None,
            'token_count': 0,
        }
        
        try:
            processed = ProcessedReview.objects.get(review=review)
            review_data['cleaned_text'] = processed.cleaned_text
            tokens = json.loads(processed.tokenized_text)
            review_data['token_count'] = len(tokens)
        except ProcessedReview.DoesNotExist:
            pass
        
        recent_reviews.append(review_data)
    
    # Get trained models
    models = dataset.models.all() # type: ignore
    trained_models = models.count()  # NEW!
    latest_model = models.order_by('-created_at').first()  # NEW!
    
    context = {
        'dataset': dataset,
        'total_reviews': total_reviews,
        'labeled_reviews': labeled_reviews,
        'preprocessed_reviews': preprocessed_reviews,
        'label_distribution': label_distribution,
        'recent_reviews': recent_reviews,
        'models': models,
        'trained_models': trained_models,  # NEW!
        'latest_model': latest_model,  # NEW!
    }
    
    return render(request, 'datasets/detail.html', context)


def dataset_delete(request, pk):
    """Delete dataset"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    if request.method == 'POST':
        name = dataset.name
        dataset.delete()
        messages.success(request, f'Dataset "{name}" deleted successfully.')
        return redirect('core:dataset_list')
    
    return render(request, 'datasets/delete_confirm.html', {'dataset': dataset})


def dataset_preprocess(request, pk):
    """Preprocess dataset page"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    context = {
        'dataset': dataset,
    }
    return render(request, 'datasets/preprocess.html', context)


@require_http_methods(["POST"])
def start_preprocessing(request, pk):
    """Start preprocessing - SYNCHRONOUS for Windows compatibility"""
    try:
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Get options
        use_stemming = request.POST.get('use_stemming', 'false').lower() == 'true'
        
        print(f"[PREPROCESSING] Starting for dataset {pk}")
        print(f"[PREPROCESSING] Use stemming: {use_stemming}")
        print(f"[PREPROCESSING] Total reviews: {dataset.reviews.count()}") # type: ignore
        
        # Import service
        from .services import PreprocessingService
        
        # Run preprocessing SYNCHRONOUSLY
        service = PreprocessingService()
        stats = service.preprocess_dataset(dataset, use_stemming=use_stemming)
        
        print(f"[PREPROCESSING] Complete! Stats: {stats}")
        
        return JsonResponse({
            'status': 'completed',
            'stats': stats
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[PREPROCESSING ERROR] {error_trace}")
        
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'traceback': error_trace
        }, status=500)


@require_http_methods(["GET"])
def check_preprocessing_status(request, pk):
    """Check if preprocessing is complete"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    return JsonResponse({
        'is_preprocessed': dataset.is_preprocessed,
        'stats': dataset.preprocessing_stats if dataset.is_preprocessed else None
    })


def pre_analysis(request, pk):
    """Preliminary analysis BEFORE preprocessing (Raw Data Analysis)"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    try:
        from .services import AnalysisService
        
        service = AnalysisService()
        raw_analysis = service.get_raw_data_analysis(dataset)
        
        context = {
            'dataset': dataset,
            'raw_analysis': raw_analysis,
        }
        
        return render(request, 'datasets/pre_analysis.html', context)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        messages.error(request, f'Error analyzing dataset: {str(e)}')
        return redirect('core:dataset_detail', pk=pk)

    
def preliminary_analysis(request, pk):
    """Preliminary analysis AFTER preprocessing"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    if not dataset.is_preprocessed:
        messages.warning(request, 'Please preprocess the dataset first.')
        return redirect('core:dataset_preprocess', pk=pk)
    
    try:
        from .services import AnalysisService
        
        service = AnalysisService()
        
        # Get statistics
        basic_stats = service.get_dataset_statistics(dataset)
        preprocessed_stats = service.get_preprocessed_statistics(dataset)
        label_dist = service.get_label_distribution(dataset)
        imbalance_check = service.check_imbalance(dataset)
        quality_report = service.get_data_quality_report(dataset)
        
        # Format label distribution for template
        label_table = []
        for label, count in label_dist['counts'].items():
            percentage = label_dist['percentages'].get(label, 0)
            label_table.append({
                'label': label if label else 'unlabeled',
                'count': count,
                'percentage': percentage
            })
        
        context = {
            'dataset': dataset,
            'basic_stats': basic_stats,
            'preprocessed_stats': preprocessed_stats,
            'label_distribution': label_dist,
            'label_table': label_table,
            'imbalance_check': imbalance_check,
            'quality_report': quality_report,
        }
        
        return render(request, 'datasets/analysis.html', context)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        messages.error(request, f'Error analyzing dataset: {str(e)}')
        return redirect('core:dataset_detail', pk=pk)


def preprocessing_results(request, pk):
    """Show detailed preprocessing results with step-by-step breakdown"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    if not dataset.is_preprocessed:
        messages.warning(request, 'Please preprocess the dataset first.')
        return redirect('core:dataset_preprocess', pk=pk)
    
    # Get paginated reviews
    from django.core.paginator import Paginator
    
    page_number = request.GET.get('page', 1)
    per_page = 10  # Show 10 reviews per page
    
    # Get all reviews with their processed versions
    reviews = Review.objects.filter(dataset=dataset).order_by('id')
    paginator = Paginator(reviews, per_page)
    page_obj = paginator.get_page(page_number)
    
    # Build detailed results for each review in current page
    detailed_results = []
    for review in page_obj:
        try:
            processed = ProcessedReview.objects.get(review=review)
            
            # Get step-by-step preprocessing
            preprocessor = TextPreprocessor()
            
            # Step 0: Original
            original = review.text
            
            # Step 1: Lowercase
            step1 = original.lower()
            
            # Step 2: Remove URLs and mentions
            import re
            step2 = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', step1)
            step2 = re.sub(r'@\w+', '', step2)
            
            # Step 3: Normalize slang
            step3_words = step2.split()
            step3_normalized = [preprocessor.slang_dict.get(word, word) for word in step3_words]
            step3 = ' '.join(step3_normalized)
            
            # Step 4: Remove stopwords
            step4_words = step3.split()
            step4_filtered = [word for word in step4_words if word not in preprocessor.stopwords]
            step4 = ' '.join(step4_filtered)
            
            # Step 5: Stemming (if used)
            tokens = json.loads(processed.tokenized_text)
            step5 = ' '.join(tokens)
            
            # Count tokens at each step
            detailed_results.append({
                'review': review,
                'original': original,
                'original_length': len(original.split()),
                'step1_lowercase': step1,
                'step2_clean': step2,
                'step3_normalized': step3,
                'step4_stopwords_removed': step4,
                'step4_length': len(step4_filtered),
                'step5_stemmed': step5,
                'final_length': len(tokens),
                'tokens_removed': len(original.split()) - len(tokens),
            })
        except ProcessedReview.DoesNotExist:
            continue
    
    # Get overall statistics
    stats = dataset.preprocessing_stats or {}
    
    context = {
        'dataset': dataset,
        'page_obj': page_obj,
        'detailed_results': detailed_results,
        'stats': stats,
        'total_reviews': reviews.count(),
    }
    
    return render(request, 'datasets/preprocessing_results.html', context)


def dataset_label(request, pk):
    """Manual labeling interface"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    if not dataset.is_preprocessed:
        messages.warning(request, 'Please preprocess the dataset first.')
        return redirect('core:dataset_preprocess', pk=pk)
    
    # Get IDs of reviews that have been preprocessed (valid reviews only)
    processed_review_ids = ProcessedReview.objects.filter(
        review__dataset=dataset
    ).values_list('review_id', flat=True)
    
    # Get labeling statistics (ONLY for processed reviews)
    total_reviews = dataset.reviews.filter(id__in=processed_review_ids).count() # type: ignore
    labeled_count = dataset.reviews.filter(id__in=processed_review_ids, label__isnull=False).count() # type: ignore
    unlabeled_count = total_reviews - labeled_count
    
    # Dynamic minimum label requirement
    if total_reviews >= 5000:
        min_labels_required = 500
        dataset_size_category = 'large'
    else:
        min_labels_required = 100
        dataset_size_category = 'small'
    
    # Get filter from query params
    filter_type = request.GET.get('filter', 'unlabeled')
    page_number = request.GET.get('page', 1)
    
    # Filter reviews based on type (ONLY processed reviews!)
    if filter_type == 'unlabeled':
        reviews = dataset.reviews.filter( # type: ignore
            id__in=processed_review_ids,
            label__isnull=True
        ).order_by('id')
    elif filter_type == 'all':
        reviews = dataset.reviews.filter( # type: ignore
            id__in=processed_review_ids
        ).order_by('id')
    elif filter_type in ['positive', 'negative', 'neutral']:
        reviews = dataset.reviews.filter( # type: ignore
            id__in=processed_review_ids,
            label=filter_type
        ).order_by('id')
    else:
        reviews = dataset.reviews.filter( # type: ignore
            id__in=processed_review_ids,
            label__isnull=True
        ).order_by('id')
    
    # Pagination
    from django.core.paginator import Paginator
    paginator = Paginator(reviews, 1)
    page_obj = paginator.get_page(page_number)
    
    # Get current review with processed text
    current_review = None
    cleaned_text = None
    token_count = 0
    
    if page_obj.object_list:
        current_review = page_obj.object_list[0]
        
        # Get cleaned text (guaranteed to exist now!)
        try:
            processed = ProcessedReview.objects.get(review=current_review)
            cleaned_text = processed.cleaned_text
            tokens = json.loads(processed.tokenized_text)
            token_count = len(tokens)
        except ProcessedReview.DoesNotExist:
            # This shouldn't happen anymore, but just in case
            print(f"WARNING: Review {current_review.id} has no ProcessedReview!")
    
    # Calculate progress
    progress_percentage = (labeled_count / total_reviews * 100) if total_reviews > 0 else 0
    can_train = labeled_count >= min_labels_required
    
    context = {
        'dataset': dataset,
        'current_review': current_review,
        'cleaned_text': cleaned_text,
        'token_count': token_count,
        'page_obj': page_obj,
        'filter_type': filter_type,
        'total_reviews': total_reviews,
        'labeled_count': labeled_count,
        'unlabeled_count': unlabeled_count,
        'progress_percentage': progress_percentage,
        'min_labels_required': min_labels_required,
        'dataset_size_category': dataset_size_category,
        'can_train': can_train,
    }
    
    return render(request, 'datasets/label.html', context)


@require_http_methods(["POST"])
def save_labels(request, pk):
    """Save single label via AJAX"""
    try:
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Get data from POST
        review_id = request.POST.get('review_id')
        label = request.POST.get('label')
        
        # Validate
        if not review_id or not label:
            return JsonResponse({'error': 'Missing review_id or label'}, status=400)
        
        if label not in ['positive', 'negative', 'neutral']:
            return JsonResponse({'error': 'Invalid label'}, status=400)
        
        # Get review
        review = get_object_or_404(Review, pk=review_id, dataset=dataset)
        
        # Save label
        review.label = label
        review.save()
        
        # Get updated statistics
        total_reviews = dataset.reviews.count() # type: ignore
        labeled_count = dataset.reviews.filter(label__isnull=False).count() # type: ignore
        progress_percentage = (labeled_count / total_reviews * 100) if total_reviews > 0 else 0
        
        return JsonResponse({
            'success': True,
            'message': f'Review labeled as {label}',
            'labeled_count': labeled_count,
            'progress_percentage': round(progress_percentage, 1)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


def train_model(request, pk):
    """Train model interface"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    if not dataset.is_preprocessed:
        messages.warning(request, 'Please preprocess the dataset first.')
        return redirect('core:dataset_preprocess', pk=pk)
    
    # Check if enough labels
    labeled_count = dataset.reviews.filter(label__isnull=False).count() # type: ignore
    
    # Dynamic minimum
    if dataset.reviews.count() >= 5000: # type: ignore
        min_required = 500
    else:
        min_required = 100
    
    if labeled_count < min_required:
        messages.warning(request, f'You need at least {min_required} labeled reviews. Currently: {labeled_count}')
        return redirect('core:dataset_label', pk=pk)
    
    # Get label distribution
    from django.db.models import Count
    label_dist = dataset.reviews.filter(label__isnull=False).values('label').annotate(count=Count('id')) # type: ignore
    
    # Get existing models
    existing_models = dataset.models.all().order_by('-created_at') # type: ignore
    
    context = {
        'dataset': dataset,
        'labeled_count': labeled_count,
        'label_distribution': label_dist,
        'existing_models': existing_models,
    }
    
    return render(request, 'datasets/train.html', context)


@require_http_methods(["POST"])
def start_training(request, pk):
    """Start training"""
    try:
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Get parameters
        test_size = float(request.POST.get('test_size', 0.2))
        n_components = int(request.POST.get('n_components', 100))
        kernel = request.POST.get('kernel', 'rbf')
        C = float(request.POST.get('C', 1.0))
        max_iter = int(request.POST.get('max_iter', 1000))
        apply_smote = request.POST.get('apply_smote', 'true').lower() == 'true'
        extract_lda = request.POST.get('extract_lda', 'true').lower() == 'true'  # NEW!
        n_topics = int(request.POST.get('n_topics', 5))  # NEW!
        
        print(f"[TRAINING] Starting with params:")
        print(f"  test_size={test_size}, n_components={n_components}")
        print(f"  kernel={kernel}, C={C}, max_iter={max_iter}")
        print(f"  apply_smote={apply_smote}, extract_lda={extract_lda}, n_topics={n_topics}")
        
        # Import service
        from .services.training_service import TrainingService
        
        # Train model SYNCHRONOUSLY
        service = TrainingService()
        results = service.train_model(
            dataset,
            test_size=test_size,
            n_components=n_components,
            kernel=kernel,
            C=C,
            max_iter=max_iter,
            apply_smote=apply_smote,
            extract_lda_topics=extract_lda,  # NEW!
            n_topics=n_topics  # NEW!
        )
        
        print(f"[TRAINING] Complete! Model ID: {results['model_id']}")
        
        return JsonResponse({
            'status': 'completed',
            'results': results
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[TRAINING ERROR] {error_trace}")
        
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'traceback': error_trace
        }, status=500)


def model_detail(request, pk):
    """Model detail page"""
    model = get_object_or_404(SentimentModel, pk=pk)
    
    # Get training details from parameters
    parameters = model.parameters
    
    context = {
        'model': model,
        'parameters': parameters,
    }
    
    return render(request, 'models/detail.html', context)


def model_evaluate(request, pk):
    """Model evaluation"""
    model = get_object_or_404(SentimentModel, pk=pk)
    messages.info(request, 'Model evaluation feature will be implemented in next section.')
    return redirect('core:dataset_detail', pk=model.dataset.pk)


def model_topics(request, pk):
    """Model topics"""
    model = get_object_or_404(SentimentModel, pk=pk)
    messages.info(request, 'Model topics feature will be implemented in next section.')
    return redirect('core:dataset_detail', pk=model.dataset.pk)


@require_http_methods(["POST"])
def preview_csv(request):
    """Preview CSV columns and sample data"""
    try:
        uploaded_file = request.FILES.get('file')
        
        if not uploaded_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        # Save to temporary location
        import os
        from django.conf import settings
        
        # Create temp directory
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save file
        import uuid
        filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        temp_path = os.path.join('temp', filename)
        full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
        
        with open(full_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Read CSV
        df = pd.read_csv(full_path)
        
        # Store temp path in session
        request.session['temp_csv_path'] = temp_path
        
        # Get columns
        columns = df.columns.tolist()
        
        # Get first 5 rows as preview
        preview_data = df.head(5).to_dict('records')
        
        # Convert to serializable format
        for row in preview_data:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = ''
                else:
                    row[key] = str(value)[:100]  # Limit length for preview
        
        return JsonResponse({
            'columns': columns,
            'preview': preview_data,
            'total_rows': len(df),
            'success': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)


@require_http_methods(["GET"])
def training_status(request, pk):
    """Get training status"""
    return JsonResponse({
        'status': 'idle',
        'progress': 0,
        'message': 'Training not started'
    })

def model_evaluation(request, pk):
    """Model evaluation page with visualizations"""
    model = get_object_or_404(SentimentModel, pk=pk)
    
    # Get parameters
    parameters = model.parameters
    
    # Get predictions
    y_test = parameters.get('y_test', [])
    y_pred = parameters.get('y_test_pred', [])
    
    # Calculate correct/incorrect
    correct_count = sum(1 for t, p in zip(y_test, y_pred) if t == p)
    incorrect_count = len(y_test) - correct_count
    
    # Get topics if available
    topics = parameters.get('topics', {})
    
    context = {
        'model': model,
        'parameters': parameters,
        'topics': topics,
        'y_test': y_test,
        'y_pred': y_pred,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
    }
    
    return render(request, 'models/evaluation.html', context)

def batch_prediction(request, pk):
    """Batch prediction interface"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    # Get available models for this dataset
    available_models = SentimentModel.objects.filter(dataset=dataset).order_by('-created_at')
    
    # Get prediction summary
    from .services.batch_prediction_service import BatchPredictionService
    service = BatchPredictionService()
    summary = service.get_prediction_summary(dataset)
    
    context = {
        'dataset': dataset,
        'available_models': available_models,
        'summary': summary,
    }
    
    return render(request, 'datasets/batch_prediction.html', context)


@require_http_methods(["POST"])
def start_batch_prediction(request, pk):
    """Start batch prediction process"""
    try:
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Get parameters
        model_id = int(request.POST.get('model_id'))
        overwrite = request.POST.get('overwrite', 'false').lower() == 'true'
        
        model = get_object_or_404(SentimentModel, pk=model_id)
        
        print(f"[BATCH PREDICTION] Starting...")
        print(f"  Dataset: {dataset.name}")
        print(f"  Model: {model.name}")
        print(f"  Overwrite: {overwrite}")
        
        # Run batch prediction
        from .services.batch_prediction_service import BatchPredictionService
        service = BatchPredictionService()
        
        results = service.predict_batch(
            dataset=dataset,
            model=model,
            overwrite_existing=overwrite
        )
        
        print(f"[BATCH PREDICTION] Complete!")
        
        return JsonResponse({
            'status': 'completed',
            'results': results
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[BATCH PREDICTION ERROR] {error_trace}")
        
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'traceback': error_trace
        }, status=500)

def realtime_prediction(request):
    """Real-time prediction interface"""
    # Get all available models
    available_models = SentimentModel.objects.all().order_by('-created_at')
    
    # Get recent predictions
    from .services.realtime_prediction_service import RealtimePredictionService
    service = RealtimePredictionService()
    
    recent_predictions = PredictionRequest.objects.all()[:10]
    
    context = {
        'available_models': available_models,
        'recent_predictions': recent_predictions,
    }
    
    return render(request, 'prediction/realtime.html', context)


@require_http_methods(["POST"])
def predict_sentiment(request):
    """Predict sentiment for text"""
    try:
        # Get parameters
        text = request.POST.get('text', '').strip()
        model_id = int(request.POST.get('model_id'))
        
        if not text:
            return JsonResponse({
                'status': 'error',
                'message': 'Text is required'
            }, status=400)
        
        if len(text) < 10:
            return JsonResponse({
                'status': 'error',
                'message': 'Text too short (minimum 10 characters)'
            }, status=400)
        
        model = get_object_or_404(SentimentModel, pk=model_id)
        
        print(f"[PREDICT] Starting prediction...")
        print(f"  Text: {text[:100]}...")
        print(f"  Model: {model.name}")
        
        # Get client IP
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip_address = x_forwarded_for.split(',')[0]
        else:
            ip_address = request.META.get('REMOTE_ADDR')
        
        # Predict
        from .services.realtime_prediction_service import RealtimePredictionService
        service = RealtimePredictionService()
        
        results = service.predict(
            text=text,
            model=model,
            source='web',
            ip_address=ip_address,
            save_to_db=True
        )
        
        print(f"[PREDICT] Complete! Sentiment: {results['sentiment']}")
        
        return JsonResponse({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[PREDICT ERROR] {error_trace}")
        
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'traceback': error_trace
        }, status=500)


def prediction_history(request):
    """View prediction history"""
    predictions = PredictionRequest.objects.all().order_by('-created_at')[:100]
    
    # Get statistics
    total_predictions = PredictionRequest.objects.count()
    
    sentiment_counts = PredictionRequest.objects.values('predicted_sentiment').annotate(
        count=Count('id')
    )
    
    avg_confidence = PredictionRequest.objects.aggregate(
        avg=Avg('confidence_score') # type: ignore
    )['avg'] or 0
    
    context = {
        'predictions': predictions,
        'total_predictions': total_predictions,
        'sentiment_counts': sentiment_counts,
        'avg_confidence': avg_confidence,
    }
    
    return render(request, 'prediction/history.html', context)