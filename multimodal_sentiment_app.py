# Run this cell to create the complete multimodal_sentiment_app.py file

import streamlit as st
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    CLIPProcessor, CLIPModel, pipeline
)
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
import io
import base64

class MultimodalSentimentAnalyzer:
    def __init__(self):
        self.setup_models()
        
    @st.cache_resource
    def setup_models(_self):
        """Initialize pre-trained models"""
        try:
            # Text sentiment model (BERT-based)
            _self.text_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            _self.text_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            # CLIP model for image-text understanding
            _self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # Emotion detection pipeline
            _self.emotion_pipeline = pipeline("text-classification", 
                                            model="j-hartmann/emotion-english-distilroberta-base", 
                                            return_all_scores=True)
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\\w+|#\\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of text using BERT-based model"""
        try:
            # Preprocess text
            clean_text = self.preprocess_text(text)
            
            # Tokenize and predict
            inputs = self.text_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Map labels (RoBERTa sentiment model uses LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive)
            labels = ['negative', 'neutral', 'positive']
            scores = predictions[0].numpy()
            
            result = {
                'text': text,
                'sentiment': labels[np.argmax(scores)],
                'confidence': float(np.max(scores)),
                'scores': {
                    'negative': float(scores[0]),
                    'neutral': float(scores[1]),
                    'positive': float(scores[2])
                }
            }
            
            # Add emotion analysis
            emotions = self.emotion_pipeline(clean_text)[0]
            result['emotions'] = {emotion['label']: emotion['score'] for emotion in emotions}
            
            # TextBlob for additional insights
            blob = TextBlob(clean_text)
            result['polarity'] = blob.sentiment.polarity
            result['subjectivity'] = blob.sentiment.subjectivity
            
            return result
            
        except Exception as e:
            st.error(f"Error in text analysis: {e}")
            return None
    
    def analyze_image_sentiment(self, image):
        """Analyze sentiment of image using CLIP and emotion detection"""
        try:
            # Convert PIL image to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Emotion prompts for CLIP
            emotion_prompts = [
                "a happy person", "a sad person", "an angry person",
                "a surprised person", "a disgusted person", "a fearful person",
                "a neutral expression", "positive emotions", "negative emotions"
            ]
            
            # Process image and text prompts
            inputs = self.clip_processor(text=emotion_prompts, images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # Map probabilities to emotions
            emotion_scores = probs[0].numpy()
            emotions = ['happy', 'sad', 'angry', 'surprised', 'disgusted', 'fearful', 'neutral', 'positive', 'negative']
            
            # Calculate overall sentiment
            positive_score = emotion_scores[0] + emotion_scores[3] + emotion_scores[7]  # happy + surprised + positive
            negative_score = emotion_scores[1] + emotion_scores[2] + emotion_scores[4] + emotion_scores[5] + emotion_scores[8]  # sad + angry + disgusted + fearful + negative
            neutral_score = emotion_scores[6]  # neutral
            
            # Normalize scores
            total = positive_score + negative_score + neutral_score
            if total > 0:
                positive_score /= total
                negative_score /= total
                neutral_score /= total
            
            # Determine primary sentiment
            sentiment_scores = [negative_score, neutral_score, positive_score]
            sentiment_labels = ['negative', 'neutral', 'positive']
            primary_sentiment = sentiment_labels[np.argmax(sentiment_scores)]
            
            # Color analysis
            img_array = np.array(image)
            avg_color = np.mean(img_array, axis=(0, 1))
            brightness = np.mean(avg_color)
            
            result = {
                'sentiment': primary_sentiment,
                'confidence': float(np.max(sentiment_scores)),
                'scores': {
                    'negative': float(negative_score),
                    'neutral': float(neutral_score),
                    'positive': float(positive_score)
                },
                'detailed_emotions': {emotions[i]: float(emotion_scores[i]) for i in range(len(emotions))},
                'color_analysis': {
                    'average_rgb': avg_color.tolist(),
                    'brightness': float(brightness),
                    'warmth': 'warm' if avg_color[0] > avg_color[2] else 'cool'
                }
            }
            
            return result
            
        except Exception as e:
            st.error(f"Error in image analysis: {e}")
            return None
    
    def combine_analyses(self, text_result, image_result, text_weight=0.6, image_weight=0.4):
        """Combine text and image sentiment analyses"""
        try:
            # Weighted combination of scores
            combined_scores = {
                'negative': text_result['scores']['negative'] * text_weight + image_result['scores']['negative'] * image_weight,
                'neutral': text_result['scores']['neutral'] * text_weight + image_result['scores']['neutral'] * image_weight,
                'positive': text_result['scores']['positive'] * text_weight + image_result['scores']['positive'] * image_weight
            }
            
            # Determine overall sentiment
            sentiment_labels = ['negative', 'neutral', 'positive']
            primary_sentiment = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[primary_sentiment]
            
            # Agreement analysis
            text_sentiment = text_result['sentiment']
            image_sentiment = image_result['sentiment']
            agreement = text_sentiment == image_sentiment
            
            result = {
                'sentiment': primary_sentiment,
                'confidence': float(confidence),
                'scores': combined_scores,
                'agreement': agreement,
                'text_weight': text_weight,
                'image_weight': image_weight,
                'individual_results': {
                    'text': text_result,
                    'image': image_result
                }
            }
            
            return result
            
        except Exception as e:
            st.error(f"Error in combining analyses: {e}")
            return None

def create_sentiment_visualization(results, analysis_type):
    """Create visualizations for sentiment analysis results"""
    
    if analysis_type == 'text' and 'emotions' in results:
        # Emotion distribution pie chart
        emotions = results['emotions']
        fig_emotions = px.pie(
            values=list(emotions.values()),
            names=list(emotions.keys()),
            title="Emotion Distribution"
        )
        st.plotly_chart(fig_emotions)
    
    # Sentiment scores bar chart
    scores = results['scores']
    fig_sentiment = go.Figure(data=[
        go.Bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            marker_color=['red', 'gray', 'green']
        )
    ])
    fig_sentiment.update_layout(
        title=f"Sentiment Scores - {analysis_type.title()} Analysis",
        xaxis_title="Sentiment",
        yaxis_title="Score"
    )
    st.plotly_chart(fig_sentiment)

def display_text_results(result):
    """Display text analysis results"""
    st.subheader(" Text Sentiment Analysis")
    
    # Main sentiment
    sentiment = result['sentiment']
    confidence = result['confidence']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sentiment", sentiment.title(), f"{confidence:.1%} confidence")
    with col2:
        st.metric("Polarity", f"{result['polarity']:.2f}", "(-1 to +1)")
    with col3:
        st.metric("Subjectivity", f"{result['subjectivity']:.2f}", "(0 to 1)")
    
    # Sentiment scores
    create_sentiment_visualization(result, 'text')
    
    # Detailed metrics
    with st.expander(" Detailed Metrics"):
        st.json(result)

def display_image_results(result):
    """Display image analysis results"""
    st.subheader(" Image Sentiment Analysis")
    
    # Main sentiment
    sentiment = result['sentiment']
    confidence = result['confidence']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sentiment", sentiment.title(), f"{confidence:.1%} confidence")
    with col2:
        color_analysis = result['color_analysis']
        st.metric("Brightness", f"{color_analysis['brightness']:.0f}", f"{color_analysis['warmth']} tones")
    with col3:
        dominant_emotion = max(result['detailed_emotions'], key=result['detailed_emotions'].get)
        st.metric("Dominant Emotion", dominant_emotion.title())
    
    # Visualizations
    create_sentiment_visualization(result, 'image')
    
    # Detailed emotions
    with st.expander("😊 Detailed Emotions"):
        emotions_df = pd.DataFrame([
            {"Emotion": k.title(), "Score": f"{v:.3f}"} 
            for k, v in result['detailed_emotions'].items()
        ])
        st.dataframe(emotions_df, use_container_width=True)

def display_combined_results(result):
    """Display combined analysis results"""
    st.subheader(" Combined Multimodal Analysis")
    
    # Main metrics
    sentiment = result['sentiment']
    confidence = result['confidence']
    agreement = result['agreement']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Sentiment", sentiment.title(), f"{confidence:.1%} confidence")
    with col2:
        st.metric("Modality Agreement", " Yes" if agreement else " No")
    with col3:
        st.metric("Combination", f"Text: {result['text_weight']:.0%}, Image: {result['image_weight']:.0%}")
    
    # Combined scores
    create_sentiment_visualization(result, 'combined')
    
    # Agreement analysis
    with st.expander("🤝 Agreement Analysis"):
        text_sent = result['individual_results']['text']['sentiment']
        image_sent = result['individual_results']['image']['sentiment']
        
        if agreement:
            st.success(f"Both text and image analysis agree on **{sentiment}** sentiment")
        else:
            st.warning(f" Disagreement: Text shows **{text_sent}** while image shows **{image_sent}**")
        
        st.write("This disagreement could indicate:")
        st.write("- Sarcasm or irony in text")
        st.write("- Mismatched emotional content")
        st.write("- Contextual nuances requiring human interpretation")

def main():
    st.set_page_config(page_title="Multimodal Sentiment Analysis", page_icon="🎭", layout="wide")
    
    st.title(" Multimodal Sentiment Analysis")
    st.markdown("Advanced sentiment analysis using state-of-the-art AI models for text and images")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("Loading AI models... This may take a moment."):
            st.session_state.analyzer = MultimodalSentimentAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Text Only", "Image Only", "Multimodal (Text + Image)"]
    )
    
    if analysis_mode == "Multimodal (Text + Image)":
        text_weight = st.sidebar.slider("Text Weight", 0.0, 1.0, 0.6, 0.1)
        image_weight = 1.0 - text_weight
        st.sidebar.write(f"Image Weight: {image_weight:.1f}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        if analysis_mode in ["Text Only", "Multimodal (Text + Image)"]:
            st.subheader("Text Input")
            text_input = st.text_area(
                "Enter text to analyze:",
                placeholder="Type your message here... Try: 'I absolutely love this amazing product!' or 'This service was terrible and disappointing.'",
                height=100
            )
        
        # Image input
        if analysis_mode in ["Image Only", "Multimodal (Text + Image)"]:
            st.subheader("Image Input")
            uploaded_image = st.file_uploader(
                "Upload an image:",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image containing faces or emotional content"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("Quick Test Examples")
        
        if st.button("😊 Positive Text Example"):
            st.session_state.sample_text = "I absolutely love this fantastic product! The quality is amazing and the customer service exceeded my expectations. Highly recommended!"
        
        if st.button("😢 Negative Text Example"):
            st.session_state.sample_text = "This product is terrible and completely disappointing. The quality is poor and the customer service was unhelpful and rude."
        
        if st.button("😐 Neutral Text Example"):
            st.session_state.sample_text = "The product arrived on time. It functions as described in the documentation. The packaging was adequate."
        
        # Use sample text if set
        if 'sample_text' in st.session_state:
            text_input = st.session_state.sample_text
            del st.session_state.sample_text
    
    # Analysis button
    if st.button("Analyze Sentiment", type="primary"):
        
        # Validate inputs
        has_text = analysis_mode in ["Text Only", "Multimodal (Text + Image)"] and 'text_input' in locals() and text_input.strip()
        has_image = analysis_mode in ["Image Only", "Multimodal (Text + Image)"] and 'uploaded_image' in locals() and uploaded_image is not None
        
        if not has_text and analysis_mode in ["Text Only", "Multimodal (Text + Image)"]:
            st.error("Please enter some text to analyze.")
            return
        
        if not has_image and analysis_mode in ["Image Only", "Multimodal (Text + Image)"]:
            st.error("Please upload an image to analyze.")
            return
        
        # Perform analysis
        with st.spinner("Analyzing sentiment..."):
            results = {}
            
            # Text analysis
            if has_text:
                with st.status("Analyzing text sentiment..."):
                    text_result = analyzer.analyze_text_sentiment(text_input)
                    if text_result:
                        results['text'] = text_result
            
            # Image analysis
            if has_image:
                with st.status("Analyzing image sentiment..."):
                    image_result = analyzer.analyze_image_sentiment(image)
                    if image_result:
                        results['image'] = image_result
            
            # Combined analysis
            if analysis_mode == "Multimodal (Text + Image)" and 'text' in results and 'image' in results:
                with st.status("Combining analyses..."):
                    combined_result = analyzer.combine_analyses(
                        results['text'], 
                        results['image'], 
                        text_weight, 
                        image_weight
                    )
                    if combined_result:
                        results['combined'] = combined_result
        
        # Display results
        if results:
            st.success("Analysis completed!")
            
            # Results tabs
            if analysis_mode == "Text Only":
                display_text_results(results['text'])
            elif analysis_mode == "Image Only":
                display_image_results(results['image'])
            else:  # Multimodal
                tab1, tab2, tab3 = st.tabs([" Combined Analysis", " Text Analysis", "Image Analysis"])
                
                with tab1:
                    display_combined_results(results['combined'])
                
                with tab2:
                    display_text_results(results['text'])
                
                with tab3:
                    display_image_results(results['image'])

    # Info section
    st.divider()
    st.markdown("""
    **🔧 Technical Implementation:**
    - **Text Analysis:** RoBERTa model for sentiment + DistilRoBERTa for emotions
    - **Image Analysis:** CLIP model for image-text understanding + color analysis
    - **Multimodal:** Weighted combination of text and image sentiment scores
    - **Visualizations:** Interactive Plotly charts for emotion and sentiment distribution
    """)

if __name__ == "__main__":
    main()


# Now save it to a file
#with open('C:\\Users\\ashri\\Documents\\multimodal Sentiment Analysis\\multimodal_sentiment_app.py', 'w') as f:
   # f.write(multimodal_code)

#print("✅ Created multimodal_sentiment_app.py successfully!")