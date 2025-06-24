import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# Initialize models (loaded only once)
@st.cache_resource
def load_models():
    return {
        'tokenizer': AutoTokenizer.from_pretrained("bert-base-uncased"),
        'sentence_model': SentenceTransformer('all-MiniLM-L6-v2')
    }

models = load_models()

def hf_tokenize(text):
    tokens = models['tokenizer'].tokenize(text.lower())
    stop_words = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'}  # Add more as needed
    return [token for token in tokens if token not in stop_words]

def answer_correctness(ground_truth, generated_answer):
    """Calculate answer correctness using cosine similarity of TF-IDF vectors"""
    vectorizer = TfidfVectorizer(tokenizer=hf_tokenize, lowercase=False)
    tfidf = vectorizer.fit_transform([ground_truth, generated_answer])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def answer_relevance(query, ground_truth, generated_answer):
    """Calculate relevance using HuggingFace tokenization"""
    query_terms = set(hf_tokenize(query))
    gt_terms = set(hf_tokenize(ground_truth))
    gen_terms = set(hf_tokenize(generated_answer))
    
    if not query_terms:
        return 0.5
    
    ideal_matches = query_terms.intersection(gt_terms)
    actual_matches = query_terms.intersection(gen_terms)
    
    if not ideal_matches:
        return 0.5
    
    relevance_score = len(actual_matches) / len(ideal_matches)
    return max(0, min(1, relevance_score))

def semantic_similarity(ground_truth, generated_answer):
    """Calculate similarity using sentence transformers"""
    embeddings = models['sentence_model'].encode([ground_truth, generated_answer])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def bleu_score(ground_truth, generated_answer):
    """Calculate smoothed BLEU score"""
    reference = [hf_tokenize(ground_truth)]
    candidate = hf_tokenize(generated_answer)
    
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        reference, 
        candidate,
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25)
    )

def main():
    st.title("📊 Answer Evaluation System")
    st.write("Upload a CSV file with columns containing query, ground truth, and generated answers")
    
    metric_options = {
        "Correctness (TF-IDF Cosine)": answer_correctness,
        "Relevance (Query Terms)": answer_relevance,
        "Semantic Similarity (BERT)": semantic_similarity,
        "BLEU Score": bleu_score,
    }
    
    selected_metrics = st.multiselect(
        "Choose metrics to calculate:",
        list(metric_options.keys()),
        default=["Correctness (TF-IDF Cosine)", "Relevance (Query Terms)"]
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())
        
        # Flexible column mapping
        col_map = {
            'query': None,
            'ground_truth': None,
            'generated_answer': None
        }
        
        # Auto-detect columns (case insensitive)
        for col in df.columns:
            lower_col = col.lower()
            if 'query' in lower_col:
                col_map['query'] = col
            elif 'ground' in lower_col or 'truth' in lower_col:
                col_map['ground_truth'] = col
            elif 'generated' in lower_col or 'answer' in lower_col:
                col_map['generated_answer'] = col
        
        # Verify we found all required columns
        missing = [k for k,v in col_map.items() if v is None]
        if missing:
            st.error(f"❌ Could not auto-detect columns for: {', '.join(missing)}")
            st.info("Please ensure your CSV contains columns with these terms in their names:")
            st.markdown("- **Query** (contains 'query')")
            st.markdown("- **Ground Truth** (contains 'ground' or 'truth')")
            st.markdown("- **Generated Answer** (contains 'generated' or 'answer')")
            return
        
        st.success(f"✅ Detected columns: {', '.join(col_map.values())}")
        
        # Calculate selected metrics
        with st.spinner(f'Calculating {len(selected_metrics)} metrics...'):
            for metric in selected_metrics:
                metric_name = metric.split(" (")[0]  # Extract base name
                if metric_name == "Relevance":
                    df[metric_name] = df.apply(
                        lambda row: answer_relevance(
                            row[col_map['query']], 
                            row[col_map['ground_truth']], 
                            row[col_map['generated_answer']]
                        ), 
                        axis=1
                    )
                else:
                    df[metric_name] = df.apply(
                        lambda row: metric_options[metric](
                            row[col_map['ground_truth']], 
                            row[col_map['generated_answer']]
                        ), 
                        axis=1
                    )
        
        st.success("✅ Metrics calculated!")
        
        # Show results - use the base names for formatting
        st.write("### Results with Metrics:")
        metric_columns = [m.split(" (")[0] for m in selected_metrics]
        st.dataframe(df.style.format("{:.2f}", subset=metric_columns))
        
        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download results as CSV",
            data=csv,
            file_name='evaluated_answers.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()