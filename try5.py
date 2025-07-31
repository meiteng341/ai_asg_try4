import streamlit as st
import json
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# Download NLTK punkt tokenizer for sentence tokenization if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 1. Data Preprocessing ---

def load_data(greetings_file, faq_file):
    """Loads data from JSON files."""
    with open(greetings_file, 'r', encoding='utf-8') as f:
        greetings_data = json.load(f)
    
    faq_data = []
    with open(faq_file, 'r', encoding='utf-8') as f:
        for line in f:
            faq_data.append(json.loads(line))
            
    return greetings_data, faq_data

def preprocess_text(text):
    """Cleans and preprocesses text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def prepare_datasets(greetings_data, faq_data):
    """
    Prepares combined dataset for training and separate response mappings.
    Assigns intents and stores responses.
    """
    corpus = []
    intents = []
    responses = {}

    # Add greetings data
    for entry in greetings_data:
        context = preprocess_text(entry['context'])
        response = entry['response']
        corpus.append(context)
        intents.append('greeting')
        if 'greeting' not in responses:
            responses['greeting'] = []
        responses['greeting'].append(response)

    # Add FAQ data
    for i, entry in enumerate(faq_data):
        question = preprocess_text(entry['question'])
        answer = entry['answer']
        
        # Create a unique intent for each FAQ question
        # Replace non-alphanumeric with underscore and remove leading/trailing underscores
        intent_name = re.sub(r'\W+', '_', entry['question']).strip('_').lower()
        if not intent_name: # Fallback if question becomes empty after cleaning
            intent_name = f"faq_{i}"

        corpus.append(question)
        intents.append(intent_name)
        responses[intent_name] = [answer] # FAQ intents have a single, direct answer

    return corpus, intents, responses

# --- 2. Model Training ---

def train_intent_classifier(corpus, intents):
    """
    Trains a Logistic Regression model for intent classification.
    Uses TF-IDF for feature extraction.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    y = np.array(intents)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return vectorizer, model, X_test, y_test, y_train

# --- 3. Generate or Select Responses ---

def get_chatbot_response(user_input, vectorizer, model, all_responses, threshold=0.6):
    """
    Predicts intent and generates a response.
    Includes a confidence threshold for unknown intents.
    """
    processed_input = preprocess_text(user_input)
    input_vec = vectorizer.transform([processed_input])
    
    # Get probabilities for each class
    probabilities = model.predict_proba(input_vec)[0]
    predicted_intent_idx = np.argmax(probabilities)
    predicted_intent = model.classes_[predicted_intent_idx]
    confidence = probabilities[predicted_intent_idx]

    if confidence >= threshold:
        if predicted_intent == 'greeting':
            return random.choice(all_responses['greeting'])
        elif predicted_intent in all_responses:
            return all_responses[predicted_intent][0] # FAQ has single answer
        else:
            return "I'm sorry, I don't have an answer for that specific question. Can you please rephrase or ask something else?"
    else:
        return "I'm not sure I understand. Could you please rephrase your question or ask about something else related to our e-commerce store?"

# --- 4. Build the Chatbot UI using Streamlit ---

st.set_page_config(page_title="E-commerce FAQ Chatbot", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #ccc;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .chat-message {
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 15px;
        max-width: 80%;
        font-family: "Inter", sans-serif;
    }
    .user-message {
        background-color: #e6f7ff;
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }
    .bot-message {
        background-color: #ffffff;
        align-self: flex-start;
        border-bottom-left-radius: 2px;
    }
    .chat-container {
        height: 500px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fdfdfd;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛍️ E-commerce FAQ Chatbot")
st.write("Hello! I'm here to help you with common questions about our store. Ask me anything!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load and prepare data (cached to avoid reloading on every rerun)
@st.cache_resource
def load_and_train_model():
    greetings_data, faq_data = load_data('greetings_dataset.json', 'train_expanded.json')
    corpus, intents, responses = prepare_datasets(greetings_data, faq_data)
    vectorizer, model, X_test, y_test, y_train = train_intent_classifier(corpus, intents)
    return vectorizer, model, responses, X_test, y_test, y_train, intents

vectorizer, model, all_responses, X_test, y_test, y_train, all_intents = load_and_train_model()

# Display chat messages from history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message["content"]}</div>', unsafe_allow_html=True)

# User input
user_query = st.text_input("Type your question here:", key="user_input")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with chat_container:
        st.markdown(f'<div class="chat-message user-message">{user_query}</div>', unsafe_allow_html=True)

    # Get bot response
    bot_response = get_chatbot_response(user_query, vectorizer, model, all_responses)
    
    # Add bot message to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    with chat_container:
        st.markdown(f'<div class="chat-message bot-message">{bot_response}</div>', unsafe_allow_html=True)

    # Clear input box after sending
    st.session_state.user_input = "" # This clears the text input

# --- 5. Evaluate and Compare Results ---

st.sidebar.title("Model Evaluation")

if st.sidebar.button("Run Evaluation"):
    st.sidebar.write("Running evaluation on test set...")
    y_pred = model.predict(X_test)

    st.sidebar.subheader("Intent Classification Metrics:")
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"Accuracy: {accuracy:.4f}")

    # Calculate precision, recall, f1-score for each class and average
    # Handle cases where a class might not be present in y_test or y_pred
    # Use 'zero_division=0' to prevent warnings for classes with no true samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    st.sidebar.write(f"Precision (weighted): {precision:.4f}")
    st.sidebar.write(f"Recall (weighted): {recall:.4f}")
    st.sidebar.write(f"F1 Score (weighted): {f1:.4f}")

    # Detailed classification report
    st.sidebar.text("Classification Report:")
    # Filter out intents that are not in y_train to avoid errors in classification_report
    # because some intents might only appear in the training set and not in the test set.
    # This ensures that the report only includes labels that are actually present in y_true and y_pred.
    unique_labels_in_test = np.unique(np.concatenate((y_test, y_pred)))
    report = classification_report(y_test, y_pred, labels=unique_labels_in_test, zero_division=0)
    st.sidebar.code(report)

    st.sidebar.subheader("Response Quality Metrics (BLEU/ROUGE):")
    st.sidebar.write("""
    For this retrieval-based FAQ chatbot, BLEU/ROUGE scores are less directly applicable
    than for generative models. Our goal is to retrieve the exact pre-defined answer.
    """)
    st.sidebar.write("""
    However, for demonstration, we can calculate them for the test set's predicted responses
    against their ground truth responses. A perfect score would indicate exact retrieval.
    """)

    # Example BLEU/ROUGE calculation (conceptual for retrieval)
    # This part is more for understanding how these metrics work,
    # as for a retrieval system, it's essentially an exact match check.
    
    # Map intents back to original questions/answers for evaluation
    reverse_faq_map = {re.sub(r'\W+', '_', entry['question']).strip('_').lower(): entry['answer'] for entry in faq_data}
    
    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction() # For BLEU score smoothing

    # Create a mapping from intent to the *single* expected response for evaluation
    # This is crucial because `all_responses['greeting']` is a list of possible responses,
    # but for evaluation, we need a single ground truth or a set of references.
    # For simplicity, we'll pick one if multiple exist for greetings.
    eval_ground_truth_responses = {}
    for intent, res_list in all_responses.items():
        if intent == 'greeting':
            # For greetings, we can't have a single "ground truth" so we'll just use one for calculation
            # In a real evaluation, you'd compare against all possible valid responses.
            eval_ground_truth_responses[intent] = res_list[0] 
        else:
            eval_ground_truth_responses[intent] = res_list[0] # FAQ has only one answer

    for i, test_query_vec in enumerate(X_test):
        true_intent = y_test[i]
        predicted_intent = model.predict(test_query_vec)[0]

        # Get the original user query from the corpus for context (if needed, not directly used in metrics)
        # This requires mapping back from X_test index to original corpus index, which is complex.
        # Instead, we'll use the true_intent to get the expected response.

        # Ground truth response for the true intent
        reference_response = eval_ground_truth_responses.get(true_intent, "I don't know.")
        
        # Predicted response based on the predicted intent
        predicted_response = all_responses.get(predicted_intent, ["I'm not sure I understand."])[0] # Take first for simplicity

        # BLEU Score (requires tokenized sentences)
        reference_tokens = nltk.word_tokenize(reference_response)
        candidate_tokens = nltk.word_tokenize(predicted_response)
        
        # Ensure candidate is not empty to avoid error
        if candidate_tokens:
            bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=chencherry.method1)
            bleu_scores.append(bleu_score)

        # ROUGE Score
        scores = scorer.score(reference_response, predicted_response)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    if bleu_scores:
        avg_bleu = np.mean(bleu_scores)
        st.sidebar.write(f"Average BLEU Score: {avg_bleu:.4f}")
    else:
        st.sidebar.write("No BLEU scores calculated (possibly no valid predictions).")

    if rouge1_scores:
        avg_rouge1 = np.mean(rouge1_scores)
        avg_rougeL = np.mean(rougeL_scores)
        st.sidebar.write(f"Average ROUGE-1 F-measure: {avg_rouge1:.4f}")
        st.sidebar.write(f"Average ROUGE-L F-measure: {avg_rougeL:.4f}")
    else:
        st.sidebar.write("No ROUGE scores calculated (possibly no valid predictions).")

    st.sidebar.write("""
    **Interpretation of BLEU/ROUGE for Retrieval:**
    - A BLEU/ROUGE score close to 1.0 indicates that the retrieved response
      is very similar or identical to the ground truth response.
    - Lower scores suggest that the model either predicted the wrong intent
      leading to a different response, or the "ground truth" itself has variations.
    - For FAQ chatbots, the primary goal is accurate intent classification,
      which directly leads to the correct pre-defined answer.
    """)
