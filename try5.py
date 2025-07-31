import streamlit as st
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import random
import re

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- 1. Data Preprocessing ---

@st.cache_data
def load_and_preprocess_data(train_expanded_path, greetings_dataset_path):
    """
    Loads and preprocesses the FAQ and greetings datasets.
    Assigns intents and prepares data for BERT.
    """
    # Load FAQ data
    with open(train_expanded_path, 'r', encoding='utf-8') as f:
        faq_data = [json.loads(line) for line in f]

    # Load greetings data
    with open(greetings_dataset_path, 'r', encoding='utf-8') as f:
        greetings_data = json.load(f)

    # Prepare FAQ data
    faq_df = pd.DataFrame(faq_data)
    faq_df.columns = ['context', 'response']
    # Assign unique intent for each FAQ question
    faq_df['intent'] = faq_df['context'].apply(lambda x: f"faq_{re.sub(r'[^a-z0-9]+', '_', x.lower())[:50]}")

    # Prepare Greetings data
    greetings_df = pd.DataFrame(greetings_data)
    greetings_df['intent'] = 'greeting'

    # Combine datasets
    combined_df = pd.concat([faq_df, greetings_df], ignore_index=True)

    # Create intent to ID mapping
    unique_intents = combined_df['intent'].unique().tolist()
    intent_to_id = {intent: i for i, intent in enumerate(unique_intents)}
    id_to_intent = {i: intent for intent, i in intent_to_id.items()}
    combined_df['intent_id'] = combined_df['intent'].map(intent_to_id)

    st.write(f"Total unique intents: {len(unique_intents)}")
    st.write("Sample combined data:")
    st.dataframe(combined_df.head())

    return combined_df, intent_to_id, id_to_intent

class FAQDataset(Dataset):
    """Custom Dataset for BERT fine-tuning."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- 2. Train a BERT Model ---

@st.cache_resource
def train_bert_model(df, intent_to_id):
    """
    Trains a BERT model for intent classification.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_to_id))

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['context'].tolist(),
        df['intent_id'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['intent_id'] # Stratify to maintain class distribution
    )

    # Tokenize texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = FAQDataset(train_encodings, train_labels)
    val_dataset = FAQDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none" # Disable integrations like wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    st.write("Training BERT model...")
    trainer.train()
    st.write("Training complete!")

    return tokenizer, model, trainer, val_dataset, val_labels

# --- 3. Response Generation/Selection & Evaluation Metrics ---

def compute_metrics(p):
    """Computes metrics for intent classification."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def get_response(user_input, model, tokenizer, id_to_intent, combined_df):
    """
    Classifies user intent and returns the appropriate response.
    """
    # Preprocess user input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)

    # Predict intent
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_intent = id_to_intent[predicted_class_id]

    # Retrieve response
    if predicted_intent == 'greeting':
        # Filter for greeting responses and pick a random one
        greeting_responses = combined_df[combined_df['intent'] == 'greeting']['response'].tolist()
        response = random.choice(greeting_responses) if greeting_responses else "Hello! How can I assist you today?"
    elif predicted_intent.startswith('faq_'):
        # Find the exact question from the FAQ that matches the predicted intent
        # This assumes the intent is derived directly from the question for FAQs
        faq_entry = combined_df[combined_df['intent'] == predicted_intent]
        if not faq_entry.empty:
            response = faq_entry['response'].iloc[0]
        else:
            response = "I'm sorry, I couldn't find an answer to that specific FAQ. Can you rephrase or ask something else?"
    else:
        response = "I'm not sure how to respond to that. Can you please rephrase?"

    return predicted_intent, response

# --- 4. Build the Chatbot UI using Streamlit ---

def main():
    st.set_page_config(page_title="E-commerce FAQ Chatbot", layout="centered")
    st.title("🛍️ E-commerce FAQ Chatbot")
    st.markdown("Ask me anything about our products, orders, or general inquiries!")

    # Load and preprocess data
    combined_df, intent_to_id, id_to_intent = load_and_preprocess_data('train_expanded.json', 'greetings_dataset.json')

    # Train model
    tokenizer, model, trainer, val_dataset, val_labels = train_bert_model(combined_df, intent_to_id)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                predicted_intent, bot_response = get_response(prompt, model, tokenizer, id_to_intent, combined_df)
                st.markdown(bot_response)
                st.caption(f"Predicted Intent: `{predicted_intent}`")
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # --- 5. Evaluate and Compare Results ---
    st.sidebar.title("Evaluation Metrics")
    if st.sidebar.button("Run Evaluation"):
        st.sidebar.write("Running evaluation on the validation set...")
        eval_results = trainer.evaluate()
        st.sidebar.subheader("Intent Classification Metrics:")
        st.sidebar.write(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
        st.sidebar.write(f"Precision: {eval_results['eval_precision']:.4f}")
        st.sidebar.write(f"Recall: {eval_results['eval_recall']:.4f}")
        st.sidebar.write(f"F1 Score: {eval_results['eval_f1']:.4f}")

        st.sidebar.subheader("Response Quality (BLEU/ROUGE) - Sampled:")
        # For BLEU/ROUGE, we'll sample from the validation set
        # This is a simplified evaluation since we are selecting, not generating
        sample_size = min(50, len(val_dataset)) # Limit sample size for performance
        sampled_indices = random.sample(range(len(val_dataset)), sample_size)

        bleu_scores = []
        rouge1_fscores = []
        rouge2_fscores = []
        rougel_fscores = []

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoothie = SmoothingFunction().method1

        for i in sampled_indices:
            input_text = val_dataset.encodings['input_ids'][i]
            # Decode the input_text to get the actual query
            query_text = tokenizer.decode(input_text, skip_special_tokens=True)

            # Get the true intent and response
            true_intent_id = val_labels[i]
            true_intent = id_to_intent[true_intent_id]
            
            # Find the corresponding response from the combined_df based on the query text
            # This is a bit tricky because the validation set only has tokenized inputs,
            # not the original 'context'. We need to find the original context to get the true response.
            # A more robust approach would be to include original texts in val_dataset or map back.
            # For now, let's try to find the row in combined_df that matches the query_text and true_intent
            
            # Simplified approach: find the response by true intent and context (if available)
            # This assumes the context in combined_df is exactly what was used to train.
            # For a real system, you'd map back to the original text.
            
            # For the purpose of this demo, we'll assume the `query_text` from the tokenizer
            # is close enough to find the original `context` in `combined_df`.
            # A better way would be to pass `val_texts` to the evaluation loop.
            
            # Let's get the original context from the train_test_split directly
            original_val_context = val_texts[i]
            true_response_row = combined_df[(combined_df['context'] == original_val_context) & (combined_df['intent'] == true_intent)]
            
            if not true_response_row.empty:
                true_response = true_response_row['response'].iloc[0]
            else:
                true_response = "No true response found for evaluation."
                st.sidebar.warning(f"Could not find true response for: {query_text} (Intent: {true_intent})")
                continue # Skip if true response not found

            # Get the predicted response from the chatbot logic
            _, predicted_response = get_response(query_text, model, tokenizer, id_to_intent, combined_df)

            # BLEU Score
            reference = [true_response.split()]
            candidate = predicted_response.split()
            bleu_scores.append(sentence_bleu(reference, candidate, smoothing_function=smoothie))

            # ROUGE Score
            scores = scorer.score(true_response, predicted_response)
            rouge1_fscores.append(scores['rouge1'].fmeasure)
            rouge2_fscores.append(scores['rouge2'].fmeasure)
            rougel_fscores.append(scores['rougeL'].fmeasure)

        if bleu_scores:
            st.sidebar.write(f"Average BLEU Score: {np.mean(bleu_scores):.4f}")
            st.sidebar.write(f"Average ROUGE-1 F-score: {np.mean(rouge1_fscores):.4f}")
            st.sidebar.write(f"Average ROUGE-2 F-score: {np.mean(rouge2_fscores):.4f}")
            st.sidebar.write(f"Average ROUGE-L F-score: {np.mean(rougel_fscores):.4f}")
        else:
            st.sidebar.write("No valid samples for BLEU/ROUGE evaluation.")

if __name__ == "__main__":
    main()
