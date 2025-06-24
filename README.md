# Python-hugging-face-transformer
simple Python chatbot project that uses Hugging Face’s transformers library to summarize long articles and documents.
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load summarization pipeline
def load_summarizer():
    print("Loading model...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Model loaded.")
    return summarizer

# Chatbot loop
def chatbot():
    summarizer = load_summarizer()
    print("\nWelcome to the Summarizer Bot! Paste your long text and I will summarize it for you.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        if len(user_input.split()) < 30:
            print("Bot: Please enter a longer text (at least ~30 words) for a meaningful summary.")
            continue

        print("Bot: Summarizing...")
        try:
            summary = summarizer(user_input, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            print(f"Bot: {summary}\n")
        except Exception as e:
            print(f"Bot: Sorry, I couldn't summarize that. Error: {e}")

# Run chatbot
if __name__ == "__main__":
    chatbot()
