import streamlit as st
from time import sleep
from stqdm import stqdm
import pandas as pd
from transformers import pipeline
import json
import spacy
import spacy_streamlit


def draw_all(key, plot=False):
    st.write(
        """
        # DocGen
        
        This Natural Language Processing-based web application leverages pretrained transformer models to process and analyze textual data.

        ```python
        # Key Features
        1. Advanced Text Summarizer
        2. Named Entity Recognition (NER)
        3. Sentiment Analysis
        4. Question Answering
        5. Text Completion
        ```
        """
    )

with st.sidebar:
    draw_all("sidebar")


def main():
    st.title("DocGen")
    menu = ["--Select--", "Summarizer", "Named Entity Recognition", "Sentiment Analysis", "Question Answering", "Text Completion"]
    choice = st.sidebar.selectbox("Choose an operation", menu)

    if choice == "--Select--":
        st.write("This web app demonstrates the power of Natural Language Processing using transformer models.")
        st.write("NLP is a field within Artificial Intelligence focused on enabling machines to understand and process human language.")
        st.image("banner_image.jpg")

    elif choice == "Summarizer":
        st.subheader("Text Summarization")
        raw_text = st.text_area("Enter the text to summarize:")
        num_words = st.number_input("Minimum number of words in the summary")

        if raw_text and num_words:
            summarizer = pipeline("summarization")
            summary = summarizer(raw_text, min_length=int(num_words), max_length=50)
            result_summary = summary[0]["summary_text"]
            result_summary = '. '.join([s.strip().capitalize() for s in result_summary.split('.')])
            st.write("Summary:")
            st.write(result_summary)

    elif choice == "Named Entity Recognition":
        st.subheader("Named Entity Recognition")
        nlp = spacy.load("en_core_web_sm")
        raw_text = st.text_area("Enter the text for entity extraction:")

        if raw_text.strip():
            for _ in stqdm(range(50), desc="Processing..."):
                sleep(0.1)
            doc = nlp(raw_text)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title="Entities")

    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        sentiment_pipeline = pipeline("sentiment-analysis")
        raw_text = st.text_area("Enter text to analyze sentiment:")

        if raw_text.strip():
            for _ in stqdm(range(50), desc="Processing..."):
                sleep(0.1)
            result = sentiment_pipeline(raw_text)[0]
            sentiment = result["label"]
            st.write(f"Sentiment: {sentiment.capitalize()}")

    elif choice == "Question Answering":
        st.subheader("Question Answering")
        context = st.text_area("Enter the context:")
        question = st.text_area("Enter your question:")

        if context.strip() and question.strip():
            qa_pipeline = pipeline("question-answering")
            result = qa_pipeline(question=question, context=context)
            answer = result["answer"]
            answer = '. '.join([s.strip().capitalize() for s in answer.split('.')])
            st.write("Answer:")
            st.write(answer)

    elif choice == "Text Completion":
        st.subheader("Text Completion")
        message = st.text_area("Enter the beginning of the text:")

        if message.strip():
            generator = pipeline("text-generation")
            result = generator(message)[0]
            generated_text = result["generated_text"]
            generated_text = '. '.join([s.strip().capitalize() for s in generated_text.split('.')])
            st.write("Completed Text:")
            st.write(generated_text)


if __name__ == "__main__":
    main()
