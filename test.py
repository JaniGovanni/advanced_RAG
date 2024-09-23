from app.chat import ChatConfig, get_result_docs

def main():
    # Create a ChatConfig instance
    chat_config = ChatConfig(
        tag="attention",
        use_bm25=True,  # False
        k=5,  # Number of results to retrieve
        llm_choice="groq",  # or "ollama", depending on your preference
        reranking=False,
    )

    # Set up any additional configuration if needed
    chat_config.history_awareness(False)  # Disable history awareness for this example

    # Define a query
    query = "This query is meant to confuse the similarity search, by randomly typing BLEU."

    # Get the results
    result_docs, joint_query = get_result_docs(chat_config, query)
    result_docs = [doc for doc in result_docs if "BLEU" in doc]
    print(result_docs)

if __name__ == "__main__":
    main()