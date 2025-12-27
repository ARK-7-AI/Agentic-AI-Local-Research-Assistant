from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from yaspin import yaspin
from tqdm import tqdm


def reseach_assistant(test_script: bool = False):
    # Load text file
    loader = TextLoader('./docs/world_models_cleaned.txt', encoding='utf8')
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    chunk_word_counts = [len(chunk.page_content.split()) for chunk in chunks]
    print(f"Total words across all chunks: {sum(chunk_word_counts)}")

    # Create Embeddings
    embedding_fn = OllamaEmbeddings(model="llama3.2:3b")

    # populate the vectorstore and store the vectors in memory
    with yaspin(text="Embedding and indexing chunks..."):
        vectorstore = FAISS.from_documents(
            tqdm(chunks, desc="Embedding chunks"),
            embedding_fn
        )

    # Instantiate the LLM
    llm = ChatOllama(model="llama3.2:3b")

    with yaspin(text="Loading index and preparing retriever..."):
        retriever = vectorstore.as_retriever()
        retriever.search_kwargs["k"] = 3
    if not test_script:
        while True:
            question = input("Enter your question: (or type 'quit' to exit)")
            if question.lower() == 'quit':
                print("\nThanks for using the 583-Research-Assistant.")
                break
            else:
                docs = retriever.invoke(question)

                context = "\n\n".join([doc.page_content for doc in docs])
                # Prompt
                prompt = f"Use the following context to answer concisely:\n{context}\nQuestion: {question}"
                with yaspin(text="Processing..."):
                    answer = llm.invoke(prompt)

                print(f"User: {question}\n")
                if hasattr(answer, 'content'):
                    print(f"Assistant: {answer.content}\n")
                else:
                    print(f"Assistant: {answer}\n")

    return llm, retriever


if __name__ == "__main__":
    reseach_assistant()
