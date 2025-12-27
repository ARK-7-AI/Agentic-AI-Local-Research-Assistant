from yaspin import yaspin
from research_assistant import reseach_assistant


llm, retriever = reseach_assistant(True)
questions = ["What are World Models?",
             "What do you mean by probabilistic representation for a World Model?", "Summarize the goal of the paper"]

# Create the RetrievalQA chain
for question in questions:
    question = question.strip()

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
