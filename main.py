import sys

from lib.rag_internet_search import answer_complex_question

if __name__ == "__main__":
    question = sys.argv[1]

    if question is None:
        print('No question provided')
        sys.exit()

    print(f"Answering for the question: {question}")
    answer = answer_complex_question(question)

    print("============================")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
