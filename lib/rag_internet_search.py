from typing import Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from lib.runnables import google_queries_merger_runnable, decompose_question_runnable, complex_answer_with_context

from lib.runnables import rag_runnable
from lib.scrapping import get_html, get_text
from lib.search import do_search


def answer_simple_question(question: str):
    search_results = do_search(question)

    pages: list[Document] = []

    for result in search_results["organic"]:
        link = result["link"]

        html = get_html(link)
        text = get_text(html, link, False)

        pages.append(Document(page_content=text, metadata=result))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)

    vectorstore = FAISS.from_documents(
        splits, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    runnable = rag_runnable(retriever)

    return runnable.invoke(question)


def _generate_qna_template(questions, answers):
    qnas = []
    for q, a in zip(questions, answers):
        qnas.append(f"""
QUESTION:
{q}
ANSWER:
{a}
""")
    return "".join(qnas)


def answer_complex_question(question: str):
    runnable = decompose_question_runnable()

    output_questions_model = runnable.invoke(question)

    print(f"Split complex question to set of: {output_questions_model}")

    answer: Optional[str] = None
    answers: list[str] = []

    for question in output_questions_model.questions:
        combined_question = question
        if answer:
            merger_runnable = google_queries_merger_runnable()
            combined_question = merger_runnable.invoke(". ".join([answer, question]))

        print(combined_question)
        answer = answer_simple_question(combined_question)
        answers.append(answer)
        print(answer)

    qnas = _generate_qna_template(output_questions_model.questions, answers)
    complex_answer_with_context_runnable = complex_answer_with_context()

    return complex_answer_with_context_runnable.invoke(
        {"qnas": qnas, "question": question})
