import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

load_dotenv()


def google_queries_merger_runnable():
    template = """Transform next queries to the single Google Search query:
    {queries}
    
    Tip: Make sure you provided correct output.
    Tip: Make sure you provide just text and do not include it in quotes
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model_name="gpt-4-1106-preview")

    return (
            {"queries": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )


def rag_runnable(retriever):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model_name="gpt-4-1106-preview")

    return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )


def decompose_question_runnable():
    template = """I have a Google search request: "{question}". Please decompose the question into several subsequent questions where I can pass the answer from the previous question to the next. Each question should only depend on the answer of the previous question. Make sure each subsequent query has a placeholder in square brackets explains what information need to be passed. Make wording of these questions as a Google search request query. Please respond with a maximum of 3 decomposed questions.
    Tip: Make sure that questions do not repeat each other.
    Tip: Answer in json format.
    Tip: Returned json object should contain "questions" field with array of string questions inside.
    Tip: If the question is already simple - do not split it to multiple questions
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model_name="gpt-4-1106-preview")
    model.bind(response_format={"type": "json_object"})

    class OutputQuestionsSchema(BaseModel):
        questions: list[str]

    return (
            {"question": RunnablePassthrough()}
            | prompt
            | model
            | PydanticOutputParser(pydantic_object=OutputQuestionsSchema)
    )


def complex_answer_with_context():
    template = """
    I have a complex question that was split on multiple simple questions and answers. I need to summarise these questions and answers to provide correct answer to the complex question. Summarize the answer only based on provided questions and answers:
    {qnas}

    Please provide an answer for the complex question:
    {question}

    Tip: Make sure you provided correct output.
    Tip: Make sure you provide just text and do not include it in quotes
    Tip: Do not propose google some part of the answer, keep the summary with the answers in questions-answers.
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model_name="gpt-4-1106-preview")

    return (
            {"qnas": itemgetter("qnas"), "question": itemgetter("question")}
            | prompt
            | model
            | StrOutputParser()
    )
