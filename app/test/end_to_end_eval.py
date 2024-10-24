from app.llm import get_groq_llm, get_ollama_llm
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser, PydanticOutputParser
import os
from pydantic import BaseModel, Field
from typing import Optional

def evaluate_answer(query: str, generated_answer: str, reference_answer: str) -> int:
    """
    Evaluates the relevance and quality of a generated answer to a given query,
    taking into account a reference answer.
    
    Args:
    query (str): The original query.
    generated_answer (str): The answer generated by the RAG system.
    reference_answer (str): The known correct or reference answer.
    
    Returns:
    int: A score from 0 to 100 indicating the quality and relevance of the answer.
    """
    
  
    llm = get_groq_llm()

    # Define the evaluation prompt
    evaluation_prompt = ChatPromptTemplate.from_template("""
    You are an expert evaluator of question-answering systems. Your task is to evaluate the relevance and quality of a generated answer to a given query, comparing it to a reference answer.

    Query: {query}
    Generated Answer: {generated_answer}
    Reference Answer: {reference_answer}

    First, check if the generated answer indicates that the provided context doesn't contain information about the query. If so, assign a score of 0.

    If the answer doesn't indicate a lack of information, please evaluate the generated answer based on the following criteria:
    1. Relevance: How well does the answer address the query?
    2. Accuracy: Is the information in the answer correct and factual when compared to the reference answer?
    3. Completeness: Does the answer provide a comprehensive response to the query, covering key points from the reference answer?
    4. Clarity: Is the answer clear and easy to understand?
    5. Comparison to Reference: How well does the generated answer align with the information provided in the reference answer?

    Provide a score from 0 to 100, where:
    0: The answer indicates that the provided context doesn't contain information about the query
    1-20: Completely irrelevant, incorrect, or contradicts the reference answer
    21-40: Partially relevant but missing key information or containing significant inaccuracies
    41-60: Moderately relevant and accurate, but incomplete or lacking clarity
    61-80: Mostly relevant, accurate, and clear, with minor omissions or imperfections
    81-100: Highly relevant, accurate, complete, and clear, closely aligning with the reference answer

    Also provide a brief explanation for your score, highlighting strengths and areas for improvement.

    {format_instructions}
    """)

    # Define the output schema
    response_schemas = [
        ResponseSchema(name="score", description="The numerical score from 0 to 100"),
        ResponseSchema(name="explanation", description="A brief explanation for the given score, highlighting strengths and areas for improvement")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Format the prompt with the output parser instructions
    format_instructions = output_parser.get_format_instructions()
    formatted_prompt = evaluation_prompt.format_messages(
        query=query,
        generated_answer=generated_answer,
        reference_answer=reference_answer,
        format_instructions=format_instructions
    )

    
    response = llm.invoke(formatted_prompt)

    # Parse the response
    parsed_response = output_parser.parse(response.content)

    # Return the score
    return int(parsed_response["score"])




class EvaluationResult(BaseModel):
    score: int = Field(..., ge=0, le=100)
    explanation: str

def evaluate_answer_v1(query: str, generated_answer: str, reference_answer: str) -> Optional[int]:
    """
    Evaluates the relevance and quality of a generated answer to a given query,
    taking into account a reference answer.
    
    Args:
    query (str): The original query.
    generated_answer (str): The answer generated by the RAG system.
    reference_answer (str): The known correct or reference answer.
    
    Returns:
    Optional[int]: A score from 0 to 100 indicating the quality and relevance of the answer,
                   or None if evaluation fails.
    """
    
    # Initialize the Groq LLM
    llm = get_groq_llm()

    # Define the evaluation prompt
    evaluation_prompt = ChatPromptTemplate.from_template("""
    You are an expert evaluator of question-answering systems. Your task is to evaluate the relevance and quality of a generated answer to a given query, comparing it to a reference answer.

    Query: {query}
    Generated Answer: {generated_answer}
    Reference Answer: {reference_answer}

    First, check if the generated answer indicates that the provided context doesn't contain information about the query. If so, assign a score of 0.

    If the answer doesn't indicate a lack of information, please evaluate the generated answer based on the following criteria:
    1. Relevance: How well does the answer address the query?
    2. Accuracy: Is the information in the answer correct and factual when compared to the reference answer?
    3. Completeness: Does the answer provide a comprehensive response to the query, covering key points from the reference answer?
    4. Clarity: Is the answer clear and easy to understand?
    5. Comparison to Reference: How well does the generated answer align with the information provided in the reference answer?

    Provide a score from 0 to 100, where:
    0: The answer indicates that the provided context doesn't contain information about the query
    1-20: Completely irrelevant, incorrect, or contradicts the reference answer
    21-40: Partially relevant but missing key information or containing significant inaccuracies
    41-60: Moderately relevant and accurate, but incomplete or lacking clarity
    61-80: Mostly relevant, accurate, and clear, with minor omissions or imperfections
    81-100: Highly relevant, accurate, complete, and clear, closely aligning with the reference answer

    Also provide a brief explanation for your score, highlighting strengths and areas for improvement.

    {format_instructions}
    """)

    # Use PydanticOutputFunctionsParser for more reliable parsing
    # Define the output parser
    output_parser = PydanticOutputParser(pydantic_object=EvaluationResult)


    # Format the prompt with the output parser instructions
    format_instructions = output_parser.get_format_instructions()
    formatted_prompt = evaluation_prompt.format(
        query=query,
        generated_answer=generated_answer,
        reference_answer=reference_answer,
        format_instructions=format_instructions
    )

    try:
        # Get the LLM response
        response = llm.invoke(formatted_prompt)

        # Parse the response
        parsed_response = output_parser.parse(response.content)

        # Return the score
        return parsed_response.score
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        
        # Fallback: Try to extract score using regex
        import re
        match = re.search(r'score"?\s*:\s*(\d+)', response.content)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None