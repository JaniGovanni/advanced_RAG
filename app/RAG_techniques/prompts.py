MULTI_QUERY_PROMPT = """
You are a knowledgeable research assistant. Your task is to generate five different versions 
of the given user question to retrieve relevant documents from a vector database. By generating 
multiple perspectives on the user question, your goal is to help the user overcome some of the 
limitations of the distance-based similarity search.

Provide your response in the following JSON format:
{
    "queries": [
        "Alternative query 1",
        "Alternative query 2",
        "Alternative query 3",
        "Alternative query 4",
        "Alternative query 5"
    ]
}

Ensure that each alternative query is complete, directly related to the original inquiry, 
and covers various aspects of the topic. Make the queries concise and focused on a single topic.
"""

HYDE_PROMPT = """
    You are a knowledgeable and helpful research assistant.
    Provide an example answer to the given question,
    that could be found in a scholarly or professional document or an 
    scientific paper, and 
    might actually be true.
    """

HYDE_PROMPT_V2 = """
You are an expert in generating hypothetical document excerpts. Your task is to create a plausible, detailed snippet that could be found in a scholarly article, professional report, or authoritative website, addressing the given question.

Follow these guidelines:
1. Write in a formal, academic tone.
2. Include specific details, statistics, or references where appropriate.
3. Aim for a length of 3-5 sentences.
4. Ensure the content is factual and could realistically appear in a reputable source.
5. Do not preface or conclude the excerpt; write as if it's taken directly from the middle of a document.
6. Avoid first-person pronouns and directly addressing the reader.

Remember, your goal is to create a snippet that seems like it could be a genuine excerpt from a real document, containing information that answers the given question.

Question: {question}

Generate a hypothetical document excerpt:
"""

# for HyDE
HYDE_PROMPT_TEST = """
    Assume you have access to a hypothetical document. 
    This could be a website, a scientific paper, a report, a Wikipedia article, or anything else. 
    Your task is to provide an text snippet from such a hypothetical document 
    that could contain the answer 
    to a user's question. Please dont output anything else. Here is an example:

    user_question_1: Is Discretization necessary?
    
    answer_1: However, the parameterization of Mamba still used the same discretization 
    step as in prior structured SSMs, where there is another parameter Δ being modeled. 
    We do this because the discretization step has other side effects such as properly normalizing the activations 
    which is important for performance.
    The initializations and parameterizations from the 
    previous  SSMs still work out-of-the-box, so why fix what’s not broken?
    Despite this, we’re pretty sure that the discretization step isn’t necessary for Mamba. 
    In the Mamba-2 paper, we chose to work directly with the “discrete parameters” 𝐴 and 𝐵, 
    which in all previous structured SSM papers (including Mamba-1) were denoted
    
    user_question_2: What is the Hooded Man?
    
    answer_2: The Hooded Man (or The Man on the Box)[1] is an image showing 
    a prisoner at Abu Ghraib prison with wires attached to his fingers, 
    standing on a box with a covered head. The photo has been portrayed as an iconic 
    photograph of the Iraq War, "the defining image of the scandal" 
    and "symbol of the torture at Abu Ghraib". The image was published on the 
    cover of The Economist's 8 May 2004 issue, the opening photo of The New 
    Yorker on 10 May 2004, and on 11 March 2006 in The New York Times's 
    first section at the top left-hand corner.
    """

MULTI_QUERY_PROMPT_TEST = template = """
You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""