from langchain.prompts import PromptTemplate

# General-purpose prompt template
general_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant. Use the following context to answer the question.
If the answer is not found in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)

# Factual QA prompt (more concise)
factual_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Based on the facts below, provide a direct and concise answer.

Facts:
{context}

Question:
{question}

Answer:
"""
)

# Open-ended reasoning prompt
reasoning_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Read the context and think step-by-step to answer the question in detail.

Context:
{context}

Question:
{question}

Detailed Answer:
"""
)

# Function to retrieve prompt based on query type
def get_prompt_template(query_type="general"):
    if query_type == "factual":
        return factual_prompt
    elif query_type == "reasoning":
        return reasoning_prompt
    else:
        return general_prompt
