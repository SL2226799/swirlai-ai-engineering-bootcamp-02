import openai

from qdrant_client import QdrantClient
import os
import instructor
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery
from langsmith import traceable, get_current_run_tree
from src.chatbot_ui.core.config import config

qdrant_url = os.getenv('QDRANT_URL', 'http://qdrant:6333')
qdrant_client = QdrantClient(url=qdrant_url)

collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'Amazon-items-collection-01')

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={
        "ls_model": config.EMBEDDING_MODEL,
        "ls_model_provider": config.EMBEDDING_MODEL_PROVIDER,
    }
)
def get_embedding(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding

@traceable(
    name="retrieve_top_k",
    run_type="retriever",
)
def retrieve_context(query, qdrant_client, top_k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid",
        prefetch=[
            Prefetch(
                query=query_embedding,
                limit=20
            ),
            Prefetch(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchText(text=query)
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k
    )


    retrieved_context = []
    retrieved_context_ids = []
    similarity_scores = []

    print("results:", results)

    for result in results.points:
        retrieved_context.append(result.payload['text'])
        retrieved_context_ids.append(result.id)
        similarity_scores.append(result.score)

    return {
        "retrieved_context": retrieved_context, 
        "retrieved_context_ids": retrieved_context_ids, 
        "similarity_scores": similarity_scores
    }

@traceable(
    name="format_retrieved_context",
    run_type="prompt",
)
def process_context(context):

    formatted_context = ""

    for chunk in context["retrieved_context"]:
        formatted_context += f"- {chunk}\n"

    return formatted_context

@traceable(
    name="render_prompt",
    run_type="prompt",
)
def build_prompt(context, query):

    processed_context = process_context(context)
    
    prompt = f"""
    You are a shopping assistant that answers questions based on products in stock.

    Instructions:
    - Answer questions based only on the provided context.
    - Do not say the word "context".
    - If the question cannot be answered based on the context, respond with "There is no information to answer this question."

    Context: {processed_context}

    Question: {query}
    """
    return prompt

class RAGGenerationResponse(BaseModel):
    answer: str

@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={
        "ls_model": config.GENERATION_MODEL,
        "ls_model_provider": config.GENERATION_MODEL_PROVIDER,
    }
)
def generate_answer(prompt):

    client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=RAGGenerationResponse,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
    return response

@traceable(
    name="rag_pipeline",
)
def rag_pipeline(query, qdrant_client, top_k=5):
    context = retrieve_context(query, qdrant_client, top_k)
    prompt = build_prompt(context, query)
    answer = generate_answer(prompt)

    final_output = {
        "answer": answer,
        "question": query,
        "retrieved_context": context["retrieved_context"],
        "retrieved_context_ids": context["retrieved_context_ids"],
        "similarity_scores": context["similarity_scores"]
    }
    return final_output