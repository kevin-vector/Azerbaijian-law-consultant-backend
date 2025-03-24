from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import openai
import tiktoken
from langdetect import detect
from typing import List
import os

app = FastAPI()

# Supabase setup
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Embedding model setup
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone = Pinecone(api_key=PINECONE_API_KEY)
law = pinecone.Index("law")
rule = pinecone.Index("rule")
post = pinecone.Index("post")

# OpenAI setup
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TPM_LIMIT = 30000
encoder = tiktoken.encoding_for_model("gpt-4o")

# Pydantic model for request
class QueryRequest(BaseModel):
    query: str

# Embedding function
def get_embedding(text: str) -> List[float]:
    return model.encode([text])[0].tolist()

# Fetch content from Supabase
async def fetch_content_from_supabase(id: str, table_name: str) -> str:
    table = f"Ajerbaijian_{table_name}"
    try:
        response = await supabase.table(table).select("content").eq("id", id).execute()
        return response.data[0]["content"] if response.data else "No content found"
    except Exception as e:
        print(f"Supabase error for {table_name} ID {id}: {str(e)}")
        return "No content found"

# Fetch results from Pinecone
async def fetch_results(index, query_embedding: List[float], index_name: str) -> List[str]:
    try:
        results = await index.query(vector=query_embedding, top_k=3, include_metadata=True)
        filtered_results = [
            await fetch_content_from_supabase(match['id'], index_name)
            for match in results['matches'] if match['score'] >= 0.5
        ]
        return filtered_results if filtered_results else ["No relevant content found"]
    except Exception as e:
        print(f"Pinecone error for {index_name}: {str(e)}")
        return ["No relevant content found"]

# Base prompt
base_prompt = """You are an advanced legal analysis AI built to assist users in understanding and interpreting legal documents, laws, and related social media posts. Your primary focus is on legal documents ('rules' and 'laws'), which include detailed legal texts ('rules') and formal announcements of law changes ('laws'). You also have access to social media posts ('posts') discussing law-related topics, but these are secondary to the legal documents unless otherwise specified. Your task is to analyze the provided data and generate responses based on user queries.

Respond in the following language: {}.

The user has provided a dataset containing the following retrieved entries:
- 'Rules': {}
- 'Laws': {}
- 'Posts': {}

Follow these instructions for every response:
1. Analyze the provided data and generate structured responses in bullet-point format.
2. Ensure responses are:
   - Logically coherent and legally accurate based on the data.
   - Specific to the provided data, avoiding vague or generic answers.
   - Optionally include citations to the source material (e.g., 'Rule: [title/section]', 'Law: [title/date]', 'Post: [summary/ID]') if relevant to support your reasoning—include them only when they add value or clarity.
3. Provide two response modes that the user can toggle between:
   - 'Detailed': Comprehensive answers with in-depth analysis, explanations, and examples from the data.
   - 'Summarized': Concise answers focusing on key points without excessive elaboration.
   Use '{}' mode for this response.
4. If the user asks for an explanation of a legal concept, provide a clear and accurate explanation grounded in the data, using examples where applicable.
5. If the query cannot be fully answered with the provided data due to insufficient or irrelevant content, do not speculate or provide incomplete answers. Instead, respond with: '{}' (translated to the appropriate language).

For this task, the user’s query is provided separately. Analyze the provided dataset and respond accordingly. Begin your response with a note indicating the mode, e.g., '[{} Response]'."""

# Adjust prompt for token limits
def adjust_prompt_tokens(base_prompt: str, results_rule: List[str], results_law: List[str], results_post: List[str], query: str, mode: str, tpm_limit: int) -> str:
    lang = detect(query)
    output_lang = "English" if lang == "en" else "Azerbaijani"
    no_answer_msg = "Please contact a professional" if output_lang == "English" else "Zəhmət olmasa, peşəkarla əlaqə saxlayın"

    system_prompt = base_prompt.format(output_lang, ', '.join(results_rule), ', '.join(results_law), ', '.join(results_post), mode, no_answer_msg, mode)
    full_input = system_prompt + "\n" + query
    token_count = len(encoder.encode(full_input))

    if token_count <= tpm_limit:
        return system_prompt

    adjusted_law = results_law.copy()
    adjusted_post = results_post.copy()
    adjusted_rule = results_rule.copy()

    while adjusted_law and token_count > tpm_limit:
        adjusted_law.pop(0)
        system_prompt = base_prompt.format(output_lang, ', '.join(adjusted_rule), ', '.join(adjusted_law), ', '.join(adjusted_post), mode, no_answer_msg, mode)
        full_input = system_prompt + "\n" + query
        token_count = len(encoder.encode(full_input))

    while adjusted_post and token_count > tpm_limit:
        adjusted_post.pop(0)
        system_prompt = base_prompt.format(output_lang, ', '.join(adjusted_rule), ', '.join(adjusted_law), ', '.join(adjusted_post), mode, no_answer_msg, mode)
        full_input = system_prompt + "\n" + query
        token_count = len(encoder.encode(full_input))

    while adjusted_rule and token_count > tpm_limit:
        adjusted_rule.pop(0)
        system_prompt = base_prompt.format(output_lang, ', '.join(adjusted_rule), ', '.join(adjusted_law), ', '.join(adjusted_post), mode, no_answer_msg, mode)
        full_input = system_prompt + "\n" + query
        token_count = len(encoder.encode(full_input))

    return system_prompt

# Query endpoint
@app.post("/query")
async def process_query(request: QueryRequest):
    query_input = request.query
    mode = "Summarized" if query_input.lower().startswith("summarized:") else "Detailed"
    query = query_input[len("Summarized:"):] if mode == "Summarized" else query_input

    # Generate embedding
    query_embedding = get_embedding(query)

    # Fetch results
    results_law = await fetch_results(law, query_embedding, "law")
    results_rule = await fetch_results(rule, query_embedding, "rule")
    results_post = await fetch_results(post, query_embedding, "post")

    # Adjust prompt
    system_prompt = adjust_prompt_tokens(base_prompt, results_rule, results_law, results_post, query, mode, TPM_LIMIT)

    if system_prompt in ["Please contact a professional", "Zəhmət olmasa, peşəkarla əlaqə saxlayın"]:
        return {"response": system_prompt}

    # Generate response
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}