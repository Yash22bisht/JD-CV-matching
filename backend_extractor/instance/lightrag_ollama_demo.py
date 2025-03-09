import asyncio
import nest_asyncio

nest_asyncio.apply()
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./resume_demo"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2m",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Insert example text
    with open("./knowledge.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Test different query modes
    print("\nNaive Search:")
    print(
        rag.query(
            '''Follow these rules:

1. **Job Title Flexibility**:
   - Include resumes with interchangeable titles (e.g., "Developer" vs. "Engineer") but prioritize exact matches.
   - Flag industry-specific title variations (e.g., "Data Scientist" vs. "ML Engineer") in reasoning.

2. **Skill Matching**:
   - **Exact Matches**: Rank highest for direct skill alignment (e.g., "React.js" in JD and resume).
   - **Prerequisite Skills**: Include resumes with foundational skills (e.g., HTML/CSS for React roles) but rank lower. Explicitly list missing core skills in missing_skills.

3. **Experience Hierarchy**:
   Prioritize sections as: Professional Experience > Open Source Contributions > Projects > No relevant sections. Highlight the strongest section in reasoning.

4. **Bias Mitigation**:
   - **Anonymization**: Ignore names, gender, age, race, schools, and locations.
   - **Inclusive Language Check**: Flag non-neutral JD terms (e.g., "ninja", "young graduates") in bias_checks.
   - **Fair Evaluation**: Strictly focus on skills, experience, and role-specific qualifications."

5. **Performance Metrics**:
   - **Precision**: Optimize for exact matches in top ranks.
   - **Recall**: Ensure relevant resumes (exact + prerequisites) are included.
   - **Diversity Score**: Aim for variation in candidate backgrounds.
   - **Speed**: Process resumes efficiently within context limits.

Output Format (JSON array):
[
  {
    "filename": "...",
    "score": 0-100,  # Exact match = 90-100, Prerequisites = 60-89
    "rank": 1-N,
    "reasoning": "...",  # E.g., "Strong React.js match + 5YOE at Fortune 500"
    "bias_checks": ["School names ignored", "No biased language detected"],
    "missing_skills": ["Skill1", "Skill2"]  # Only if applicable
  },
  ...
]

Return ONLY the JSON array. No additional text before or after.''', param=QueryParam(mode="naive")
        )
    )

    print("\nLocal Search:")
    print(
        rag.query(
            '''Follow these rules:

1. **Job Title Flexibility**:
   - Include resumes with interchangeable titles (e.g., "Developer" vs. "Engineer") but prioritize exact matches.
   - Flag industry-specific title variations (e.g., "Data Scientist" vs. "ML Engineer") in reasoning.

2. **Skill Matching**:
   - **Exact Matches**: Rank highest for direct skill alignment (e.g., "React.js" in JD and resume).
   - **Prerequisite Skills**: Include resumes with foundational skills (e.g., HTML/CSS for React roles) but rank lower. Explicitly list missing core skills in missing_skills.

3. **Experience Hierarchy**:
   Prioritize sections as: Professional Experience > Open Source Contributions > Projects > No relevant sections. Highlight the strongest section in reasoning.

4. **Bias Mitigation**:
   - **Anonymization**: Ignore names, gender, age, race, schools, and locations.
   - **Inclusive Language Check**: Flag non-neutral JD terms (e.g., "ninja", "young graduates") in bias_checks.
   - **Fair Evaluation**: Strictly focus on skills, experience, and role-specific qualifications."
   
5. **Performance Metrics**:
   - **Precision**: Optimize for exact matches in top ranks.
   - **Recall**: Ensure relevant resumes (exact + prerequisites) are included.
   - **Diversity Score**: Aim for variation in candidate backgrounds.
   - **Speed**: Process resumes efficiently within context limits.

Output Format (JSON array):
[
  {
    "filename": "...",
    "score": 0-100,  # Exact match = 90-100, Prerequisites = 60-89
    "rank": 1-N,
    "reasoning": "...",  # E.g., "Strong React.js match + 5YOE at Fortune 500"
    "bias_checks": ["School names ignored", "No biased language detected"],
    "missing_skills": ["Skill1", "Skill2"]  # Only if applicable
  },
  ...
]

Return ONLY the JSON array. No additional text before or after.''', param=QueryParam(mode="local")
        )
    )

    print("\nGlobal Search:")
    print(
        rag.query(
            '''Follow these rules:

1. **Job Title Flexibility**:
   - Include resumes with interchangeable titles (e.g., "Developer" vs. "Engineer") but prioritize exact matches.
   - Flag industry-specific title variations (e.g., "Data Scientist" vs. "ML Engineer") in reasoning.

2. **Skill Matching**:
   - **Exact Matches**: Rank highest for direct skill alignment (e.g., "React.js" in JD and resume).
   - **Prerequisite Skills**: Include resumes with foundational skills (e.g., HTML/CSS for React roles) but rank lower. Explicitly list missing core skills in missing_skills.

3. **Experience Hierarchy**:
   Prioritize sections as: Professional Experience > Open Source Contributions > Projects > No relevant sections. Highlight the strongest section in reasoning.

4. **Bias Mitigation**:
   - **Anonymization**: Ignore names, gender, age, race, schools, and locations.
   - **Inclusive Language Check**: Flag non-neutral JD terms (e.g., "ninja", "young graduates") in bias_checks.
   - **Fair Evaluation**: Strictly focus on skills, experience, and role-specific qualifications."
   
5. **Performance Metrics**:
   - **Precision**: Optimize for exact matches in top ranks.
   - **Recall**: Ensure relevant resumes (exact + prerequisites) are included.
   - **Diversity Score**: Aim for variation in candidate backgrounds.
   - **Speed**: Process resumes efficiently within context limits.

Output Format (JSON array):
[
  {
    "filename": "...",
    "score": 0-100,  # Exact match = 90-100, Prerequisites = 60-89
    "rank": 1-N,
    "reasoning": "...",  # E.g., "Strong React.js match + 5YOE at Fortune 500"
    "bias_checks": ["School names ignored", "No biased language detected"],
    "missing_skills": ["Skill1", "Skill2"]  # Only if applicable
  },
  ...
]

Return ONLY the JSON array. No additional text before or after.''', param=QueryParam(mode="global")
        )
    )

    print("\nHybrid Search:")
    print(
        rag.query(
            '''Follow these rules:

1. **Job Title Flexibility**:
   - Include resumes with interchangeable titles (e.g., "Developer" vs. "Engineer") but prioritize exact matches.
   - Flag industry-specific title variations (e.g., "Data Scientist" vs. "ML Engineer") in reasoning.

2. **Skill Matching**:
   - **Exact Matches**: Rank highest for direct skill alignment (e.g., "React.js" in JD and resume).
   - **Prerequisite Skills**: Include resumes with foundational skills (e.g., HTML/CSS for React roles) but rank lower. Explicitly list missing core skills in missing_skills.

3. **Experience Hierarchy**:
   Prioritize sections as: Professional Experience > Open Source Contributions > Projects > No relevant sections. Highlight the strongest section in reasoning.

4. **Bias Mitigation**:
   - **Anonymization**: Ignore names, gender, age, race, schools, and locations.
   - **Inclusive Language Check**: Flag non-neutral JD terms (e.g., "ninja", "young graduates") in bias_checks.
   - **Fair Evaluation**: Strictly focus on skills, experience, and role-specific qualifications."
   
5. **Performance Metrics**:
   - **Precision**: Optimize for exact matches in top ranks.
   - **Recall**: Ensure relevant resumes (exact + prerequisites) are included.
   - **Diversity Score**: Aim for variation in candidate backgrounds.
   - **Speed**: Process resumes efficiently within context limits.

Output Format (JSON array):
[
  {
    "filename": "...",
    "score": 0-100,  # Exact match = 90-100, Prerequisites = 60-89
    "rank": 1-N,
    "reasoning": "...",  # E.g., "Strong React.js match + 5YOE at Fortune 500"
    "bias_checks": ["School names ignored", "No biased language detected"],
    "missing_skills": ["Skill1", "Skill2"]  # Only if applicable
  },
  ...
]

Return ONLY the JSON array. No additional text before or after.''', param=QueryParam(mode="hybrid")
        )
    )

    # stream response
    resp = rag.query(
        '''Follow these rules:

1. **Job Title Flexibility**:
   - Include resumes with interchangeable titles (e.g., "Developer" vs. "Engineer") but prioritize exact matches.
   - Flag industry-specific title variations (e.g., "Data Scientist" vs. "ML Engineer") in reasoning.

2. **Skill Matching**:
   - **Exact Matches**: Rank highest for direct skill alignment (e.g., "React.js" in JD and resume).
   - **Prerequisite Skills**: Include resumes with foundational skills (e.g., HTML/CSS for React roles) but rank lower. Explicitly list missing core skills in missing_skills.

3. **Experience Hierarchy**:
   Prioritize sections as: Professional Experience > Open Source Contributions > Projects > No relevant sections. Highlight the strongest section in reasoning.

4. **Bias Mitigation**:
   - **Anonymization**: Ignore names, gender, age, race, schools, and locations.
   - **Inclusive Language Check**: Flag non-neutral JD terms (e.g., "ninja", "young graduates") in bias_checks.
   - **Fair Evaluation**: Strictly focus on skills, experience, and role-specific qualifications."
   
5. **Performance Metrics**:
   - **Precision**: Optimize for exact matches in top ranks.
   - **Recall**: Ensure relevant resumes (exact + prerequisites) are included.
   - **Diversity Score**: Aim for variation in candidate backgrounds.
   - **Speed**: Process resumes efficiently within context limits.

Output Format (JSON array):
[
  {
    "filename": "...",
    "score": 0-100,  # Exact match = 90-100, Prerequisites = 60-89
    "rank": 1-N,
    "reasoning": "...",  # E.g., "Strong React.js match + 5YOE at Fortune 500"
    "bias_checks": ["School names ignored", "No biased language detected"],
    "missing_skills": ["Skill1", "Skill2"]  # Only if applicable
  },
  ...
]

Return ONLY the JSON array. No additional text before or after.''',
        param=QueryParam(mode="hybrid", stream=True),
    )

    if inspect.isasyncgen(resp):
        asyncio.run(print_stream(resp))
    else:
        print(resp)


if __name__ == "__main__":
    main()
