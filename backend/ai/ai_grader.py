# backend/ai/ai_grader.py
import os
import logging
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

grade_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You're an experienced professor. Read the student's assignment text below, and assign a score from 0 to 100 based on clarity, depth, and relevance.

    Return ONLY the number. No extra commentary.

    ---
    {text}
    ---
    Score:
    """
)

def evaluate_assignment_text(text: str) -> int:
    try:
        prompt = grade_prompt.format(text=text)
        result = llm.predict(prompt).strip()

        score = int("".join([c for c in result if c.isdigit()]))
        return max(0, min(score, 100))  # Clamp between 0-100
    except Exception as e:
        logging.error(f"AI grading failed: {e}")
        return None
