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
    input_variables=["text", "assignment_title", "assignment_description"],
    template="""
    You're an experienced professor evaluating a student assignment.
    
    Assignment Title: {assignment_title}
    Assignment Description: {assignment_description}
    
    Read the student's assignment text below, and evaluate it based on clarity, depth, and relevance.
    
    ---
    {text}
    ---
    
    Provide your evaluation in the following format:
    SCORE: [number between 0-100]
    FEEDBACK: [2-3 sentences of constructive feedback]
    """
)

def evaluate_assignment_text(text: str, assignment_title: str, assignment_description: str) -> tuple:
    try:
        prompt = grade_prompt.format(
            text=text,
            assignment_title=assignment_title,
            assignment_description=assignment_description
        )
        result = llm.predict(prompt).strip()
        
        # Parse the result to extract score and feedback
        lines = result.split('\n')
        score_line = next((line for line in lines if line.startswith("SCORE:")), "SCORE: 0")
        score = int("".join([c for c in score_line if c.isdigit()]))
        score = max(0, min(score, 100))  # Clamp between 0-100
        
        # Get feedback (everything after FEEDBACK:)
        feedback_index = result.find("FEEDBACK:")
        feedback = result[feedback_index + 9:].strip() if feedback_index != -1 else "No feedback provided."
        
        return score, feedback
    except Exception as e:
        logging.error(f"AI grading failed: {e}")
        return None, "AI evaluation failed. Please review manually."