# backend/ai/ai_grader.py
import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Standard grading prompt (used when no rubric is available)
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

# Rubric-based grading prompt
rubric_grade_prompt = PromptTemplate(
    input_variables=["text", "assignment_title", "assignment_description", "rubric_json"],
    template="""
    You're an experienced professor evaluating a student assignment using a specific rubric.
   
    Assignment Title: {assignment_title}
    Assignment Description: {assignment_description}
    
    Below is the rubric to use for evaluation:
    {rubric_json}
   
    Read the student's assignment text below:
    ---
    {text}
    ---
   
    For each criterion in the rubric:
    1. Evaluate the student's work against that specific criterion
    2. Choose the most appropriate performance level based on the descriptions
    3. Provide brief feedback explaining your evaluation for that criterion
    
    Return your evaluation in valid JSON format as follows:
    ```json
    {{
      "criteria_evaluations": [
        {{
          "criterion_id": [id],
          "name": [criterion name],
          "selected_level_id": [id of selected level],
          "points_awarded": [points for the selected level],
          "feedback": [1-2 sentences of specific feedback for this criterion]
        }},
        ...
      ],
      "total_score": [weighted total score between 0-100],
      "overall_feedback": [2-3 sentences of overall constructive feedback]
    }}
    ```
    
    Make sure your response contains ONLY this JSON. Do not include any text before or after the JSON.
    """
)

def evaluate_assignment_text(text: str, assignment_title: str, assignment_description: str, 
                          rubric_criteria: List[Dict[str, Any]] = None) -> Tuple[float, str, Optional[List[Dict[str, Any]]]]:
    """
    Evaluate a student's assignment text with or without a rubric.
    
    Args:
        text: The student's submitted text
        assignment_title: Title of the assignment
        assignment_description: Description of the assignment
        rubric_criteria: List of rubric criteria with their levels (optional)
        
    Returns:
        tuple: (score, feedback, criteria_evaluations)
        - score: Overall score (0-100)
        - feedback: General feedback text
        - criteria_evaluations: Detailed evaluation per criterion (None if no rubric was used)
    """
    try:
        if not rubric_criteria:
            # Use standard evaluation if no rubric is provided
            return _evaluate_standard(text, assignment_title, assignment_description)
        else:
            # Use rubric-based evaluation
            return _evaluate_with_rubric(text, assignment_title, assignment_description, rubric_criteria)
    except Exception as e:
        logging.error(f"AI grading failed: {e}")
        return 0, "AI evaluation failed. Please review manually.", None

def _evaluate_standard(text: str, assignment_title: str, assignment_description: str) -> Tuple[float, str, None]:
    """Standard evaluation without rubrics"""
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
   
    return score, feedback, None

def _evaluate_with_rubric(text: str, assignment_title: str, assignment_description: str, 
                       rubric_criteria: List[Dict[str, Any]]) -> Tuple[float, str, List[Dict[str, Any]]]:
    """Rubric-based evaluation"""
    # Convert rubric criteria to JSON format for the prompt
    rubric_json = json.dumps(rubric_criteria, indent=2)
    
    prompt = rubric_grade_prompt.format(
        text=text,
        assignment_title=assignment_title,
        assignment_description=assignment_description,
        rubric_json=rubric_json
    )
    
    # Get AI evaluation response
    result = llm.predict(prompt).strip()
    
    # Extract JSON part from response if necessary
    json_start = result.find('{')
    json_end = result.rfind('}') + 1
    
    if json_start >= 0 and json_end > json_start:
        result_json = result[json_start:json_end]
    else:
        # Fallback if JSON structure isn't found
        raise ValueError("Failed to extract valid JSON from AI evaluation")
    
    # Parse the response
    try:
        evaluation = json.loads(result_json)
        total_score = float(evaluation.get('total_score', 0))
        overall_feedback = evaluation.get('overall_feedback', 'No feedback provided.')
        criteria_evaluations = evaluation.get('criteria_evaluations', [])
        
        # Validate and clamp the score
        total_score = max(0, min(total_score, 100))
        
        return total_score, overall_feedback, criteria_evaluations
    except json.JSONDecodeError:
        logging.error(f"Failed to parse AI evaluation JSON: {result_json}")
        return 0, "AI evaluation failed to provide structured feedback. Please review manually.", None

def prepare_rubric_for_ai(db_rubric_criteria):
    """
    Convert database rubric criteria format to the format needed for AI evaluation.
    
    Args:
        db_rubric_criteria: List of RubricCriterion objects from the database
        
    Returns:
        List of dictionaries with criteria and their levels formatted for the AI
    """
    rubric_for_ai = []
    
    for criterion in db_rubric_criteria:
        criterion_data = {
            "id": criterion.id,
            "name": criterion.name,
            "weight": criterion.weight,
            "levels": []
        }
        
        for level in criterion.levels:
            level_data = {
                "id": level.id,
                "description": level.description,
                "points": level.points
            }
            criterion_data["levels"].append(level_data)
        
        # Sort levels by points (typically ascending)
        criterion_data["levels"].sort(key=lambda x: x["points"])
        rubric_for_ai.append(criterion_data)
    
    return rubric_for_ai