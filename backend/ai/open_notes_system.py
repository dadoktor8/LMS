from datetime import date, datetime
from typing import List, Dict, Union, Optional
import uuid
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import SQLChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import re
import json
from pydantic import BaseModel, Field

from backend.ai.aws_ai import load_faiss_vectorstore, upload_file_to_s3_from_path
from backend.db.database import get_db
from backend.db.models import CourseModule, CourseSubmodule, ModuleTextChunk, QuizQuota

# Define models for our study materials
class FlashCard(BaseModel):
    question: str
    answer: str
    tags: List[str] = Field(default_factory=list)

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer_index: int
    explanation: str

class Quiz(BaseModel):
    title: str
    description: str
    questions: List[QuizQuestion]

class StudyGuide(BaseModel):
    title: str
    sections: List[Dict[str, str]]  # Each dict has 'heading' and 'content' keys
    summary: str

class StudyMaterials(BaseModel):
    material_type: str  # 'flashcards', 'quiz', or 'study_guide'
    content: Union[List[FlashCard], Quiz, StudyGuide]

'''def generate_study_materials(query: str, material_type: str, course_id: int, student_id: str) -> str:
    """
    Generate interactive study materials based on course content
    
    Args:
        query: Topic or subject to create materials for
        material_type: Type of material ('flashcards', 'quiz', or 'study_guide')
        course_id: Course identifier
        student_id: Student identifier
        
    Returns:
        JSON string containing the requested study materials
    """
    try:
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not isinstance(course_id, int) or course_id <= 0:
            raise ValueError("Course ID must be a positive integer")
        if not student_id or not isinstance(student_id, str):
            raise ValueError("Student ID must be a non-empty string")
        if material_type not in ["flashcards", "quiz", "study_guide"]:
            raise ValueError("Material type must be 'flashcards', 'quiz', or 'study_guide'")
            
        # Load FAISS index
        try:
            index_path = f"faiss_index_{course_id}"
            db = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 10})  # Increased k for more context
        except FileNotFoundError:
            raise ValueError(f"FAISS index not found for course ID: {course_id}")
        except Exception as e:
            raise ConnectionError(f"Error loading FAISS index: {str(e)}")
            
        # Set up OpenAI GPT-4
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        try:
            chat = ChatOpenAI(
                model="gpt-4.1-mini",  # Could use a more powerful model if available
                temperature=0.2,
                openai_api_key=api_key
            )
        except Exception as e:
            raise ConnectionError(f"Error initializing OpenAI client: {str(e)}")
        
        # Create appropriate prompt based on material type
        if material_type == "flashcards":
            print("PROMPT FOR FLASHCARDS!", query)
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are creating flashcards for a student on a specific topic using their course materials.
                
                Based on the course materials below, create 10 flashcards on the topic: {question}
                
                Each flashcard should have:
                1. A clear, concise question
                2. A comprehensive but brief answer
                3. 1-3 relevant tags or categories
                
                Format your output as a JSON array of flashcard objects with the following structure:
                [
                  {{
                    "question": "Question text here",
                    "answer": "Answer text here",
                    "tags": ["tag1", "tag2"]
                  }},
                  ...more flashcards...
                ]
                
                Course Materials:
                {context}
                
                Topic: {question}
                
                Flashcards (JSON format):
                """
            )
        elif material_type == "quiz":
            print("PROMPT FOR QUIZ!", query)
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are creating a quiz for a student on a specific topic using their course materials.
                
                Based on the course materials below, create a quiz of 5 multiple-choice questions on the topic: {question}
                
                Format your output as a JSON object with the following structure:
                {{
                  "title": "Quiz title here",
                  "description": "Brief description of the quiz",
                  "questions": [
                    {{
                      "question": "Question text here",
                      "options": ["Option A", "Option B", "Option C", "Option D"],
                      "correct_answer_index": 0,  // Index of the correct answer (0-based)
                      "explanation": "Explanation of why the answer is correct"
                    }},
                    ...more questions...
                  ]
                }}
                
                Course Materials:
                {context}
                
                Topic: {question}
                
                Quiz (JSON format):
                """
            )
        else:  # study_guide
            print("PROMPT FOR STUDY GUIDE!", query)
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are creating a comprehensive study guide for a student on a specific topic using their course materials.
                
                Based on the course materials below, create a structured study guide on the topic: {question}
                
                Format your output as a JSON object with the following structure:
                {{
                  "title": "Study Guide title here",
                  "sections": [
                    {{
                      "heading": "Section heading",
                      "content": "Detailed content with key points, examples, and explanations"
                    }},
                    ...more sections...
                  ],
                  "summary": "A brief summary of the key points covered in this study guide"
                }}
                
                Course Materials:
                {context}
                
                Topic: {question}
                
                Study Guide (JSON format):
                """
            )
        
        # RetrievalQA chain
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=chat,
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt_template, "verbose": False},
                chain_type="stuff"
            )
            
            # Get response with timeout
            result = qa_chain({"query": query})
            response = result.get("result", "")
            print("LLM RAW RESPONSE:", response[:800])
            
            if not response:
                return json.dumps({"error": "I couldn't generate study materials. Please try again or rephrase your request."})
                
        except TimeoutError:
            return json.dumps({"error": "The request timed out. Please try again or request materials on a more specific topic."})
        except Exception as e:
            raise ConnectionError(f"Error during materials generation: {str(e)}")
            
        # Clean and validate JSON
        try:
            # Clean up any non-JSON text that might be in the response
            json_start = response.find("[") if material_type == "flashcards" else response.find("{")
            json_end = response.rfind("]") + 1 if material_type == "flashcards" else response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_response = response[json_start:json_end]
                # Parse and re-serialize to ensure valid JSON
                parsed_json = json.loads(json_response)
                return json.dumps(parsed_json)
            else:
                raise ValueError("Could not extract valid JSON from response")
                
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": "Could not parse response as JSON. Please try again.",
                "raw_response": response[:200] + "..." if len(response) > 200 else response
            })
            
        # Record activity in history (optional)
        session_id = f"{student_id}_{course_id}"
        try:
            sql_history = SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat_history.db")
            sql_history.add_user_message(f"Generate {material_type} on: {query}")
            sql_history.add_ai_message(f"Generated {material_type} materials successfully")
        except Exception as e:
            print(f"Warning: Could not save activity to chat history: {e}")
            
        return response
        
    except ValueError as e:
        error_msg = f"Input error: {str(e)}"
        print(error_msg)
        return json.dumps({"error": f"There was an issue with your request: {str(e)}. Please contact support if this continues."})
        
    except ConnectionError as e:
        error_msg = f"Connection error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise
        
    except Exception as e:
        error_msg = f"Unexpected error in study materials generation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return json.dumps({"error": "I encountered an unexpected error. Our team has been notified, and we're working to fix it."})'''


def generate_study_material(query: str, material_type: str, course_id: int, student_id: str, module_id: Optional[int] = None) -> str:
    # --- Step 1: Validate inputs ---
    if not query or not query.strip():
        return json.dumps({"error": "Query cannot be empty."})
    if not isinstance(course_id, int) or course_id <= 0:
        return json.dumps({"error": "Course ID must be a positive integer."})
    if not student_id or not isinstance(student_id, str):
        return json.dumps({"error": "Student ID must be a non-empty string."})
    if material_type not in {"flashcards", "quiz", "study_guide"}:
        return json.dumps({"error": "Invalid material type."})
    
    # --- Step 2: Get context using the enhanced retrieve_course_context function ---
    try:
        context_result = retrieve_course_context(course_id, query, module_id)
        
        # Check if we have an error (no content available)
        if isinstance(context_result, dict) and "error" in context_result:
            return json.dumps({"error": f"Content retrieval failed: {context_result['error']}"})
        
        context = context_result
        if not context.strip():
            return json.dumps({"error": "Insufficient context found for your query."})
            
    except Exception as e:
        return json.dumps({"error": f"Error retrieving course content: {str(e)}"})
    
    # --- Step 3: Setup OpenAI Client ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return json.dumps({"error": "OpenAI API key missing in environment variables."})
    try:
        chat = ChatOpenAI(
            model="gpt-4.1-mini",  
            temperature=0.2,
            openai_api_key=api_key
        )
    except Exception as e:
        return json.dumps({"error": f"Could not setup OpenAI Client: {str(e)}"})
    
    # --- Step 4: Fetch Prompt Template ---
    try:
        prompt_template = get_prompt_for_material_type(material_type, module_id)
        if not prompt_template:
            return json.dumps({"error": "Invalid or missing prompt template."})
    except Exception as e:
        return json.dumps({"error": f"Error fetching Prompt: {str(e)}"})
    
    # --- Step 5: Generate Study Material ---
    try:
        # Use direct LLM call with context instead of retrieval chain
        formatted_prompt = prompt_template.format(context=context, question=query)
        response = chat.predict(formatted_prompt).strip()
        
        if not response:
            return json.dumps({"error": "No response from LLM. Please retry with a different or more specific query."})
    except TimeoutError:
        return json.dumps({"error": "OpenAI response timed out. Retry with simpler or shorter query."})
    except Exception as e:
        return json.dumps({"error": f"LLM generation failed: {str(e)}"})
    
    # --- Step 6: Robust JSON Parsing ---
    try:
        # Extract JSON based on type (flashcards are [], others {})
        start_char, end_char = ("[", "]") if material_type == "flashcards" else ("{", "}")
        json_start = response.find(start_char)
        json_end = response.rfind(end_char) + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("Invalid/malformed JSON from LLM response.")
        json_content = response[json_start:json_end]
        # Validate JSON parsing
        parsed_json = json.loads(json_content)
        return json.dumps(parsed_json, indent=2)
    except (json.JSONDecodeError, ValueError) as e:
        return json.dumps({
            "error": f"Generated material isn't parsable JSON: {str(e)}",
            "raw_response": response
        })
    finally:
        log_generation_event(student_id, course_id, material_type, query)

def get_prompt_for_material_type(material_type: str, module_id: Optional[int] = None):
    """Get the appropriate prompt template based on material type and context."""
    
    if material_type == "study_guide":
        if module_id:
            template_text = """
            You are an expert educational content creator. Create a comprehensive study guide based on the specific module content.
            
            The study guide should be well-structured, informative, and easy to follow.
            
            Course Content:
            {context}
            
            Topic: {question}
            
            Create a study guide in the following JSON format:
            {{
                "title": "Study Guide: [Topic Name]",
                "sections": [
                    {{
                        "heading": "Section Title",
                        "points": ["Key point 1", "Key point 2", "Key point 3"]
                    }}
                ],
                "summary": "Brief summary of the key takeaways"
            }}
            
            Make sure to:
            1. Include 4-6 main sections
            2. Each section should have 3-5 key points
            3. Focus on the most important concepts
            4. Use clear, student-friendly language
            5. Include practical examples where relevant
            6. Provide a concise summary at the end
            
            Return only the JSON object, no additional text.
            """
        else:
            template_text = """
            You are an expert educational content creator. Create a comprehensive study guide based on the course materials.
            
            The study guide should be well-structured, informative, and easy to follow.
            
            Course Content:
            {context}
            
            Topic: {question}
            
            Create a study guide in the following JSON format:
            {{
                "title": "Study Guide: [Topic Name]",
                "sections": [
                    {{
                        "heading": "Section Title",
                        "points": ["Key point 1", "Key point 2", "Key point 3"]
                    }}
                ],
                "summary": "Brief summary of the key takeaways"
            }}
            
            Make sure to:
            1. Include 4-6 main sections
            2. Each section should have 3-5 key points
            3. Focus on the most important concepts
            4. Use clear, student-friendly language
            5. Include practical examples where relevant
            6. Provide a concise summary at the end
            
            Return only the JSON object, no additional text.
            """
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template_text
        )
        
    elif material_type == "flashcards":
        if module_id:
            template_text = """
            You are an expert educational content creator. Create flashcards based on the specific module content.
            
            Course Content:
            {context}
            
            Topic: {question}
            
            Create flashcards in the following JSON format:
            [
                {{
                    "front": "Question or term",
                    "back": "Answer or definition"
                }}
            ]
            
            Make sure to:
            1. Create 8-12 flashcards
            2. Cover key concepts, definitions, and important facts
            3. Use clear, concise language
            4. Include both factual and conceptual questions
            5. Make questions specific and answerable
            
            Return only the JSON array, no additional text.
            """
        else:
            template_text = """
            You are an expert educational content creator. Create flashcards based on the course materials.
            
            Course Content:
            {context}
            
            Topic: {question}
            
            Create flashcards in the following JSON format:
            [
                {{
                    "front": "Question or term",
                    "back": "Answer or definition"
                }}
            ]
            
            Make sure to:
            1. Create 8-12 flashcards
            2. Cover key concepts, definitions, and important facts
            3. Use clear, concise language
            4. Include both factual and conceptual questions
            5. Make questions specific and answerable
            
            Return only the JSON array, no additional text.
            """
            
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template_text
        )
    else:
        # Default case or other material types
        return None

def log_generation_event(student_id: str, course_id: int, material_type: str, query: str):
    """Log the generation of study materials"""
    try:
        # You can implement logging to a database or file here
        print(f"Generated {material_type} for student {student_id} in course {course_id} on topic '{query}'")
    except Exception as e:
        print(f"Error logging generation event: {e}")

def render_flashcards_htmx(materials_json):
    """Render responsive, square, Tailwind-styled flashcards with error handling."""
    try:
        materials = json.loads(materials_json)
        
        # Handle error cases
        if "error" in materials:
            return f'''
            <div class="text-center py-8">
                <div class="text-red-600 font-semibold mb-2">⚠️ Error</div>
                <p class="text-gray-700">{materials["error"]}</p>
            </div>
            '''
        
        html = '''
        <div class="w-full mx-auto px-2 py-6">
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-12 place-items-center">
        '''
        for i, card in enumerate(materials):
            # Handle both old format (question/answer) and new format (front/back)
            front_text = card.get("front", card.get("question", "No question"))
            back_text = card.get("back", card.get("answer", "No answer"))
            
            html += f'''
            <div class="perspective w-[23rem] h-[23rem] max-w-full" id="card-{i}">
                <div class="flashcard-inner w-full h-full" id="inner-{i}">
                    <!-- Front -->
                    <div class="flashcard-front absolute inset-0 bg-white border-2 border-blue-300 shadow-xl rounded-2xl flex flex-col h-full w-full justify-between items-center p-8 [backface-visibility:hidden]">
                        <div class="w-full flex-1 flex flex-col justify-center items-center">
                            <h3 class="text-2xl font-bold text-blue-800 text-center break-words">{front_text}</h3>
                        </div>
                        <div class="w-full flex justify-center mt-4">
                            <button type="button"
                                onclick="flipCard({i})"
                                class="px-6 py-2 rounded-lg bg-blue-100 text-blue-700 font-semibold hover:bg-blue-200 transition duration-300 text-base shadow-md">
                                Show Answer
                            </button>
                        </div>
                    </div>
                    <!-- Back -->
                    <div class="flashcard-back absolute inset-0 bg-blue-50 border-2 border-blue-300 shadow-xl rounded-2xl flex flex-col h-full w-full justify-between items-center p-8 [backface-visibility:hidden]" style="transform: rotateY(180deg);">
                        <div class="w-full flex-1 flex flex-col justify-center items-center">
                            <div class="text-xl font-semibold text-green-800 text-center break-words">{back_text}</div>
                            <div class="mt-6 flex flex-wrap justify-center gap-2">
                                {''.join([f'<span class="inline-block bg-green-100 text-green-700 px-3 py-1 rounded-full text-sm font-medium">{tag}</span>' for tag in card.get("tags", [])])}
                            </div>
                        </div>
                        <div class="w-full flex justify-center mt-4">
                            <button type="button"
                                onclick="flipCard({i})"
                                class="px-6 py-2 rounded-lg bg-blue-100 text-blue-700 font-semibold hover:bg-blue-200 transition duration-300 text-base shadow-md">
                                Show Question
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        '''
        html += '''
            </div>
        </div>
        <script>
        function flipCard(index) {
            const inner = document.getElementById(`inner-${index}`);
            inner.classList.toggle('flipped');
        }
        </script>
        <style>
        .perspective {
            perspective: 1500px;
        }
        .flashcard-inner {
            position: relative;
            width: 100%;
            height: 100%;
            transition: transform 0.8s;
            transform-style: preserve-3d;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border-radius: 1rem;
        }
        .flashcard-front, .flashcard-back {
            backface-visibility: hidden;
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            display: flex;
            flex-direction: column;
            border-radius: 1rem;
        }
        .flashcard-back {
            transform: rotateY(180deg);
        }
        .flashcard-inner.flipped {
            transform: rotateY(180deg);
        }
        </style>
        '''
        return html
    except Exception as e:
        return f"<div class='text-red-600 font-bold'>Error rendering flashcards: {str(e)}</div>"

def render_quiz_htmx(materials_json):
    """
    Render quiz in HTMX format with Tailwind CSS classes, supporting multiple question types.
    Includes robust error handling and validation.
    """
    try:
        # Basic validation
        if not materials_json or not materials_json.strip():
            return "<div class='p-4 bg-red-100 text-red-700 rounded-lg'>Error: No quiz data provided</div>"
        
        # Parse the JSON with error handling
        try:
            materials = json.loads(materials_json)
        except json.JSONDecodeError as e:
            return f"<div class='p-4 bg-red-100 text-red-700 rounded-lg'>Error parsing quiz data: {str(e)}</div>"
        
        # Check if there's an error message in the materials
        if isinstance(materials, dict) and "error" in materials:
            return f"<div class='p-4 bg-red-100 text-red-700 rounded-lg'>Error: {materials['error']}</div>"
        
        # Validate required fields
        required_fields = ["title", "description", "questions"]
        missing_fields = [field for field in required_fields if field not in materials]
        
        if missing_fields:
            missing_list = ", ".join(missing_fields)
            return f"<div class='p-4 bg-red-100 text-red-700 rounded-lg'>Error: Missing required fields: {missing_list}</div>"
        
        # Validate questions structure
        if not isinstance(materials["questions"], list):
            return "<div class='p-4 bg-red-100 text-red-700 rounded-lg'>Error: 'questions' must be a list</div>"
        
        # If we have an empty quiz, show a message instead of an error
        if len(materials["questions"]) == 0:
            return "<div class='p-4 bg-yellow-100 text-yellow-700 rounded-lg'>No questions available in this quiz.</div>"
            
        # Begin rendering the quiz HTML
        html = f'''
        <div class="max-w-2xl mx-auto">
          <h2 class="text-2xl font-bold text-blue-800 mb-2">{html_escape(materials["title"])}</h2>
          <p class="text-gray-700 mb-8 text-lg">{html_escape(materials["description"])}</p>
          <form id="quiz-form" class="space-y-8">
        '''
        
        # Render questions
        for i, question in enumerate(materials["questions"]):
            # Validate each question has the minimum required fields
            if "question" not in question:
                continue
                
            question_type = question.get("type", "mcq")
            
            html += f'''
            <div class="mb-8 p-5 rounded-xl bg-gray-50 border border-gray-200 shadow">
              <h3 class="text-lg mb-4 font-semibold text-gray-800">Question {i+1}: {html_escape(question["question"])}</h3>
              <div class="flex flex-col gap-3">
            '''
            
            # Render different question types
            if question_type == "mcq":
                # Validate options exist
                if "options" not in question or not isinstance(question["options"], list) or len(question["options"]) == 0:
                    html += "<p class='text-red-600'>Error: Missing options for multiple choice question</p>"
                else:
                    # Multiple Choice Question (Single Answer)
                    for j, option in enumerate(question["options"]):
                        html += f'''
                        <label class="flex items-center gap-3 cursor-pointer text-gray-700 text-base">
                          <input type="radio"
                                 id="q{i}-o{j}"
                                 name="q{i}"
                                 value="{j}"
                                 class="form-radio h-4 w-4 text-blue-600 transition"
                          >
                          {html_escape(option)}
                        </label>
                        '''
            elif question_type == "msq":
                # Validate options exist
                if "options" not in question or not isinstance(question["options"], list) or len(question["options"]) == 0:
                    html += "<p class='text-red-600'>Error: Missing options for multiple select question</p>"
                else:
                    # Multiple Select Question (Multiple Answers)
                    for j, option in enumerate(question["options"]):
                        html += f'''
                        <label class="flex items-center gap-3 cursor-pointer text-gray-700 text-base">
                          <input type="checkbox"
                                 id="q{i}-o{j}"
                                 name="q{i}[]"
                                 value="{j}"
                                 class="form-checkbox h-4 w-4 text-blue-600 transition"
                          >
                          {html_escape(option)}
                        </label>
                        '''
            elif question_type == "essay":
                # Essay/Open-ended Question
                html += f'''
                <textarea
                    id="q{i}-essay"
                    name="q{i}-essay"
                    rows="5"
                    placeholder="Type your answer here..."
                    class="w-full border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                ></textarea>
                '''
            elif question_type == "short_answer":
                # Short Answer Question
                html += f'''
                <input type="text"
                    id="q{i}-short"
                    name="q{i}-short"
                    placeholder="Enter your answer"
                    class="w-full border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                '''
            else:
                # Unrecognized question type
                html += f"<p class='text-red-600'>Error: Unrecognized question type '{html_escape(question_type)}'</p>"
            
            html += f'''
              </div>
              <div class="explanation hidden mt-4 rounded-lg px-4 py-3 text-base" id="explanation-{i}"></div>
            </div>
            '''
            
        html += '''
          <div class="flex justify-between pt-2">
            <button type="button"
              onclick="checkAnswers()"
              class="py-3 px-8 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition text-lg"
            >
              Check Answers
            </button>
        '''
        
        # Add export buttons for teachers
        html += '''
            <div class="flex gap-2">
              <button type="button"
                onclick="exportQuiz('pdf', false)"
                class="py-3 px-6 bg-gray-600 text-white font-semibold rounded-lg hover:bg-gray-700 transition"
              >
                Export Question Paper
              </button>
              <button type="button"
                onclick="exportQuiz('pdf', true)"
                class="py-3 px-6 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition"
              >
                Export with Answer Key
              </button>
            </div>
          </div>
          </form>
        </div>
        <script>
        function checkAnswers() {
          const quizData = JSON.parse(`''' + materials_json.replace('`', '\\`') + '''`);
          let score = 0;
          let totalAnswerable = 0;
          
          quizData.questions.forEach((question, i) => {
            const questionType = question.type || "mcq";
            const explanationDiv = document.getElementById(`explanation-${i}`);
            
            if (!explanationDiv) return; // Skip if explanation div doesn't exist
            
            if (questionType === "mcq") {
              totalAnswerable++;
              const selectedOption = document.querySelector(`input[name="q${i}"]:checked`);
              
              if (!selectedOption) {
                explanationDiv.innerHTML = "Please select an answer";
                explanationDiv.className = "explanation warning mt-4 bg-yellow-100 border border-yellow-300 text-yellow-900";
                explanationDiv.classList.remove("hidden");
                return;
              }
              
              const selectedIndex = parseInt(selectedOption.value);
              const correctIndex = question.correct_answer_index || 0;
              
              if (selectedIndex === correctIndex) {
                score++;
                explanationDiv.innerHTML = "✅ <b>Correct!</b> " + (question.explanation || "");
                explanationDiv.className = "explanation correct mt-4 bg-green-100 border border-green-300 text-green-900";
              } else {
                explanationDiv.innerHTML = "❌ <b>Incorrect.</b> " + (question.explanation || "");
                explanationDiv.className = "explanation incorrect mt-4 bg-red-100 border border-red-300 text-red-900";
              }
              explanationDiv.classList.remove("hidden");
            }
            else if (questionType === "msq") {
              totalAnswerable++;
              const selectedOptions = Array.from(document.querySelectorAll(`input[name="q${i}[]"]:checked`))
                                         .map(input => parseInt(input.value));
              
              if (selectedOptions.length === 0) {
                explanationDiv.innerHTML = "Please select at least one answer";
                explanationDiv.className = "explanation warning mt-4 bg-yellow-100 border border-yellow-300 text-yellow-900";
                explanationDiv.classList.remove("hidden");
                return;
              }
              
              // Check if selected options match correct answers
              const correctAnswers = question.correct_answer_indices || [];
              const isCorrect = 
                selectedOptions.length === correctAnswers.length && 
                selectedOptions.every(opt => correctAnswers.includes(opt));
              
              if (isCorrect) {
                score++;
                explanationDiv.innerHTML = "✅ <b>Correct!</b> " + (question.explanation || "");
                explanationDiv.className = "explanation correct mt-4 bg-green-100 border border-green-300 text-green-900";
              } else {
                explanationDiv.innerHTML = "❌ <b>Incorrect.</b> " + (question.explanation || "");
                explanationDiv.className = "explanation incorrect mt-4 bg-red-100 border border-red-300 text-red-900";
              }
              explanationDiv.classList.remove("hidden");
            }
            else if (questionType === "essay" || questionType === "short_answer") {
              // For essay and short answer, just show the model answer
              const answerField = questionType === "essay" 
                ? document.getElementById(`q${i}-essay`)
                : document.getElementById(`q${i}-short`);
              
              if (answerField && answerField.value.trim()) {
                explanationDiv.innerHTML = "<b>Model Answer:</b> " + (question.model_answer || "");
                explanationDiv.className = "explanation model mt-4 bg-blue-100 border border-blue-300 text-blue-900";
                explanationDiv.classList.remove("hidden");
              } else {
                explanationDiv.innerHTML = "Please provide an answer";
                explanationDiv.className = "explanation warning mt-4 bg-yellow-100 border border-yellow-300 text-yellow-900";
                explanationDiv.classList.remove("hidden");
              }
            }
          });
          
          // Only show score if there are answerable questions
          if (totalAnswerable > 0) {
            // Remove previous score if any
            document.querySelectorAll('.quiz-score').forEach(div=>div.remove());
            
            // Show score at the end of the form
            const scoreDiv = document.createElement("div");
            scoreDiv.className = "quiz-score mt-6 py-4 rounded-lg bg-blue-50 border border-blue-200 text-blue-900 text-center text-lg font-bold";
            scoreDiv.innerHTML = `Your score: <span class='text-blue-700 font-extrabold'>${score}</span> / <span class="text-blue-700">${totalAnswerable}</span>`;
            document.getElementById('quiz-form').appendChild(scoreDiv);
          }
        }
        
        function exportQuiz(format, includeAnswers) {
          try {
          const course_id = getCourseIdFromPath();
          if (!course_id) {
            alert("Course ID not found!");
            return;
          }

            window.location.href = `/ai/courses/${course_id}/quiz/export?format=${format}&include_answers=${includeAnswers}`;
            //window.location.href = `/ai/courses/${course_id}/quiz/export`; 
          } catch (error) {
            console.error("Export failed:", error);
            alert("Failed to export quiz. Please try again.");
          }
        }
        function getCourseIdFromPath() {
            const match = window.location.pathname.match(/courses\/(\d+)/);
            return match ? match[1] : null;
        }
        function getUrlParam(param) {
          const urlParams = new URLSearchParams(window.location.search);
          return urlParams.get(param);
        }
        </script>
        '''
        return html
    except Exception as e:
        return f"<div class='p-4 bg-red-100 text-red-700 rounded-lg'>Error rendering quiz: {str(e)}</div>"

def html_escape(text):
    """Escape HTML special characters in text"""
    if not isinstance(text, str):
        text = str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

def render_study_guide_htmx(materials_json):
    """Render study guide HTML (Tailwind-friendly) ready for drop-in into your studyguide.html template."""
    try:
        materials = json.loads(materials_json)
        
        # Handle error cases
        if "error" in materials:
            return f'''
            <div class="text-center py-8">
                <div class="text-red-600 font-semibold mb-2">⚠️ Error</div>
                <p class="text-gray-700">{materials["error"]}</p>
            </div>
            '''
        
        html = f'''
        <div>
            <h2 class="text-2xl font-bold text-blue-800 mb-6">{materials["title"]}</h2>
            <!-- Table of Contents -->
            <div class="mb-8">
                <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">📘 Table of Contents</h3>
                <ul class="list-disc pl-6 space-y-1">
        '''
        # Table of Contents
        for i, section in enumerate(materials["sections"]):
            html += f'<li><a href="#section-{i}" class="text-blue-600 hover:underline">{section["heading"]}</a></li>'
        html += '''
                </ul>
            </div>
            <!-- Study Guide Content -->
            <div>
        '''
        for i, section in enumerate(materials["sections"]):
            html += f'''
            <div class="mb-8" id="section-{i}">
                <h3 class="text-xl font-semibold text-gray-800 mb-2">{section["heading"]}</h3>
                <ul class="list-inside list-disc text-base space-y-2 pl-4">
            '''
            for point in section["points"]:
                html += f'<li>{point}</li>'
            html += '''
                </ul>
            </div>
            '''
        # Summary Section
        html += f'''
            <div class="mt-10">
                <h3 class="text-xl font-semibold mb-2 flex items-center gap-2">📝 Summary</h3>
                <p class="text-base text-gray-700">{materials["summary"]}</p>
            </div>
            </div>
            <div class="flex justify-center mt-8">
                <button onclick="window.print()" class="px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition flex items-center gap-2 text-lg">
                    🖨️ Print Study Guide
                </button>
            </div>
        </div>
        '''
        return html
    except Exception as e:
        return f"<div class='text-red-600 font-bold'>Error rendering study guide: {str(e)}</div>"

def generate_study_material_quiz(
    query: str,
    material_type: str,
    course_id: int,
    teacher_id: str = None,
    question_types: list = None,
    num_questions: int = 10,
    module_id: int = None,        # New parameter
    difficulty: str = "medium",   # New parameter
    use_dummy: bool = False
) -> str:
    """
    Generate study material quiz with module selection and difficulty levels.
    
    Args:
        query: Topic or question to generate quiz about
        material_type: Type of material (should be "quiz" for this function)
        course_id: ID of the course
        teacher_id: ID of the teacher creating the quiz
        question_types: List of question types to include
        num_questions: Number of questions to generate
        module_id: Optional module ID to limit content to specific module
        difficulty: Difficulty level ("easy", "medium", "hard")
        use_dummy: Whether to use dummy data instead of AI generation
    
    Returns:
        JSON string containing the generated quiz
    """
    # Check if module is selected but no query is provided
    if module_id and (not query or not query.strip()):
        # Get the module title to use as the query
        try:
            db_session = next(get_db())
            module = db_session.query(CourseModule).filter_by(id=module_id, course_id=course_id).first()
            if module:
                query = module.title
            else:
                return json.dumps({"error": "Selected module not found."})
        except Exception as e:
            return json.dumps({"error": f"Error retrieving module information: {str(e)}"})
    
    # Validate inputs
    if not query or not query.strip():
        return json.dumps({"error": "Query cannot be empty."})
    if not isinstance(course_id, int) or course_id <= 0:
        return json.dumps({"error": "Course ID must be a positive integer."})
    if material_type != "quiz":
        return json.dumps({"error": "This function only supports quiz generation."})
    
    # Validate difficulty level
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"  # Default to medium if invalid
    
    # Set default question types if none provided
    if not question_types or len(question_types) == 0:
        question_types = ["mcq"]
    
    # Fallback to dummy implementation if no OpenAI API key or use_dummy set
    api_key = os.getenv("OPENAI_API_KEY")
    if use_dummy or not api_key:
        return generate_dummy_quiz(query, question_types, num_questions, difficulty)
    
    try:
        # Retrieve context with module filtering
        context = retrieve_course_context(course_id, query, module_id)
        if isinstance(context, dict) and "error" in context:
            return json.dumps(context)
        
        # Generate material using LLM with difficulty
        return generate_llm_quiz(query, question_types, num_questions, context, api_key, difficulty)
        
    except Exception as e:
        return json.dumps({"error": f"Error generating quiz: {str(e)}"})
    finally:
        # Audit logging
        responsible_id = teacher_id or "unknown"
        log_generation_event(responsible_id, course_id, material_type, query, module_id, difficulty)

def retrieve_course_context(course_id, query, module_id=None):
    """Helper function to retrieve context from course knowledge base.
    Optionally filter by specific module."""
    try:
        
        db_session = next(get_db())
        
        # If module_id is specified, get context only from that module
        if module_id:
            module = db_session.query(CourseModule).filter_by(id=module_id, course_id=course_id).first()
            if not module:
                return {"error": f"Module {module_id} not found in course {course_id}"}
            
            # Get text chunks from all submodules of this module
            chunks = db_session.query(ModuleTextChunk).join(CourseSubmodule).filter(
                CourseSubmodule.module_id == module_id
            ).all()
            
            if not chunks:
                return {"error": f"No content found in module '{module.title}'"}
            
            # Combine chunks for context
            context = "\n\n".join([chunk.chunk_text for chunk in chunks])
            
        else:
            # Use existing FAISS retrieval for entire course
            db = load_faiss_vectorstore(course_id, openai_api_key=None)  
            retriever = db.as_retriever(search_kwargs={"k": 10})
            retrieved_docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        if not context.strip():
            return {"error": "Insufficient context found for your query."}
        return context
        
    except Exception as e:
        return {"error": f"Problem loading course content: {str(e)}"}

def generate_dummy_quiz(query, question_types, num_questions, difficulty="medium"):
    """Generate dummy quiz data with difficulty levels for testing"""
    difficulty_modifiers = {
        "easy": {
            "title_suffix": " - Basics",
            "description": "Basic concepts and fundamental understanding",
            "sample_questions": {
                "mcq": "What is the basic definition of {topic}?",
                "msq": "Which of the following are simple characteristics of {topic}?",
                "short_answer": "Define {topic} in your own words.",
                "essay": "Explain the basic importance of {topic}."
            }
        },
        "medium": {
            "title_suffix": " - Application",
            "description": "Application and analysis of concepts",
            "sample_questions": {
                "mcq": "How would you apply {topic} in a practical scenario?",
                "msq": "Which methods can be used to analyze {topic}?",
                "short_answer": "Explain how {topic} relates to other concepts.",
                "essay": "Analyze the role of {topic} in its broader context."
            }
        },
        "hard": {
            "title_suffix": " - Advanced",
            "description": "Critical thinking and synthesis",
            "sample_questions": {
                "mcq": "What are the complex implications of {topic}?",
                "msq": "Which advanced concepts are interconnected with {topic}?",
                "short_answer": "Critically evaluate the significance of {topic}.",
                "essay": "Synthesize and evaluate the comprehensive impact of {topic}."
            }
        }
    }
    
    modifier = difficulty_modifiers.get(difficulty, difficulty_modifiers["medium"])
    
    dummy_quiz = {
        "title": f"Quiz on {query}{modifier['title_suffix']}",
        "description": f"Test your knowledge on {query}. {modifier['description']}",
        "difficulty": difficulty,
        "questions": []
    }
    
    # Generate sample questions based on requested types
    for i in range(min(num_questions, len(question_types) * 2)):
        q_type = question_types[i % len(question_types)]
        question_template = modifier["sample_questions"].get(q_type, "Question about {topic}")
        
        if q_type == "mcq":
            dummy_quiz["questions"].append({
                "type": "mcq",
                "question": question_template.format(topic=query),
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer_index": 0,
                "explanation": f"This is the correct answer for {difficulty} level understanding."
            })
        elif q_type == "msq":
            dummy_quiz["questions"].append({
                "type": "msq",
                "question": question_template.format(topic=query),
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer_indices": [0, 2],
                "explanation": f"These are the correct answers for {difficulty} level analysis."
            })
        elif q_type == "short_answer":
            dummy_quiz["questions"].append({
                "type": "short_answer",
                "question": question_template.format(topic=query),
                "model_answer": f"A {difficulty} level answer about {query}."
            })
        elif q_type == "essay":
            dummy_quiz["questions"].append({
                "type": "essay",
                "question": question_template.format(topic=query),
                "model_answer": f"A comprehensive {difficulty} level essay response about {query}."
            })
    
    return json.dumps(dummy_quiz, indent=2)

def generate_llm_quiz(query, question_types, num_questions, context, api_key, difficulty="medium"):
    """Generate a quiz using LLM with difficulty level"""
    try:
        chat = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.2,
            openai_api_key=api_key
        )
        
        # Difficulty-specific instructions
        difficulty_instructions = {
            "easy": "Focus on basic recall, simple definitions, and straightforward concepts. Questions should test fundamental understanding.",
            "medium": "Include application and analysis questions. Test understanding of concepts and ability to apply knowledge in familiar contexts.",
            "hard": "Emphasize synthesis, evaluation, and critical thinking. Include complex scenarios, comparative analysis, and higher-order thinking skills."
        }
        
        q_types_str = ", ".join(question_types)
        difficulty_instruction = difficulty_instructions.get(difficulty, difficulty_instructions["medium"])
        
        # Enhanced instructions to focus entirely on module content when available
        instructions = (
            f"Generate a {difficulty} difficulty quiz about '{query}' with {num_questions} questions using these types: {q_types_str}.\n\n"
            f"Difficulty Level - {difficulty.title()}: {difficulty_instruction}\n\n"
            "IMPORTANT: Base ALL questions strictly on the provided context from course materials. "
            "Do not include general knowledge questions outside of the provided content. "
            "If the context is from a specific module, ensure all questions test knowledge from that module's content.\n\n"
            "Your response must be a valid JSON object with exactly this structure:\n"
            "{\n"
            '  "title": "Quiz title here",\n'
            '  "description": "Brief description of the quiz including difficulty level",\n'
            '  "difficulty": "' + difficulty + '",\n'
            '  "questions": [\n'
            '    {"type": "mcq", "question": "Question text", "options": ["Option A", "Option B", "Option C", "Option D"], "correct_answer_index": 0, "explanation": "Why this is correct"},\n'
            '    {"type": "msq", "question": "Question text", "options": ["Option A", "Option B", "Option C", "Option D"], "correct_answer_indices": [0, 2], "explanation": "Why these are correct"}\n'
            '  ]\n'
            "}\n\n"
            "For 'essay' or 'short_answer' types, use this format in the questions array:\n"
            '{"type": "essay", "question": "Question text", "model_answer": "Model answer text"}\n'
            '{"type": "short_answer", "question": "Question text", "model_answer": "Brief model answer"}\n\n'
            "Include appropriate explanations for all question types."
        )
        
        full_prompt = f"{instructions}\nContext from course materials: {context}\n"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates educational quizzes based strictly on provided course materials."},
            {"role": "user", "content": full_prompt}
        ]
        
        response = chat.invoke(messages)
        
        # Extract the content from the response
        result = response.content.strip()
        
        if not result:
            return json.dumps({
                "error": "No response from LLM. Please retry with a different or more specific query."
            })
        
        # Process and validate the LLM response
        return process_llm_quiz_response(result, query, difficulty)
        
    except TimeoutError:
        return json.dumps({"error": "OpenAI response timed out. Retry with simpler or shorter query."})
    except Exception as e:
        return json.dumps({"error": f"Error in LLM quiz generation: {str(e)}"})

def process_llm_quiz_response(response, query, difficulty="medium"):
    """Process and validate the LLM response for a quiz with difficulty"""
    # Extract JSON from the response
    parsed_json = extract_json_from_text(response)
    
    if parsed_json is None:
        # Fallback to a basic structure if JSON parsing fails
        return json.dumps({
            "title": f"Quiz on {query}",
            "description": f"Test your knowledge on {query} ({difficulty} difficulty)",
            "difficulty": difficulty,
            "questions": [],
            "error": "Failed to parse LLM response into valid JSON."
        })
    
    # Ensure the quiz has the required structure with difficulty
    quiz_json = ensure_quiz_structure(parsed_json, query, difficulty)
    return json.dumps(quiz_json, indent=2)

def extract_json_from_text(text):
    """Extract JSON from text that might contain additional content"""
    # Try to find JSON in the response
    print("extracting json from text!")
    json_patterns = [
        ('{', '}'),  # For object
        ('[', ']')   # For array
    ]
    
    for start_char, end_char in json_patterns:
        start_index = text.find(start_char)
        if start_index == -1:
            continue
        
        # Find the matching closing bracket/brace
        bracket_count = 0
        for i in range(start_index, len(text)):
            if text[i] == start_char:
                bracket_count += 1
            elif text[i] == end_char:
                bracket_count -= 1
                if bracket_count == 0:
                    end_index = i + 1
                    try:
                        return json.loads(text[start_index:end_index])
                    except json.JSONDecodeError:
                        pass
    
    # More aggressive approach - try to find the longest valid JSON substring
    for i in range(len(text)):
        if text[i] in ['{', '[']:
            for j in range(len(text), i, -1):
                try:
                    return json.loads(text[i:j])
                except json.JSONDecodeError:
                    continue
    
    return None

def ensure_quiz_structure(parsed_json, query, difficulty="medium"):
    """Ensure the quiz JSON has the required structure with difficulty"""
    # Set default title and description if missing
    if "title" not in parsed_json or not parsed_json["title"]:
        parsed_json["title"] = f"Quiz on {query}"
    
    if "description" not in parsed_json or not parsed_json["description"]:
        parsed_json["description"] = f"Test your knowledge on {query} ({difficulty} difficulty)"
    
    # Ensure difficulty is included
    parsed_json["difficulty"] = difficulty
    
    # Ensure questions array exists
    if "questions" not in parsed_json:
        parsed_json["questions"] = []
    
    # Validate question structure
    validated_questions = []
    for question in parsed_json["questions"]:
        if isinstance(question, dict) and "type" in question and "question" in question:
            # Ensure proper structure based on question type
            if question["type"] in ["mcq", "msq"]:
                if "options" in question and "correct_answer_index" in question or "correct_answer_indices" in question:
                    validated_questions.append(question)
            elif question["type"] in ["essay", "short_answer"]:
                if "model_answer" in question:
                    validated_questions.append(question)
    
    parsed_json["questions"] = validated_questions
    return parsed_json

def log_generation_event(teacher_id, course_id, material_type, query, module_id=None, difficulty=None):
    """Log quiz generation events with module and difficulty information"""
    try:
        log_data = {
            "teacher_id": teacher_id,
            "course_id": course_id,
            "material_type": material_type,
            "query": query,
            "module_id": module_id,
            "difficulty": difficulty,
            "timestamp": datetime.utcnow()
        }
        print(f"Quiz generation logged: {log_data}")
    except Exception as e:
        print(f"Logging error: {str(e)}")

def generate_quiz_export(materials_json, course_id,format="pdf", include_answers=False):
    """
    Generate a PDF export of a quiz
    
    Args:
        materials_json (str): JSON string containing quiz data
        format (str): Format to export (currently only supports 'pdf')
        include_answers (bool): Whether to include answers in the export
        
    Returns:
        str: Path to the generated file
    """
    # Parse the provided materials JSON
    quiz_data = json.loads(materials_json)
    file_id = uuid.uuid4()
    filename = f"quiz_{file_id}.{format}"
    export_dir = "static/exports"
    
    # Make sure the export directory exists
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, filename)
    
    if format == "pdf":
        # Import ReportLab modules
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
        from reportlab.lib.units import inch
        
        # Create a PDF document
        doc = SimpleDocTemplate(
            file_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#1a56db'),
            spaceAfter=12
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#1e429f'),
            spaceAfter=8
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            spaceAfter=6
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#2d3748'),
            fontName='Helvetica-Bold',
            spaceAfter=8
        )
        
        option_style = ParagraphStyle(
            'Option',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            leftIndent=20,
            spaceAfter=6
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            leftIndent=20,
            textColor=colors.HexColor('#047857'),
            spaceAfter=6
        )
        
        explanation_style = ParagraphStyle(
            'Explanation',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            leftIndent=20,
            textColor=colors.HexColor('#1e429f'),
            spaceAfter=12,
            borderWidth=1,
            borderColor=colors.HexColor('#e5e7eb'),
            borderPadding=8,
            borderRadius=4
        )
        
        # Build the PDF content
        elements = []
        
        # Add title and description
        elements.append(Paragraph(quiz_data.get("title", "Quiz"), title_style))
        elements.append(Paragraph(quiz_data.get("description", ""), normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add questions
        for i, question in enumerate(quiz_data.get("questions", [])):
            # Question text
            question_text = f"Question {i+1}: {question.get('question', '')}"
            elements.append(Paragraph(question_text, question_style))
            
            # Options for MCQ/MSQ
            question_type = question.get("type", "mcq")
            
            if question_type in ["mcq", "msq"]:
                options = question.get("options", [])
                
                for j, option in enumerate(options):
                    option_text = f"{chr(65+j)}. {option}"
                    
                    # For answer key, highlight correct options
                    if include_answers:
                        if question_type == "mcq" and j == question.get("correct_answer_index", 0):
                            elements.append(Paragraph(f"<b>{option_text} ✓</b>", answer_style))
                        elif question_type == "msq" and j in question.get("correct_answer_indices", []):
                            elements.append(Paragraph(f"<b>{option_text} ✓</b>", answer_style))
                        else:
                            elements.append(Paragraph(option_text, option_style))
                    else:
                        elements.append(Paragraph(option_text, option_style))
                
                # Add explanation if include_answers is True
                if include_answers and "explanation" in question:
                    elements.append(Paragraph(f"<i>Explanation:</i> {question['explanation']}", explanation_style))
            
            elif question_type in ["essay", "short_answer"]:
                # For essay or short answer questions
                elements.append(Paragraph("Answer space: ______________________________", option_style))
                
                # Add model answer if include_answers is True
                if include_answers and "model_answer" in question:
                    elements.append(Paragraph(f"<i>Model Answer:</i> {question['model_answer']}", explanation_style))
            
            elements.append(Spacer(1, 0.25*inch))
        
        # Generate the PDF document
        doc.build(elements)
    else:
        # If not PDF format, just save as JSON for now
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(quiz_data, indent=2))
    s3_key = f"quiz_exports/{course_id}/{filename}"
    upload_file_to_s3_from_path(file_path, s3_key)
    return s3_key, filename


def check_quiz_quota(db, teacher_id, course_id, max_quota=5):
    """
    Check if a teacher has exceeded their quiz quota for a course today
    
    Returns:
        tuple: (quota_exceeded: bool, remaining: int)
    """
    today = date.today()
    
    # Find or create today's quota record
    quota = db.query(QuizQuota).filter(
        QuizQuota.teacher_id == teacher_id,
        QuizQuota.course_id == course_id,
        QuizQuota.date == today
    ).first()
    
    if not quota:
        quota = QuizQuota(
            teacher_id=teacher_id,
            course_id=course_id,
            date=today,
            count=0
        )
        db.add(quota)
        db.commit()
    
    # Check if quota exceeded
    quota_exceeded = quota.count >= max_quota
    remaining = max(max_quota - quota.count, 0)
    
    return quota_exceeded, remaining

def increment_quiz_quota(db, teacher_id, course_id):
    """
    Increment the quiz count for today
    
    Returns:
        int: New count
    """
    today = date.today()
    
    # Find or create today's quota record
    quota = db.query(QuizQuota).filter(
        QuizQuota.teacher_id == teacher_id,
        QuizQuota.course_id == course_id,
        QuizQuota.date == today
    ).first()
    
    if not quota:
        quota = QuizQuota(
            teacher_id=teacher_id,
            course_id=course_id,
            date=today,
            count=1  # Start at 1 since we're creating a quiz now
        )
        db.add(quota)
    else:
        quota.count += 1
    
    db.commit()
    return quota.count