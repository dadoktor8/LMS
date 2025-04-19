from typing import List, Dict, Union, Optional
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


def generate_study_material(query:str, material_type:str, course_id:int, student_id:str) -> str:
    # --- Step 1: Validate inputs ---
    if not query or not query.strip():
        return json.dumps({"error": "Query cannot be empty."})
    if not isinstance(course_id, int) or course_id <= 0:
        return json.dumps({"error": "Course ID must be a positive integer."})
    if not student_id or not isinstance(student_id, str):
        return json.dumps({"error": "Student ID must be a non-empty string."})
    if material_type not in {"flashcards", "quiz", "study_guide"}:
        return json.dumps({"error": "Invalid material type."})

    # --- Step 2: Load FAISS index and fetch context ---
    try:
        index_path = f"faiss_index_{course_id}"
        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 10})
    except FileNotFoundError:
        return json.dumps({"error": f"No existing knowledge base found for course {course_id}"})
    except Exception as e:
        return json.dumps({"error": f"Problem loading FAISS index: {str(e)}"})

    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    if not context.strip():
        return json.dumps({"error":"Insufficient context found for your query."})

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
        prompt_template = get_prompt_for_material_type(material_type)
        if not prompt_template:
            return json.dumps({"error": "Invalid or missing prompt template."})
    except Exception as e:
        return json.dumps({"error": f"Error fetching Prompt: {str(e)}"})

    # --- Step 5: Generate Study Material ---
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template, "verbose": False},
            chain_type="stuff"
        )

        result = qa_chain({"query": query})
        response = result.get("result", "").strip()

        if not response:
            return json.dumps({"error": "No response from LLM. Please retry with a different or more specific query."})

    except TimeoutError:
        return json.dumps({"error": "OpenAI response timed out. Retry with simpler or shorter query."})
    except Exception as e:
        return json.dumps({"error": f"LLM generation failed: {str(e)}"})

    # --- Step 6: Robust JSON Parsing ---
    try:
        # Extract JSON based on type (flashcards are [], others {})
        start_char, end_char = ("[", "]") if material_type=="flashcards" else ("{","}")
        json_start = response.find(start_char)
        json_end = response.rfind(end_char) + 1

        if json_start==-1 or json_end==-1:
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

def get_prompt_for_material_type(material_type: str) -> PromptTemplate:
    """Returns the appropriate prompt template based on material type"""
    
    if material_type == "flashcards":
        return PromptTemplate(
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
        return PromptTemplate(
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
        return PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are Lumi, a warm, caring AI tutor creating a study guide for students using provided course materials.

            Your task is to:
            - Extract only relevant points from the context.
            - Create a *concise*, *structured*, and *easy-to-read* JSON-based study guide.
            - Focus on clarity and educational value.

            💡 Format your response strictly as a JSON object like this:

            {{
            "title": "Study Guide Title",
            "sections": [
                {{
                "heading": "Section Heading",
                "points": [
                    "Concise key point 1",
                    "Concise key point 2",
                    "Use examples if relevant",
                    "Avoid long paragraphs"
                ]
                }},
                ...more sections...
            ],
            "summary": "1-2 sentence recap of the most important takeaways"
            }}

            Only use relevant information from the provided course materials. Do not make up facts.

            📘 Course Materials:
            {context}

            📍 Topic: {question}

            Now, generate the study guide JSON:
            """
                )


def log_generation_event(student_id: str, course_id: int, material_type: str, query: str):
    """Log the generation of study materials"""
    try:
        # You can implement logging to a database or file here
        print(f"Generated {material_type} for student {student_id} in course {course_id} on topic '{query}'")
    except Exception as e:
        print(f"Error logging generation event: {e}")


def render_flashcards_htmx(materials_json):
    """Render flashcards in HTMX format"""
    try:
        materials = json.loads(materials_json)
        html = '<div class="flashcards-container">'
        
        for i, card in enumerate(materials):
            html += f'''
            <div class="flashcard" id="card-{i}">
                <div class="flashcard-inner">
                    <div class="flashcard-front">
                        <h3>{card["question"]}</h3>
                        <div class="flashcard-footer">
                            <button onclick="flipCard({i})">Show Answer</button>
                        </div>
                    </div>
                    <div class="flashcard-back">
                        <div class="answer">{card["answer"]}</div>
                        <div class="tags">
                            {" ".join([f'<span class="tag">{tag}</span>' for tag in card["tags"]])}
                        </div>
                        <div class="flashcard-footer">
                            <button onclick="flipCard({i})">Show Question</button>
                        </div>
                    </div>
                </div>
            </div>
            '''
        
        html += '''
        </div>
        <script>
        function flipCard(index) {
            const card = document.getElementById(`card-${index}`);
            card.querySelector('.flashcard-inner').classList.toggle('flipped');
        }
        </script>
        '''
        return html
    except Exception as e:
        return f"<div>Error rendering flashcards: {str(e)}</div>"

def render_quiz_htmx(materials_json):
    """Render quiz in HTMX format"""
    try:
        materials = json.loads(materials_json)
        html = f'''
        <div class="quiz-container">
            <h2>{materials["title"]}</h2>
            <p class="quiz-description">{materials["description"]}</p>
            
            <form id="quiz-form">
        '''
        
        for i, question in enumerate(materials["questions"]):
            html += f'''
            <div class="quiz-question" id="question-{i}">
                <h3>Question {i+1}: {question["question"]}</h3>
                <div class="options">
            '''
            
            for j, option in enumerate(question["options"]):
                html += f'''
                <div class="option">
                    <input type="radio" id="q{i}-o{j}" name="q{i}" value="{j}">
                    <label for="q{i}-o{j}">{option}</label>
                </div>
                '''
            
            html += f'''
                </div>
                <div class="explanation hidden" id="explanation-{i}"></div>
            </div>
            '''
        
        html += f"""
    <div class="quiz-controls">
        <button type="button" onclick="checkAnswers()">Check Answers</button>
    </div>
    </form>
</div>
<script>
function checkAnswers() {{
    const quizData = JSON.parse(`{materials_json.replace('`', '\\`')}`);
    let score = 0;

    quizData.questions.forEach((question, i) => {{
        const selectedOption = document.querySelector(`input[name="q${{i}}"]:checked`);
        const explanationDiv = document.getElementById(`explanation-${{i}}`);

        if (!selectedOption) {{
            explanationDiv.innerHTML = "Please select an answer";
            explanationDiv.className = "explanation warning";
            return;
        }}

        const selectedIndex = parseInt(selectedOption.value);

        if (selectedIndex === question.correct_answer_index) {{
            score++;
            explanationDiv.innerHTML = "Correct! " + question.explanation;
            explanationDiv.className = "explanation correct";
        }} else {{
            explanationDiv.innerHTML = "Incorrect. " + question.explanation;
            explanationDiv.className = "explanation incorrect";
        }}

        explanationDiv.classList.remove("hidden");
    }});

    const scoreDiv = document.createElement("div");
    scoreDiv.className = "quiz-score";
    scoreDiv.innerHTML = `Your score: ${{score}}/${{quizData.questions.length}}`;

    const quizControls = document.querySelector(".quiz-controls");
    quizControls.appendChild(scoreDiv);
}}
</script>
"""

        
        return html
    except Exception as e:
        return f"<div>Error rendering quiz: {str(e)}</div>"

def render_study_guide_htmx(materials_json):
    """Render study guide in HTMX format using new point-based structure"""
    try:
        materials = json.loads(materials_json)
        html = f'''
        <div class="study-guide-container">
            <h2>{materials["title"]}</h2>
            
            <div class="toc">
                <h3>📘 Table of Contents</h3>
                <ul>
        '''

        # Table of Contents
        for i, section in enumerate(materials["sections"]):
            html += f'<li><a href="#section-{i}">{section["heading"]}</a></li>'

        html += '''
                </ul>
            </div>
            
            <div class="study-guide-content">
        '''

        # Render Sections
        for i, section in enumerate(materials["sections"]):
            html += f'''
            <div class="study-guide-section" id="section-{i}">
                <h3>{section["heading"]}</h3>
                <ul class="section-points">
            '''
            for point in section["points"]:
                html += f'<li>{point}</li>'
            html += '''
                </ul>
            </div>
            '''

        # Summary Section
        html += f'''
            <div class="study-guide-summary">
                <h3>📝 Summary</h3>
                <p>{materials["summary"]}</p>
            </div>
        </div>

        <div class="study-guide-controls" style="text-align: center; margin-top: 1.5rem;">
            <button onclick="window.print()" class="btn">🖨️ Print Study Guide</button>
        </div>
        '''

        return html

    except Exception as e:
        return f"<div>Error rendering study guide: {str(e)}</div>"

