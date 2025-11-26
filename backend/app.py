from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import subprocess
import sys
from dashscope import Application
from __init__ import modify_input

# Initialize Flask application and configure CORS
app = Flask(__name__)
CORS(
    app,
    resources=r"/*",
    origins="http://localhost:8080",
    methods=["GET", "POST", "OPTIONS"],
    allow_headers="*",
    supports_credentials=True,
    max_age=3600
)

# Handle OPTIONS preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return Response(status=200)

# Long-term memory interface
@app.route("/api/long-term-memory", methods=["POST"])
def save_long_term_memory():
    data = request.get_json(silent=True) or {}
    user_input = data.get("user_input", "").strip()
    
    if not user_input:
        return jsonify({"status": "error", "message": "User input cannot be empty", "ai_answer": ""}), 400
    
    result = subprocess.run(
        [sys.executable, "CreateLongTermMemories.py", "--input", user_input],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        return jsonify({
            "status": "success",
            "message": "Long-term memory saved successfully",
            "ai_answer": ""
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": result.stderr.strip() or "Save failed",
            "ai_answer": ""
        }), 500

# RAG core interface: only catch errors in cooking steps generation without affecting other tests
@app.route("/api/rag-generate", methods=["POST"])
def rag_generate():
    data = request.get_json()
    user_input = data.get("user_input", "").strip()
    context = data.get("context", "")
    ingredients = data.get("ingredients", [])
    forbidden = data.get("forbidden", [])
    unavailable_utensils = data.get("unavailable_utensils", [])
    
    if not user_input:
        return jsonify({"status": "error", "message": "User input cannot be empty", "data": {}}), 400
    
    # Asynchronously call long-term memory script (normal test)
    subprocess.Popen(
        [sys.executable, "CreateLongTermMemories.py", "--input", user_input],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 1. Input optimization (normal test)
    modified_input = modify_input(user_input)
    response_modify = Application.call(
        api_key="sk-55f91aaf5605438997475098330d3ab5",
        app_id='85e3ba9a2fe44a49ad469e0b640f9c58',
        prompt=modified_input
    )
    conscious_conclusion = response_modify.output.text  
    
    # 2. Recipe retrieval (normal test)
    retrieval_prompt = context if context else conscious_conclusion
    if ingredients:
        retrieval_prompt += f"\nRequired Ingredients: {', '.join(ingredients)}"
    if forbidden:
        retrieval_prompt += f"\nForbidden Ingredients: {', '.join(forbidden)}"
    if unavailable_utensils:
        retrieval_prompt += f"\nUnavailable Utensils: {', '.join(unavailable_utensils)}"
    
    response_retrival = Application.call(
        api_key="sk-55f91aaf5605438997475098330d3ab5",
        app_id='d21c0471a54b41fd98acbd06b5f98bf9',
        prompt=retrieval_prompt
    )
    recipes_retrival = response_retrival.output.text  
    
    # 3. Cooking steps generation: catch errors and skip with placeholder (without affecting other tests)
    try:
        response_helper = Application.call(
            api_key="sk-55f91aaf5605438997475098330d3ab5",
            app_id='33610b43e4d642e49402e49d46f0c2b9',
            prompt=recipes_retrival
        )
        cooking_steps = response_helper.output.text
    except AttributeError:
        # Skip current error and return with placeholder, without affecting other tests
        cooking_steps = "[Temporary Placeholder] Cooking steps generation is being tested. Please check AppID/API Key later."
    
    # Return results normally; frontend can test input optimization, recipe retrieval, typewriter effect, etc.
    return jsonify({
        "status": "success",
        "data": {
            "input_modified": conscious_conclusion,
            "recipes_list": recipes_retrival,
            "cooking_steps": cooking_steps,
            "context": cooking_steps
        }
    }), 200

# Compatible with original /search-qa interface
@app.route("/api/search-qa", methods=["POST"])
def search_qa_compatible():
    data = request.get_json()
    ingredients = data.get("ingredients", [])
    taste_preference = data.get("taste_preference", "")
    forbidden = data.get("forbidden", [])
    unavailable_utensils = data.get("unavailable_utensils", [])
    
    if taste_preference:
        user_input = f"I want to make {taste_preference} dishes with {', '.join(ingredients)}. Forbidden ingredients: {', '.join(forbidden)}. Unavailable utensils: {', '.join(unavailable_utensils)}"
    else:
        user_input = f"I want to make dishes with {', '.join(ingredients)}. Forbidden ingredients: {', '.join(forbidden)}. Unavailable utensils: {', '.join(unavailable_utensils)}"
    
    request.json["user_input"] = user_input
    return rag_generate()

# Start the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)