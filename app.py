from flask import Flask, render_template, request, session, Response, stream_with_context
import uuid
from rag_pipeline import run_rag_pipeline
app = Flask(__name__)
# Secret key for session management
app.secret_key = "thread123"

@app.route("/")
def index():
    # Initialize a new thread_id when a user starts a new session
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())
    
    # Render the main chatbot interface
    return render_template("chatbot.html")

# Keep the original non-streaming route for backward compatibility
@app.route("/get", methods=["POST"])
def chatbot():
    # The incoming user message from the form data
    user_msg = request.form["msg"]

    # To ensure there's a thread_id in the session
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())
    
    # Get the thread_id from the session
    thread_id = session["thread_id"]

    # Run the RAG pipeline
    result = run_rag_pipeline(user_msg, thread_id)

    # The pipeline returns a dictionary with:
    # 1) "ai_response": the final text from the LLM
    # 2) "retrieval_used": a boolean indicating if a retrieval tool was called
    ai_answer = result["ai_response"]
    retrieval_used = result["retrieval_used"]

    # To indicate that retrieval was used
    if retrieval_used:
        final_response = f"{ai_answer}\n\n<small><i>Sources were consulted for this response.</i></small>"
    else:
        final_response = ai_answer

    return final_response

@app.route("/reset", methods=["POST"])
def reset_conversation():
    """Optional endpoint to allow users to reset their conversation history"""
    # Generate a new thread_id to start fresh
    session["thread_id"] = str(uuid.uuid4())
    return "Conversation reset successfully."

if __name__ == "__main__":
    # Run the Flask development server in debug mode for testing
    app.run(debug=True)
