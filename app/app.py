from flask import Flask, request, render_template
from utils import load_bart_model, load_flan_t5_lora, ensemble_summary

app = Flask(__name__)

# Load models
bart_tokenizer, bart_model = load_bart_model("harshvardhini123/fine_tuned_bart")
t5_tokenizer, t5_model = load_flan_t5_lora("t5-small", "harshvardhini123/fine_tuned_model")

@app.route("/")  # Front page route
def front_page():
    return render_template("index.html")

@app.route("/summarize", methods=["GET", "POST"]) 
def summarize():
    final_summary = ""
    input_text = ""

    if request.method == "POST":
        input_text = request.form["input_text"]
        perspective = request.form["perspective"]
        prompt = f"Summarize from the {perspective.upper()} perspective: {input_text}"

        final_summary = ensemble_summary(
            prompt, 
            bart_model, bart_tokenizer, 
            t5_model, t5_tokenizer
        )

    return render_template("summarizer.html", input_text=input_text, final_summary=final_summary)

@app.route("/help")
def help_page():
    return render_template("help.html")


if __name__ == "__main__":
    app.run(debug=True)
