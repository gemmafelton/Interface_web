from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialization of the FastAPI application
app = FastAPI()

# Load the model
model_path = "/Users/tannina/Desktop/API/rnn_model/rnn_model.json"
weights_path = "/Users/tannina/Desktop/API/rnn_model/rnn_model_weights.h5"
loaded_model = tf.keras.models.model_from_json(open(model_path, "r").read())
loaded_model.load_weights(weights_path)

# Load the tokenizer configuration
tokenizer_config_path = "/Users/tannina/Desktop/API/rnn_model/tokenizer_config.json"
with open(tokenizer_config_path, 'r') as tokenizer_config_file:
    tokenizer_config = json.load(tokenizer_config_file)

# Load the tokenizer vocabulary
tokenizer_vocab_path = "/Users/tannina/Desktop/API/rnn_model/tokenizer_vocab.json"
with open(tokenizer_vocab_path, 'r') as tokenizer_vocab_file:
    tokenizer_vocab = json.load(tokenizer_vocab_file)

# Reconstruct the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.__dict__.update(tokenizer_config)
tokenizer.word_index = tokenizer_vocab

# Define max_sequence_length (replace with the actual value used during training)
max_sequence_length = 40

# HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DetectTweets</title>
    <link rel="stylesheet" type="text/css" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <div class="app-name">DetectTweets</div>

        <!-- Section "Test it out" -->
        <div class="tweet-section">
            <!-- Section d'essai -->
            <div class="trial-section">
                <div class="test-details">
                    <div>Test it out</div>
                </div>
            </div>
        </div>

        <!-- Tweet input box -->
        <form action="/analyze_tweet/" method="post">
            <div class="tweet">
                <textarea name="tweet" id="tweetInput" placeholder="What's happening?" rows="4" style="width: 100%;"></textarea>
                <button type="button" class="tweet-button" id="tweetButton">
                    Test
                </button>
            </div>
        </form>

        <!-- Result box -->
        <div class="result-box" id="resultBox" style="display: none;">
            <div class="result-text" id="resultText"></div>
        </div>

        <!-- History list -->
        <ul id="historyList"></ul>

        <!-- Navigation bar -->
        <div class="navbar">
            <div class="nav-item" onclick="toggleInfo()">About</div>
            <div class="nav-item" onclick="toggleInstructions()">Instructions</div>
            <div class="nav-item" onclick="toggleContact()">Contact</div>
        </div>

        <!-- Instructions section -->
        <div class="nav-info" id="navInstructions" style="display: none;">
            <h2>Instructions</h2>
            <ol>
                <li>We have developed an application to determine if tweets contain hateful content, subsequently classifying them into different categories such as "offensive," "hateful," or "neutral." This application is based on a recurrent neural network (RNN) model.</li>
                <li>Please enter the tweet you would like to analyze and observe how it is classified into the different categories</li>
            </ol>
        </div>

        <!-- Contact section -->
        <div class="nav-info" id="navContact">
            <h2>Contact</h2>
            <p>Creators :</p>
            <ul>
                <li>Felton Gemma : 39000001@parisnanterre.fr</li>
                <li>Hamizi Tannina : 39002519@parisnanterre.fr</li>
            </ul>
            <!-- GitHub link -->
            <a href="https://github.com/gemmafelton/Interface_web.git" target="_blank" class="github-link">GitHub</a>
        </div>



        <!-- Additional Information in the Navigation Bar -->
        <div class="nav-info" id="navInfo">
            <h2>Additional Information</h2>
            <p>This application was developed as part of two courses led by Loïc Grobol:</p>
            <ul>
                <li>Neural Networks</li>
                <li>Web Interface</li>
            </ul>
            <p>It focuses on detecting tweets in English using a specifically trained artificial intelligence model.</p>
        </div>

        <script src="/static/api.js"></script>
    </div>
</body>
</html>
"""

# Route to serve the HTML content
@app.get("/")
async def read_index():
    return HTMLResponse(content=html_content, status_code=200)

# Route to analyze a tweet
@app.post("/analyze_tweet/")
async def analyze_tweet_endpoint(tweet: str = Form(...)):
    # Perform tweet analysis here
    input_sequence = tokenizer.texts_to_sequences([tweet])
    input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')

    output = loaded_model.predict(input_padded)

    probabilities_list = output
    class_labels = {0: 'hate speech', 1: 'offensive language', 2: 'neither'}

    result_text = ""
    result_percentage = ""

    for i, prob in enumerate(probabilities_list[0]):
        result_text += f"Probability that this tweet is {class_labels[i]}: {prob * 100:.2f}%<br>"
        result_percentage += f"{class_labels[i]}: {prob * 100:.2f}%<br>"

    result_text = result_text.rstrip("<br>")
    result_percentage = result_percentage.rstrip("<br>")

    response_data = {
        "result_text": result_text,
        "result_percentage": result_percentage,
        "label": class_labels[np.argmax(probabilities_list[0])]
    }

    return HTMLResponse(content=json.dumps(response_data), status_code=200, media_type="application/json", headers={"Access-Control-Allow-Origin": "*"})


# Route to serve static files
@app.get("/static/{filename}")
async def read_static_file(filename: str):
    try:
        return FileResponse(f"static/{filename}")
    except FileNotFoundError:
        return {"error": "File not found"}, 404

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
