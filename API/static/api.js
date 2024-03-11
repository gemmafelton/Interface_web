async function detectTweet(tweet) {
    const response = await fetch('http://127.0.0.1:8000/analyze_tweet/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
            'tweet': tweet
        })
    });
    const data = await response.json();
    displayResult(data);
    updateHistory(tweet, data.label);
}

function displayResult(data) {
    const resultBox = document.getElementById('resultBox');
    const resultText = document.getElementById('resultText');

    // Extract the first three lines from the result_text
    const firstThreeLines = data.result_text.split('\n').slice(0, 3).join('<br>');

    // Update the innerHTML with the extracted lines
    resultText.innerHTML = firstThreeLines;

    resultBox.style.display = 'block';
}

function updateHistory(tweet, label) {
    const historyList = document.getElementById('historyList');
    const newItem = document.createElement('li');
    newItem.innerText = `${tweet} - ${label}`;
    historyList.appendChild(newItem);
}

document.getElementById('tweetButton').addEventListener('click', function() {
    const tweetInput = document.getElementById('tweetInput').value;
    detectTweet(tweetInput);
});

document.addEventListener('DOMContentLoaded', function() {
    const navInfo = document.getElementById('navInfo');
    const navContact = document.getElementById('navContact');

    document.addEventListener('click', function(event) {
        if (event.target !== navInfo && event.target !== navContact && event.target.closest('.nav-item') === null) {
            if (navInfo.style.display === 'block') {
                navInfo.style.display = 'none';
            }
            if (navContact.style.display === 'block') {
                navContact.style.display = 'none';
            }
        }
    });

    document.getElementById('tweetButton').addEventListener('click', function() {
        const tweetInput = document.getElementById('tweetInput').value;
        detectTweet(tweetInput);
    });
});

function toggleInfo() {
    const navInfo = document.getElementById('navInfo');
    const navContact = document.getElementById('navContact');
    if (navInfo.style.display === 'none' || navInfo.style.display === '') {
        navInfo.style.display = 'block';
        navContact.style.display = 'none';
    } else {
        navInfo.style.display = 'none';
    }
}

function toggleContact() {
    const navInfo = document.getElementById('navInfo');
    const navContact = document.getElementById('navContact');
    if (navContact.style.display === 'none' || navContact.style.display === '') {
        navContact.style.display = 'block';
        navInfo.style.display = 'none';
    } else {
        navContact.style.display = 'none';
    }
}
