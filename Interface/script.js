async function detectTweet(tweet) {
    const response = await fetch('/detect_tweet/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: tweet
        })
    });
    const data = await response.json();
    displayResult(data);
}

function displayResult(data) {
    const resultBox = document.getElementById('resultBox');
    const resultText = document.getElementById('resultText');
    resultText.innerText = `${data.text} is labeled as ${data.label} with a confidence of ${data.confidence}`;
    resultBox.style.display = 'block';
}

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

document.addEventListener('click', function(event) {
    const navInfo = document.getElementById('navInfo');
    const navContact = document.getElementById('navContact');

    if (event.target !== navInfo && event.target !== navContact && event.target.closest('.nav-item') === null) {
        if (navInfo.style.display === 'block') {
            navInfo.style.display = 'none';
        }
        if (navContact.style.display === 'block') {
            navContact.style.display = 'none';
        }
    }
});

