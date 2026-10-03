// script.js - sends the typed text to Flask and updates the page without reloading

const input = document.getElementById("text-input");
let timer = null;   // used for debounce

// Send text to the Flask /analyze route and show the result
function analyze() {
  fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: input.value })
  })
    .then(response => response.json())
    .then(data => {
      document.getElementById("words").textContent = data.words;
      document.getElementById("characters").textContent = data.characters;
      document.getElementById("sentences").textContent = data.sentences;

      const sentiment = document.getElementById("sentiment");
      sentiment.textContent = data.sentiment;
      sentiment.className = data.sentiment.toLowerCase();   // changes the colour
      document.getElementById("confidence").textContent = data.confidence + "%";
    });
}

// Debounce: wait 400 ms after the user stops typing, so Flask is not called on every key press
input.addEventListener("input", function () {
  clearTimeout(timer);
  timer = setTimeout(analyze, 400);
});
