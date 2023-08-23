function sendMessage() {
    var userInput = document.getElementById("userInput").value;
    document.getElementById("chatbox").innerHTML += "<p>You: " + userInput + "</p>";
    document.getElementById("userInput").value = "";

    $.ajax({
        type: "POST",
        url: "/chat",
        contentType: "application/json",
        data: JSON.stringify({"message": userInput}),
        success: function(response) {
            document.getElementById("chatbox").innerHTML += "<p>Bot: " + response.response + "</p>";
        },
        error: function() {
            document.getElementById("chatbox").innerHTML += "<p>Bot: Sorry, something went wrong.</p>";
        }
    });
}