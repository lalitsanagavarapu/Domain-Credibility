document.addEventListener("DOMContentLoaded", function (event) {
    var but = document.getElementById("getScoreButton");
    but.onclick = getScoreForURL;
});

function getResult() {
    var inputVal = document.getElementById("searchInput").value;
    // alert(inputVal);
    window.open("http://127.0.0.1:3000/getResult?keyword="+(inputVal),"_blank"); 
}
function getScoreForURL() {
    var curURL = window.location.href;
    curURL = curURL.toString();
    curURL += "::--::--::";
    console.log(curURL);
    var req = new XMLHttpRequest();
    req.open(
        "GET",
        "http://127.0.0.1:3000/getScore/?url="+curURL,
        true);
    req.onload = function () {
        var reply = req.responseText;
        reply = reply.split("::--::--::")
        var div = document.getElementById("myScore");
        div.innerHTML = reply[0];
        div.style.display = 'block';
    };
    req.send(null);
}
