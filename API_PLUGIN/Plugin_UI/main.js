document.addEventListener("DOMContentLoaded", function (event) {
    var but = document.getElementById("getScoreButton");
    but.onclick = getScoreForURL;
});

function getResult() {
    var inputVal = document.getElementById("searchInput").value;
    // alert(inputVal);
    //window.open("http://127.0.0.1:8000/getResult?keyword="+(inputVal),"_blank"); 
    window.open("http://127.0.0.1:8000") //webcred server 
}
function getScoreForURL() {
    var curURL = window.location.href;
    curURL = curURL.toString();
    curURL += "::--::--::";
    var req = new XMLHttpRequest();
         //no such url
         //output is printed as a json file and to the terminal
    req.open(
        "GET",
        "http://127.0.0.1:8000/getScore/?url="+curURL, 
        true);
    req.onload = function () {
        var reply = req.responseText;
        reply = reply.split("::--::--::")
        var div = document.getElementById("myScore");//no such element
        div.innerHTML = reply[0];
        div.style.display = 'block';
    };
    req.send(null);
}
