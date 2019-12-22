function changeResults() {
    errorIndices = []
    elements = document.getElementsByClassName("r");
    console.log(elements.length);
    var urlArray = []
    for (var i = 0; i < elements.length; i++) {
        if (elements[i].querySelector("a")!=null) {
            var curURL = elements[i].querySelector("a").href.toString();
            // console.log("here "+i.toString()+" "+curURL);
            urlArray.push(curURL);
        } else {
            console.log("ERROR OCCURRED FOR "+ elements[i].toString);
            errorIndices.push(i);
        }
    }
    // console.log(errorIndices);
    // console.log(urlArray);
    var urlString = urlArray.join("::--::--::");
    var responseString;
    
    let url = new URL('http://127.0.0.1:3000/getScore/');
    url.searchParams.set('url', urlString);
    
    var xhr = new XMLHttpRequest();
    xhr.open("GET", url, true);
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4) {
            console.log("\n\n-------Response received from server-------------\n" + xhr.response + "\n\n");
            responseString = xhr.responseText.toString();
            responseString = responseString.split("::--::--::");
            console.log(responseString);
            for (var i = 0; i < elements.length; i++) {
                if (!errorIndices.includes(i)) {
                    var t = document.createElement("h3");
                    var text = document.createTextNode("WEBCRED score:\t " + responseString[i]);
                    t.style.font = "italic bold 20px arial,serif";
                    t.appendChild(text);
                    elements[i].appendChild(t);
                }
            }
        }
    }
xhr.send();
}
changeResults();