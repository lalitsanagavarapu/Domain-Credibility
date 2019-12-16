var express = require('express');
var router = express.Router();

router.get('/', function (req, res, next) {

    searchKey = req.query.keyword;
    console.log("here  " + searchKey);
    request = require('request-json');
    var client = request.createClient('https://www.googleapis.com');
    var data = {
        key: 'AIzaSyCXGQtPTfJrl8lwokUCVlL2fMPKhqm9WjA',
        content: 'my content',
        cx: '004686178285036570982:hqvgyyg47ky',
        q: searchKey
    };
    client.get('/customsearch/v1',data, function (err, res, body) {
        console.log(err);
        console.log(res);
        console.log(body);
    });
    res.send(searchKey);
});


module.exports = router;
