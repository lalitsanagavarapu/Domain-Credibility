var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function (req, res, next) {
    var spawn = require("child_process").spawn;
    var finalScore;
    // console.log("\n\n\n-------querystring----------\n");
    // console.log(req.query);
    // console.log("\n\n\n");
    var process = spawn('python3', ["./public/script/getScore.py", req.query.url]);
    process.stdout.on('data', function (data) {
        finalScore = data.toString();
        res.send(finalScore.toString());
    });
});

module.exports = router;
