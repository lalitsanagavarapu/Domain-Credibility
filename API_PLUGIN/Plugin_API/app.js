var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');
var scoreRouter = require('./routes/getScore');
var searchRouter= require('./routes/getResult')
// console.log("\n\n-----------scoreRouter------------\n\n");
// console.log(scoreRouter);
var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));
app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*"); // update to match the domain you will make the request from
  //res.header("Acces-Contol-Allow-Origin","https://127.0.0.1:3000");//dangerous
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});
app.use('/', indexRouter);
app.use('/users', usersRouter);
app.use('/getScore', scoreRouter);
app.use('/getResult',searchRouter);
// app.get('/getScore', scoreRouter);
// catch 404 and forward to error handler
app.use(function (req, res, next) {
  next(createError(404));
});


  
function temp(req, res) { 
    var spawn = require("child_process").spawn;
    var finalScore;
    var process = spawn('python3',["./public/script/getScore.py", 
                            req.query.url] );
    process.stdout.on('data', function(data) {
      finalScore = data.toString();
      console.log(finalScore);
      res.render('getScore', { score: finalScore});
    });
    
} 



// error handler
app.use(function (err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
