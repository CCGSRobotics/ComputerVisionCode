// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.

var fs = require("fs");
var $ = require("jquery");
var output;


function update() {
  hazmat = document.getElementById("check1");
  general = document.getElementById("check2");
  qr = document.getElementById("check3");
  point = document.getElementById("check4");

  var e = e || window.event;
  var pageX = e.pageX;
  var pageY = e.pageY;
  // IE 8
  var y = document.getElementById("iframe").scrollHeight;
  var x = document.getElementById("iframe").scrollWidth;

  var xnorm = pageX / x;
  var ynorm = pageY / y;

  console.log(point.checked);
  var data = hazmat.checked + " " + general.checked + " " + qr.checked + " " + xnorm + " " + ynorm + " " + point.checked;


  fs.writeFile("hazmat/output.txt", data, (err) => {
    if (err) console.log(err);
  });
}

fs.readFile("hazmat/output.txt", function(err, buf) {
  console.log(buf);
  var buf = buf.toString().split(" ");

  hazmat = document.getElementById("check1");
  general = document.getElementById("check2");
  qr = document.getElementById("check3");
  point = document.getElementById("check4");

  if (buf[0] === "true") {
    hazmat.checked = true;
  }

  if (buf[1] === "true") {
    general.checked = true;
  }

  if (buf[2] === "true") {
    qr.checked = true;
  }

  if (buf[5] === "true") {
    point.checked = true;
  }

  hazmat.addEventListener('change', (event) => {
    update()
})

  general.addEventListener('change', (event) => {
    update()
})

  qr.addEventListener('change', (event) => {
    update()
})
  point.addEventListener('change', (event) => {
    update()
})
});





$('#iframe').css('pointer-events', 'none');
//
// $(document).mousemove(function(event){
//   console.log("X: " + event.pageX + ", Y: " + event.pageY);
//   var x = event.pageX - $('#iframe').offset().left;
//   var y = event.pageY - $('#iframe').offset().top;
// });
//
function handler(e) {
    e = e || window.event;
    var pageX = e.pageX;
    var pageY = e.pageY;
    // IE 8
    var y = document.getElementById("iframe").scrollHeight;
    var x = document.getElementById("iframe").scrollWidth;

    if (pageX < x) {
      if (pageY < y) {
        point.checked = true;
        update()
        console.log("inside");
        //var xnorm = pageX / x;
        //var ynorm = pageY / y;
        //console.log(xnorm, ynorm);
      }
    }
    console.log(x, y);
    console.log(pageX, pageY);
}

// attach handler to the click event of the document

window.addEventListener('click', handler);
