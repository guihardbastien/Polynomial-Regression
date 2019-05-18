//dataset
let x_vals = [], y_vals = [];

//variables of polynomial function
let a, b, c, d;

//Coefficient by how much we'll move from the gradient during training
const LEARNING_RATE = 0.5;

// Stochastic gradient descent (SGD), function used to minimise mean squared error see README.md for more info
const optimizer = tf.train.sgd(LEARNING_RATE);

//Mean squared error :  mean((guess - actual_y)^2), is used to compute the error
const loss = (pred, labels) => pred.sub(labels).square().mean();

//Uses p5 library to create canvas and initialise 4 tensorflow-friendly random variable
function setup() {
    createCanvas(window.innerWidth, window.innerHeight);
    background(0);

    //using p5 random and tensorflow scalar method to initialise a random number
    a = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
    d = tf.variable(tf.scalar(random(-1, 1)));
}

//Returns predicted Ys given input Xs data
function predict(x) {
    const xTensor = tf.tensor1d(x);
    //y= ax^3+bx^2+cx+d
    const yTensor = xTensor.pow(tf.scalar(3)).mul(a).add(xTensor.square().mul(b)).add(xTensor.mul(c)).add(d);
    //console.log(yTensor);
    return yTensor;
}

//Pushes data inside dataset
function mousePressed() {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);

    x_vals.push(x);
    y_vals.push(y);
}

//p5.js draw function
function draw() {

    // clears gpu of tensors stored in memory
    tf.tidy(() => {
        if (x_vals.length > 1) {
            const ys = tf.tensor1d(y_vals);
            //Minimizing error
            optimizer.minimize(() => loss(predict(x_vals), ys));
        }
    });

    //p5.js render info
    background(51);
    stroke(255);
    strokeWeight(4);

    //Renders point on canvas see p5.js doc
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], -1, 1, 0, width);
        let py = map(y_vals[i], -1, 1, height, 0);
        point(px, py);
    }
    drawLine();
}

function drawLine() {

    //x coordinates of the drawn curve
    const curveX = [];

    //add x coordinates to render the curve
    for (let x = -1; x <= 1.01; x += 0.01) {
        curveX.push(x);
    }

    //creates the curve as its fitting
    const ys = tf.tidy(() => predict(curveX));
    console.log(ys);

    //Synchronously downloads the values from the tf.Tensor.
    let curveY = ys.dataSync();
    ys.dispose();


    //renders the curve
    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i < curveX.length; i++) {
        let x = map(curveX[i], -1, 1, 0, width);
        let y = map(curveY[i], -1, 1, height, 0);
        vertex(x, y);
    }
    endShape();

    //Checking memory leaks
    //console.log(tf.memory().numTensors)
}

