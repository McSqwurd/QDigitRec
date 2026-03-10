// Digit Classification in kdb - Complete Script
// Using MNIST dataset - 70,000 handwritten digits from 0-9
// test.csv for testing, train.csv for training
// Image size = 28x28 pixels -> pixel value ranges from 0-255
// input layer a0 = 784 units, hidden layer a1 = 10 units with ReLU activation
// output layer a2 = 10 units corresponding to 10 digit classes with softmax activation

system "S ",string `int$.z.t;                                                            / Initialize random seed based on current time

// DATA LOADING AND PREPROCESSING
loadIn:{[x] n:count "," vs first read0 x; (n#"J"; enlist ",") 0: x};                     / Load CSV with auto column count
tab:loadIn[`:../data/train.csv];                                                         / Load training data as table

row:count tab;                                                                           / Number of rows in data
colmn:count cols tab;                                                                    / Number of columns in data
shuffle:{[t] neg[count t]?t};                                                            / Shuffle function - randomises row order
tab:shuffle[tab];                                                                        / Apply shuffle to training data

/ Dev Set -> first 1000 rows
dataDev:1000#'(flip tab);                                                                / Transpose and take first 1000 values per key
yDev:dataDev[`label];                                                                    / Extract dev labels (1000 values)
xDev:(`label _ dataDev) % 255.0;                                                         / Remove label, normalise pixel values 0-1
xDevMat:value xDev;                                                                      / Convert to 784 x 1000 matrix

/ Training Set -> rows 1000 onwards
dataTrain:1000 _' (flip tab);                                                            / Transpose and drop first 1000 values per key
yTrain:dataTrain[`label];                                                                / Extract training labels
xTrain:(`label _ dataTrain) % 255.0;                                                     / Remove label, normalise pixel values 0-1
xTrainMat:value xTrain;                                                                  / Convert to 784 x mTrain matrix
mTrain:count first xTrainMat;                                                            / Number of training examples

// NEURAL NETWORK INITIALIZATION
initP:{[]
    w1:(10 784 # (10*784)?1.0) - 0.5;
    b1:(10?1.0) - 0.5;
    w2:(10 10 # (10*10)?1.0) - 0.5;
    b2:(10?1.0) - 0.5;
    `w1`b1`w2`b2!(w1;b1;w2;b2)};

// ACTIVATION FUNCTIONS
ReLU: {x | 0};                                                                           / ReLU activation: max(x, 0) - introduces non-linearity
ReLUDeriv: {x > 0};                                                                      / ReLU derivative: 1 if x>0, else 0
softmax: {flip (flip exp x) % sum exp x};                                                / Softmax: normalized exponential - sum of all output values = 1 {o<x<1}
oneHot: {(til 10) =\: x};                                                                / convert to binary vector

// FORWARD PROPAGATION
forwardProp: {[W1;b1;W2;b2;X]
    Z1: (W1 mmu X) + b1;                                                                 / Linear combination layer 1,Z1 = first layer linear combination + bias
    A1: ReLU Z1;                                                                         / Activation layer 1, A1 = Activation function applied on first layer = ReLU
    Z2: (W2 mmu A1) + b2;                                                                / Linear combination layer 2, Z2 = second layer linear combination + bias
    A2: softmax Z2;                                                                      / Activation layer 2 (output), A2 = Activation function applied on second layer = softmax (output layer)
    `Z1`A1`Z2`A2!(Z1;A1;Z2;A2) };                                                        / Return all layer outputs

// BACKWARD PROPAGATION
backwardProp: {[fwd;params;X;Y;m]                                                        / fwd: dict
    oneHotY: oneHot Y;                                                                   / One-hot encode true labels
    dZ2: (fwd`A2) - oneHotY;                                                             / Output layer error
    dW2: (1%m) * dZ2 mmu flip (fwd`A1);                                                  / Weight gradient layer 2
    db2: (1%m) * sum sum dZ2;                                                            / Bias gradient layer 2
    dZ1: (flip[params`W2] mmu dZ2) * ReLUDeriv (fwd`Z1);                                             / Hidden layer error
    dW1: (1%m) * dZ1 mmu flip X;                                                         / Weight gradient layer 1
    db1: (1%m) * sum sum dZ1;                                                            / Bias gradient layer 1
    `dW1`db1`dW2`db2!(dW1;db1;dW2;db2)                                                   / Return all gradients
    };

// UPDATE PARAMETERS
updateParams: {[params;gradients;alpha]                                                  / Gradient descent update - adjust weights and biases based on gradients
    W1_new: (params`W1) - alpha * (gradients`dW1);                                       / Update weight matrix 1
    b1_new: (params`b1) - alpha * (gradients`db1);                                       / Update bias vector 1
    W2_new: (params`W2) - alpha * (gradients`dW2);                                       / Update weight matrix 2
    b2_new: (params`b2) - alpha * (gradients`db2);                                       / Update bias vector 2
    `W1`b1`W2`b2!(W1_new;b1_new;W2_new;b2_new) };                                        / Return updated parameters

// PREDICTIONS AND ACCURACY
getPredictions: {[A2]                                                                    / Get predictions from output layer (A2) - argmax per column
    first each idesc each flip A2 };

getAccuracy: {[predictions;Y]                                                            / Get accuracy - compare predictions with true labels
    (sum predictions = Y) % count Y};

// GRADIENT DESCENT TRAINING
gradientDescent: {[X;Y;alpha;iterations]                                                 / Main training loop - trains the network over multiple iterations
    initParams: initP[];
    W1: initParams`w1;
    b1: initParams`b1;
    W2: initParams`w2;
    b2: initParams`b2;
    m: mTrain;
    
    i: 0;
    while[i < iterations;
        fwd: forwardProp[W1;b1;W2;b2;X];                                                 / Forward propagation
        A2: fwd`A2;
        
        paramDict: `W1`W2!(W1;W2); 
        bwd: backwardProp[fwd;paramDict;X;Y;m];                                          / Backward propagation
        
        currentParams: `W1`b1`W2`b2!(W1;b1;W2;b2); 
        upd: updateParams[currentParams;bwd;alpha];                                      / Update parameters
        
        W1: upd`W1;                                                                      / Updated weight matrix layer 1
        b1: upd`b1;                                                                      / Updated bias vector layer 1
        W2: upd`W2;                                                                      / Updated weight matrix layer 2
        b2: upd`b2;                                                                      / Updated bias vector layer 2
 
        if[0 = i mod 10;                                                                 / Print progress every 10 iterations
            show "Iteration: ", string i;
            predictions: getPredictions A2;
            acc: getAccuracy[predictions;Y];
            show "Training Accuracy: ", string acc;
        ];
        i: i + 1;
    ];
    `W1`b1`W2`b2!(W1;b1;W2;b2) };

show "Starting NN Training.";

start_time: .z.T;
trainedParams: gradientDescent[xTrainMat; yTrain; 0.10; 50];
elapsed: .z.T - start_time;
show "Training completed in ", (string elapsed), " s";

/ Extract final parameters
W1_final: trainedParams`W1;
b1_final: trainedParams`b1;
W2_final: trainedParams`W2;
b2_final: trainedParams`b2;

show "Successfully computed W1, b1, W2, b2";

// TEST ON DEV SET
show "Testing on dev set.";

/ Make predictions on dev set
fwdDev: forwardProp[W1_final;b1_final;W2_final;b2_final;xDevMat];
devPredictions: getPredictions fwdDev`A2;
devAccuracy: getAccuracy[devPredictions;yDev];

show "Dev Set Accuracy: ", string devAccuracy;

// HELPER FUNCTION - Test individual predictions
testPrediction: {[index;X;Y;W1;b1;W2;b2]                                                / int, matrix (784xm), Vector values, matrix (10x84), vector 10 values, matrix (10x10), vector (10 values)
    currentImage: X[;index];                                                            / Extract single image at given index (returns 1D vector of 784 pixels)
    currentImage: (784;1) # currentImage;                                               / Convert 1D vector to 2D matrix (784 rows, 1 column) for matrix multiplication
    fwd: forwardProp[W1;b1;W2;b2;currentImage];                                         / Run forward propagation on single image
    prediction: getPredictions fwd`A2;                                                  / Get predicted digit (0-9) from output layer
    label: Y[index];                                                                    / Get true digit label
    show "Test index: ", raze (string index); 
    show "Prediction: ", raze (string prediction);
    show "True Label: ", raze (string label);
    show "Correct: ", raze (string (prediction = label))};

// OPTIONAL: Test a few predictions
show "Sample predictions:";
testPrediction[0;xTrainMat;yTrain;W1_final;b1_final;W2_final;b2_final];
testPrediction[100;xTrainMat;yTrain;W1_final;b1_final;W2_final;b2_final];
testPrediction[500;xTrainMat;yTrain;W1_final;b1_final;W2_final;b2_final];

