// Digit Recogniser in Q, test.csv to test NN, train.csv to train NN

system "S ",string `int$.z.t;                                        / Initialise random seed based on current time

loadIn:{[x] n:count "," vs first read0 x; (n#"J"; enlist ",") 0: x}; / Function to load CSV with auto column count

tab:loadIn[`:../data/train.csv];                                     / Load training data as table

m:count tab;                                                         / Number of rows in data
n:count cols tab;                                                    / Number of columns in data

shuffle:{[t] neg[count t]?t};                                        / Shuffle function - randomises row order
tab:shuffle[tab];                                                    / Apply shuffle to training data

/ Dev Set (first 1000 rows) 

dataDev:1000#'(flip tab);                                            / Transpose and take first 1000 values per key
yDev:dataDev[`label];                                                / Extract dev labels (1000 values)
xDev:(`label _ dataDev) % 255.0;                                     / Remove label, normalise pixel values 0-1
xDevMat:value xDev;                                                  / Convert to 784 x 1000 matrix

/ Training Set (rows 1000 onwards) 

dataTrain:1000 _' (flip tab);                                        / Transpose and drop first 1000 values per key
yTrain:dataTrain[`label];                                            / Extract training labels
xTrain:(`label _ dataTrain) % 255.0;                                 / Remove label, normalise pixel values 0-1
xTrainMat:value xTrain;                                              / Convert to 784 x mTrain matrix
mTrain:count first xTrainMat;                                        / Number of training examples








/ Neural Network Functions ---

initP:{[]
    w1:(10 784 # (10*784)?1.0) - 0.5;
    b1:(10?1.0) - 0.5;
    w2:(10 10 # (10*10)?1.0) - 0.5;
    b2:(10?1.0) - 0.5;
    /(w1;b1;w2;b2)
    `w1`b1`w2`b2!(w1;b1;w2;b2)
    }

ReLU: {x | 0};     / get max of x and 0                                                  / ReLU activation: max(x, 0)

softmax: {flip (flip exp x) % sum exp x}                                       / Softmax: exp(x) / sum(exp(x)) per column


// check might be wrong 
forwardProp: {[W1;b1;W2;b2;X]                                        / Forward propagation
    Z1: (W1 mmu X) + b1;                                             / Linear combination layer 1
    A1: ReLU Z1;                                                     / Activation layer 1
    Z2: (W2 mmu A1) + b2;                                            / Linear combination layer 2
    A2: softmax Z2;                                                  / Activation layer 2 (output)
    `Z1`A1`Z2`A2!(Z1;A1;Z2;A2) }                                     / Return all layer outputs

/ 
to test e.g
p: initP[]
fp: forwardProp[p`w1; p`b1; p`w2; p`b2; xTrainMat]
fp`A2   / access output
\


ReLUDeriv: {x > 0};                                                  / ReLU derivative: 1 if x>0, else 0

oneHot: {(til 10) =\: x};                                            / One-hot encode labels (10 classes)







/
backwardProp: {[Z1;A1;Z2;A2;W1;W2;X;Y;m]                             / Backward propagation
    oneHotY: oneHot Y;                                               / One-hot encode true labels
    dZ2: A2 - oneHotY;                                               / Output layer error
    dW2: (1%m) * dZ2 mmu flip A1;                                    / Weight gradient layer 2
    db2: (1%m) * sum sum dZ2;                                        / Bias gradient layer 2
    dZ1: (flip[W2] mmu dZ2) * ReLUDeriv Z1;                          / Hidden layer error
    dW1: (1%m) * dZ1 mmu flip X;                                     / Weight gradient layer 1
    db1: (1%m) * sum sum dZ1;                                        / Bias gradient layer 1
    `dW1`db1`dW2`db2!(dW1;db1;dW2;db2) }                             / Return all gradients

updateParams: {[W1;b1;W2;b2;dW1;db1;dW2;db2;alpha]                   / Gradient descent update
    W1: W1 - alpha * dW1;                                            / Update weight matrix 1
    b1: b1 - alpha * db1;                                            / Update bias vector 1
    W2: W2 - alpha * dW2;                                            / Update weight matrix 2
    b2: b2 - alpha * db2;                                            / Update bias vector 2
    `W1`b1`W2`b2!(W1;b1;W2;b2) }                                     / Return updated parameters
\





// Testing Model

getPredict:{[]}

getAccuracy:{[]}

gradDecent:{[]}





