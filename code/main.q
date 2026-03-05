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





// Neural Net

initParams:{[]}

relU:{[]}

softMax:{[]}

forProp:{[]}

oneHot:{[]}

deriveRelU:{where x > 0}

backProp:{[]}

updateParams:{[]}





// Testing Model

getPredict:{[]}

getAccuracy:{[]}

gradDecent:{[]}







