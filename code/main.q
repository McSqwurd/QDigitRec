// Digit Recogniser in Q, test.csv to test NN, train.csv to train NN

system "S ",string `int$.z.t   / initilaise new random seed (make shuffle work)

loadIn: {[x] n: count "," vs first read0 x; (n#"J"; enlist ",") 0: x}   / Load train.csv

tab: loadIn[`:../data/train.csv]   / Get as table

m:count tab   / Rows of Data
n:count cols tab   / Cols of Data

shuffle:{[t] neg[count t]?t}   / mimics np.random.shuffle -> shuffles rows of tables (maintains rows ordering)
tab:shuffle[tab]   / Perform shuffle on data - doesnt do automatically


dataDev:1000#'(flip tab)   / Dictionary - each key has 1000 values
yDev:dataDev[`label]   / first 1000 `label values 
xDev: (`label _ dataDev) % 255.0   / Features normalized (784 keys, 1000 values each)
xDevMat:value xDev    / 784 x 1000 matrix -> mirrors numpy array


/ Training set (rows 1000 onwards)
dataTrain: 1000 _' (flip tab)
yTrain: dataTrain[`label]
xTrain: (`label _ dataTrain) % 255.0
xTrainMat: value xTrain
mTrain: count first xTrainMat










// Neural Net
initParams:{[]}

relU:{[]}

softMax:{[]}

forProp:{[]}

oneHot:{[]}

deriveRelU:{where x > 0}

backProp:{[]}

updateParams:{[]}





getPredict:{[]}

getAccuracy:{[]}

gradDecent:{[]}



