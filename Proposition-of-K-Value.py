###############################################################

## UNIVERSIDADE FEDERAL DE PERNAMBUCO
## Aprendizagem de Máquina - 2017.1
## Aluno: Lucas de Souza Albuquerque

###############################################################

## A Proposal for Local k Values for k-Nearest Neighbor Rule

## Autores:
## Nicolás García-Pedrajas
## Juan A. Romero del Castillo
## Gonzalo Cerruela-García

###############################################################

## SUMÁRIO ##
## Pesquise a [TAG] para pular diretamente para uma parte do código

###############################################################

#[IMPR]

##ORIGINAL IMPORT FOR REGULAR K-NN
import csv
import random
import math
import operator

#CITE DATASETS

#[LOAD]

##LOAD(filename) - Receives a filename and reads the corresponding datafile.
def load(filename):
    file  = open(filename, "rt", encoding="utf8")
    fileContent = csv.reader(file)
    dataset = list(fileContent)
    return dataset

#[SPLT]

##SPLIT(using this initially)
def split(dataset, ratio):
    global classCol, idCol
    
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i])):
            if(j != classCol) and (j != idCol):
                dataset[i][j] = float(dataset[i][j])
        if(random.random() >= ratio): testSet.append(dataset[i])
        else: trainingSet.append(dataset[i])

#[SPLS]

def splitShuffle(dataset, crossover):
    global classCol, idCol

    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i])):
            if(j != classCol) and (j != idCol):
                dataset[i][j] = float(dataset[i][j])
    random.shuffle(dataset)
    splitPoint = math.floor(crossover*len(dataset))
    trainingSet = dataset[:splitPoint]
    testSet = dataset[splitPoint:]
    return trainingSet, testSet

#[ECLD]

##EUCLID DISTANCE (gets the distance between two instances ignoring the necessary columns)
def euclidDist(instanceA, instanceB):
    global classCol, idCol
    
    result = 0
    for i in range(len(instanceA)):
        if(i != classCol) and (i != idCol): result += pow((instanceA[i] - instanceB[i]),2)

    return math.sqrt(result)

#[KNN]

def knn(trainingSet, testSet, k):
    result = []
    for i in range(0, len(testSet)):
        knn = neighborino(trainingSet, testSet[i], k)
        knnClass = decision(knn)
        result.append(knnClass)
    return result

#[NBO]

def neighborino(trainingSet, testInst, k):

    distanceList = []
    for i in range(0, len(trainingSet)):
        currDistance = euclidDist(testInst, trainingSet[i][:])
        distanceList.append((trainingSet[i][:], currDistance))
    distanceList.sort(key=operator.itemgetter(1))
    result = []
    for i in range(0, k):
        result.append(distanceList[i][:])
    return result

#[DEC]

def decision(knn):
    global weight, classCol
    classList = {}
    maxVal = 0
    result = ""
    for i in range(0, len(knn)):
        knnClass = knn[i][0][classCol]
        if knnClass in classList:
            if(knn[i][1] == 0):
                result = knnClass
                break
            else:
                if(weight.lower() == "y"): classList[knnClass]+= (1/(knn[i][1]*knn[i][1]))
                else: classList[knnClass] += 1
        else:
            if(knn[i][1] == 0):
                result = knnClass
                break
            else:
                if(weight.lower() == "y"): classList[knnClass] = (1/(knn[i][1]*knn[i][1]))
                else: classList[knnClass] = 1
        if(classList[knnClass] > maxVal):
            maxVal = classList[knnClass]
            result = knnClass
        elif(classList[knnClass] == maxVal):
            if(random.random() > 0.5):
                maxVal = classList[knnClass]
                result = knnClass
    return result

#[BMU]

def getBMU(dataset, testInst, num):
    ##Inits values
    result = []
    distList = []
    ##Combs through the dataset
    for i in dataset:
        dist = euclidDist(i, testInst)
        distList.append((i, dist))
    ##Combs through the distance array to get the smallest value (1-nn)
    distList.sort(key=operator.itemgetter(1))
    for d in range(0, num):
        result.append(distList[d])
    return result

#[CNN]

def cnn(trainingSet):
    global classCol
    result = [trainingSet[0]]
    prevResult = []
    while prevResult != result:
        prevResult = result
        for inst in trainingSet:
            bmus = neighborino(result, inst, 1)
            if(inst[classCol] != decision(bmus)):
                result.append(inst)
        for inst in trainingSet:
            bmus = neighborino(result, inst, 1)
            if(inst[classCol] != decision(bmus)):
                result.append(inst)
    return result

#[ACC]

##SUCCESSRATE (currently without 10-fold, calculates success rate)
def successRate(testSet, knnSet):
    global classCol
    result = 0
    for i in range(0, len(testSet)):
        if(testSet[i][classCol]==knnSet[i]):
            result +=1
    result = result/float(len(testSet))*100
    return result

#[NORM]

def normalize(dataset):
    global classCol, idCol
    for x in range(0, len(dataset[0])):
        if(x != classCol) and (x != idCol):
            minval = float(dataset[0][x])
            maxval = float(dataset[0][x])
            for y in dataset:
                if(float(y[x]) < minval): minval = float(y[x])
                if(float(y[x]) > maxval): maxval = float(y[x])
            for y in dataset:
                y[x] = (float(y[x]) - minval)/(maxval - minval)
    return dataset

#[CVG]
    
##CURR WITHOUT 10-CV (NEED TO ADD)
def genFolds(dataset, kfold):
    global classCol, idCol
    sets = []
    for i in range(0, kfold):
        sets.append([])
    maxsize = math.ceil(len(dataset)/kfold)
    
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i])):
            if(j != classCol) and (j != idCol):
                dataset[i][j] = float(dataset[i][j])
        num = random.randint(0, ((kfold)-1))
        while(len(sets[num]) >= maxsize): num = random.randint(0, ((kfold)-1))
        sets[num].append(dataset[i])
    ##print(sets)
    ##input()
    return sets

#[CVE]

def execFolds(folds, kfold, kMin, kMax):
    listVals = []
    for k in range(kMin, (kMax+1)):
        listVals.append([k, 0])
    for i in range(0, kfold):
        testSet = folds[i]
        trainingSet = []
        for j in range(0, kfold):
            if(i != j): trainingSet += folds[j]
        for k in range(kMin, (kMax+1)):
            currVal = successRate(testSet, knn(trainingSet, testSet, k))
            listVals[(k-kMin)][1] += currVal
        ##print(listVals)
        ##input()
    for k in range(kMin, (kMax+1)):
        listVals[(k-kMin)][1] /= kfold
    ##print(listVals)
    ##input()
    return listVals

#[PRINT]

def printResults(trainingSet, testSet, k):
    print("Success Rate for %d Neighbors: %.4f" %(k,successRate(testSet, knn(trainingSet, testSet, k))))

#[BASE]

def loop(trainingSet, testSet, kMin, kMax):
    listVals = []
    for k in range(kMin, (kMax+1)):
        currVal = successRate(testSet, knn(trainingSet, testSet, k))
        listVals.append([k, currVal])
    return listVals

#[LKTR] - LOCAL-KNN (TRAINING) - BETTER VERSION

def betLocalF(prototypes, testSet, kMin, kMax):
    thisTest = testSet[:]
    thisProt = prototypes[:]
    testProt = []
    retProt = []
    
    for inst in thisTest:
        bmus = neighborino(thisProt, inst, 3)
        testProt.append([bmus[0][0],bmus[1][0],bmus[2][0]])

    for prot in thisProt:
        protSet = []
        for i in range(0, len(testProt)):
            if(prot == testProt[i][0] or
               prot == testProt[i][1] or
               prot == testProt[i][2]):
                protSet.append(thisTest[i][:])

        if(protSet != []):
            maxScore = 0
            maxK = 0
            for k in range(kMin, (kMax+1)):
                currVal = successRate(protSet, knn(thisProt, protSet, k))
                currPos = k-kMin
                currVal += cvGlobalK[currPos][1]
                if(currVal > maxScore):
                    maxScore = currVal
                    maxK = k
            prot2 = prot[:]
            prot2.append(maxK)
            retProt.append(prot2[:])

    return retProt

#[LKTR] - LOCAL-KNN (TRAINING)
    
def localF(prototypes, testSet, kMin, kMax):

    for prototype in prototypes:
        protSet = []
        for inst in testSet:
            ##DE ALGUM JEITO ESTÁ INSERINDO INSTÂNCIAS EM TESTE?
            ##DEPOIS DE UNS LOOPS ELE APARECE COM O PROTÓTIPO
            bmus = neighborino(prototypes, inst, 3)
            if(bmus[0][0] == prototype or bmus[1][0] == prototype or bmus[2][0]):
                protSet.append(inst)
        listVals = []
        for k in range(kMin, (kMax+1)):
            currVal = successRate(protSet, knn(prototypes, protSet, k))
            listVals.append([k, currVal])
            currPos = k-kMin
            listVals[currPos][1] = listVals[currPos][1] + cvGlobalK[currPos][1]
        listVals.sort(key=operator.itemgetter(1))
        prototype.append(listVals[0][0])
    return prototypes

#[LKTE] - LOCAL-KNN (TEST)

def knnlocal(prototypes, testSet):
    result = []
    for inst in testSet:
        bmus = neighborino(prototypes, inst, 1)
        k = bmus[0][0][-1]
        bmus = neighborino(prototypes, inst, k)
        currResult = decision(bmus)
        result.append(currResult)
    rate = successRate(testSet, result)
    return rate


#filename = input("Please input your file name (without the .csv extension) >> ")
#filename += ".csv"
#weight = input("Type 'y' for weighted K-NN >> ")
#local = input("Type 'y' for local K-NN >> ")
#classCol = (int(input("What column of your dataset contains the expected result? >> ")) - 1)
#idCol = (int(input("Type the identification column, if any (-999 if none) >> ")) - 1)

#[MAIN]

for abc in range(0,5):
    print("ROUND",abc)
    filename = "seeds.csv"
    weight = local = "y"
    classCol = 7
    idCol = -999

    trainingSet = []
    testSet = []
    dataset = load(filename)
    datasetNorm = normalize(dataset[:])
    trainingSet, testSet = splitShuffle(datasetNorm[:], 0.67)
    #split(datasetNorm, 0.67)
    prototypes = cnn(trainingSet[:])
    ##prototypes = cnn(datasetNorm)
    folds = genFolds(datasetNorm[:], 10)

    #prototypes = rnn(prototypes, trainingSet)
    #print(prototypes)

    cvGlobalK = execFolds(folds[:],10,1,10)
    regularKnn = loop(trainingSet[:],testSet[:],1,10)
    protKnn = loop(prototypes[:],testSet[:],1,10)

    oldProts = betLocalF(prototypes[:],testSet[:],1,10)
    newProts = betLocalF(trainingSet[:],testSet[:],1,10)
    rate1 = knnlocal(oldProts[:], testSet[:])
    rate2 = knnlocal(newProts[:], testSet[:])

    cvGlobalK.sort(key=operator.itemgetter(1))
    regularKnn.sort(key=operator.itemgetter(1))
    protKnn.sort(key=operator.itemgetter(1))
    print("cvGlobalK (10-fold)", cvGlobalK[0])
    print("regularKnn", regularKnn[0])
    print("prototypeKnn (cnn, globalK)", protKnn[0])
    print("localKnn + cnnProts = ", rate1)
    print("localKnn + fullTrainingSet = ", rate2)
    print("")
print("###")




#Modificar LocalK (criação de sets) - PRIORIDADE PRA VENTOLOCIDADE
    #CHECK
#Cuidar de problemas de referência
    #ACHOQCHECKHU3
#Outros modos de prototipagem?
    #PROX PRIORIDADE
#Outros modos de split/cross-validation?
    #ACHO NÃO NECESSÁRIO
#Outros modos de comparação? (curva Roc/cohen/wilcoxon)
    #SEGUINDO O ARQUIVO
#Outros KNN básicos?
    #SEGUINDO O ARQUIVO
#COMB THROUGH EVERYTHING AND HIGHLIGHT ARTICLE PERFORMANCE

