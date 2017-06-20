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

## 1. SET-UP EXPERIMENTAL
## 1.1. Imports Necessários [IMPR]
## 1.2. Datasets Utilizados [DATA]
## 1.3. Preparação do Dataset [LOAD]
## 1.4. Preparação dos Folds [FOLD]

## 2. K-NEAREST NEIGHBOR
## 2.1. Cálculos de Distância [DIST]
## 2.2. Funções do K-NN padrão [K-NN]
## 2.3. Geração de Protótipos [PROT]
## 2.4. Cálculo da Taxa de Acerto [ACCR]

## 3. CÁLCULO DO MELHOR K LOCAL
## 3.1. K GLOBAIS [KGLO]
## 3.2. K LOCAIS [KLOC]

## 4. FUNÇÕES ADICIONAIS [ADDF]

## 5. MAIN [MAIN]

###############################################################

## 1. SET-UP EXPERIMENTAL

# 1.1. Imports Necessários [IMPR] #
# Foram se utilizadas as bibliotecas de CSV (para leitura de arquivos), random (para geração
# de números pseudo-aleatórios), math (para operações de raiz quadrada e potenciação) e
# operator (para ordenação de listas) nesse trabalho.

import csv
import random
import math
import operator
import cohen

##############

# 1.2. Datasets Utilizados [DATA] #
# O artigo menciona ter usado vários datasets da database do UCI, mas não especifica quais.
# Tentei entrar em contato com os autores do arquivo para saber quais bancos de dados foram usados
# e como eles foram particionados, como sugerido no próprio artigo, mas não consegui resposta e o
# link fornecido no artigo tinha expirado.

# Com isso, foram-se usados os seguintes Datasets do UCI, em ordem crescente de número de instâncias.

# IRIS - 150 instâncias, 4 atributos (1988) - Class: 4 - Id: -999
# WINE - 178 instâncias, 13 atributos (1991) - Class: 0 - Id: -999
# PARKINSONS - 197 instâncias, 23 atributos (2008) - Class: 17 - Id: 0
# SONAR - 208 instâncias, 60 atributos - Class: 60 - Id: -999
# SEEDS - 210 instâncias, 7 atributos (2012) - Class: 7 - Id: -999
# GLASS - 214 instâncias, 10 atributos (1987) - Class: 10 - Id: 0
# HABERMAN - 306 instâncias, 3 atributos (1999) - Class: 3 - Id: -999
# ECOLI - 336 instâncias, 8 atributos (1996) - Class: 8 - Id: 0
# LEAF - 340 instâncias, 16 atributos (2014) - Class: 0 - Id: 1
# IONOSPHERE - 351 instâncias, 34 atributos (1989) - Class: 34 - Id: -999
    
##############

# 1.3. Preparação do Dataset [LOAD] #
# Essa seção contém as funções básicas de leitura, preparação e separação do dataset.

## load(filename)
## Recebe um nome de arquivo e lê o arquivo correspondente.

def load(filename):
    file  = open(filename, "rt", encoding="utf8")
    fileContent = csv.reader(file)
    dataset = list(fileContent)
    return dataset

#-------------------#

## normalize(dataset)
## Recebe um dataset e normaliza as colunas.

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
                if(minval == 0) and (maxval == 0): y[x] = 0
                else: y[x] = (float(y[x]) - minval)/(maxval - minval)
    return dataset

#-------------------#

# Observação:
# O artigo não define como ele particiona os bancos para gerar os protótipos ou para encontrar as instâncias
# de teste mais próximas à estes.

# As funções splits abaixo foram usadas para gerar bancos de treinamento (que geram os protótipos) e
# bancos de teste (que serão usados na avaliação). Eles não estão relacionados com o CV de 10-folds usado
# para calcular os valores de acerto globais para cada 'k'

## split(dataset, ratio)
## Versão inicial do split - recebe um banco e um fator e distribui instâncias para o banco de treinamento ou
## teste dependendo de se um valor aleatório for maior ou menor que o fator.

def split(dataset, ratio):
    global classCol, idCol
    
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i])):
            if(j != classCol) and (j != idCol):
                dataset[i][j] = float(dataset[i][j])
        if(random.random() >= ratio): testSet.append(dataset[i])
        else: trainingSet.append(dataset[i])
        
## splitShuffle(dataset, crossover)
## Versão avançada do split - recebe um banco e um ponto de crossover, dá shuffle no dataset, e divide-o
## nos bancos de teste e treinamento no ponto do crossover.
        
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

##############

# 1.4. Preparação dos Folds [FOLD] #
# Essa seção contém a funções básicas que gera os folds para a cross-validation.

## genFolds(dataset, kfold)
## Recebe um banco de dados e um número de folds e separa o número de folds necessários.
## Obs: Não permite um fold ter mais de duas instâncias do que qualquer outro.

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
    return sets

###############################################################

## 2. FUNÇÕES DE K-NN BÁSICO

# 2.1. Cálculo de Distância [DIST] #
# Essa seção contém funções básicas de cálculo de distância.

# Observação: 
# O artigo compara seu algoritmo com o K-NN padrão, o K-NN adaptativo e o K-NN simétrico.
# Para motivos de simplicidade neste trabalho e para foco no algoritmo apresentado, foi-se focado o K-NN padrão.
# Porém, caso desejado, não é complicado introduzir novas métricas para se comparar com outros algoritmos.

## euclidDist(instanceA, instanceB)
## Calcula a distância euclideana entre duas instâncias, ignorando colunas de classe e identificação

def euclidDist(instanceA, instanceB):
    global classCol, idCol
    
    result = 0
    for i in range(len(instanceA)):
        if(i != classCol) and (i != idCol): result += pow((instanceA[i] - instanceB[i]),2)

    return math.sqrt(result)

##############

# 2.2. Funções do K-NN Padrão [K-NN] #
# Essa seção contém as funções básicas para execução do K-NN padrão

## knn(trainingSet, testSet, k)
## Recebe um banco de treinamento e um banco de teste, e para cada instância do banco de teste, pega os
## k vizinhos mais próximos, usa eles para calcular a classe prevista, e retorna uma lista dessas classes.

def knn(trainingSet, testSet, k):
    result = []
    for i in range(0, len(testSet)):
        knn = neighbor(trainingSet, testSet[i], k)
        knnClass = decision(knn)
        result.append(knnClass)
    return result

#-------------------#

## neighbor(trainingSet, testInst, k)
## Recebe um banco de treinamento e uma instância de teste A, e para cada instância B do banco de treinamento,
## calcula a distância entre A e B. No final, retorna os 'k' vizinhos mais pertos e as distâncias associadas.

def neighbor(trainingSet, testInst, k):

    distanceList = []
    for i in range(0, len(trainingSet)):
        currDistance = euclidDist(testInst, trainingSet[i][:])
        distanceList.append((trainingSet[i][:], currDistance))
    distanceList.sort(key=operator.itemgetter(1))
    result = []
    for i in range(0, k):
        result.append(distanceList[i][:])
    return result

#-------------------#

## decision(knn)
## Recebe um conjunto de instâncias (os vizinhos mais próximos) e calcula qual a classe mais prevalente.
## No final, retorna a classe prevista.

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

#-------------------#

## loop(trainingSet, testSet, kMin, kMax)
## Recebe um banco de treinamento, um banco de teste, e calcula kNN para estes bancos para todos os
## valores de 'k' entre [kMin, kMax], retornando uma lista de taxas de acerto.

def loop(trainingSet, testSet, kMin, kMax):
    listVals = []
    knnSets = {}
    for k in range(kMin, (kMax+1)):
        knnSet = knn(trainingSet, testSet, k)
        currVal = successRate(testSet, knnSet)
        listVals.append([k, currVal])
        knnSets[k] = knnSet
    return listVals, knnSets

##############

# 2.3. Geração de Protótipos [PROT] #
# Essa seção contém a função usada para se gerar protótipos.

# Observação: 
# O artigo não define como os protótipos foram criados.
# Por isso, foi-se usado uma versão básica do CNN (Condensed Nearest Neighbor)

# Se desejado, outros métodos de geração de protótipos podem ser usados, ou o banco de treinamento inteiro pode
# ser utilizado para o cálculo dos valores de 'k' locais.

## cnn(trainingSet)
## Recebe um banco de treinamento T e cria um banco novo S com apenas uma instância. Enquanto S for atualizado,
## verifique se cada instância de T é classificada corretamente, se não, insira esta instância em S.

## O CNN pode ser usado como ponto de partida na geração de protótipos e podem se usar outros métodos após ele!

def cnn(trainingSet):
    global classCol
    result = [trainingSet[0]]
    prevResult = []
    while prevResult != result:
        prevResult = result
        for inst in trainingSet:
            bmus = neighbor(result, inst, 1)
            if(inst[classCol] != decision(bmus)):
                result.append(inst)
        for inst in trainingSet:
            bmus = neighbor(result, inst, 1)
            if(inst[classCol] != decision(bmus)):
                result.append(inst)
    return result

##############

# 2.4. Cálculo da Taxa de Acerto [ACCR] #
# Essa seção contém a função usada para calcular a taxa de acerto de um algoritmo.

## successRate(testSet, knnSet)
## Recebe um banco de teste e uma lista de classes previstas. Para cada instância de teste T, verifica s
## a classe prevista é igual à classe da instância, e calcula a taxa de acerto em porcentagem.

def successRate(testSet, knnSet):
    global classCol
    result = 0
    for i in range(0, len(testSet)):
        if(testSet[i][classCol]==knnSet[i]):
            result +=1
    result = result/float(len(testSet))*100
    return result

###############################################################

## 3. CÁLCULO DO MELHOR K LOCAL

# 3.1. K GLOBAIS [KGLO] #
# Essa seção contém a função que calcula a accuracy dos valores globais para cada k.
# Estes valores serão utilizados para ajudar no cálculo dos valores locais.

## execFolds(folds, kfold, kMin, kMax)
## Recebe um conjunto de folds (gerados anteriormente), um número de folds e um range [kMin, kMax],
## e executa os folds para todos os valores de k dentro dos limites definidos.

## A função então retorna uma lista com todos os valores de 'k' e as taxas de acerto médias
## calculadas na execução do K-NN para cada fold.

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

##############

# 3.2. K LOCAIS [KLOC] #
# Essa seção contém as funções usadas para calcular o valor local de k associado à cada protótipo.

## localF(prototypes, testSet, kMin, kMax)
## Recebe um conjunto de protótipos e um banco de treinamento. Para cada instância T do banco de treinamento,
## gera uma lista dos três protótipos mais próximos. Então, cada protótipo cria um subconjunto do banco de teste
## contendo apenas as instâncias T que tem ele como um dos três vizinhos mais próximos.

## Se o conjunto for vazio, o protótipo é descartado: caso contrário, se calcula o K-NN sobre o subconjunto de
## teste e o conjunto de protótipos para todos os valores 'k' entre [kMin, kMax], e associa o protótipo ao
## valor 'k' com maior taxa de acerto.

## A função retorna os protótipos atualizados.

def localF(prototypes, testSet, kMin, kMax):
    thisTest = testSet[:]
    thisProt = prototypes[:]
    testProt = []
    retProt = []
    
    for inst in thisTest:
        bmus = neighbor(thisProt, inst, 3)
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

#-------------------#

## knnLocal(prototypes, testSet)
## Recebe um conjunto de protótipos e um conjunto de teste, e para cada instância T do banco de teste,
## pega o protótipo mais próxima da instância, e calcula o k-NN baseado no 'k' encontrado no protótipo.
## No final, retorna a taxa de acerto.

def knnlocal(prototypes, testSet):
    result = []
    totalK = 0
    for inst in testSet:
        bmus = neighbor(prototypes, inst, 1)
        k = bmus[0][0][-1]
        totalK += k
        bmus = neighbor(prototypes, inst, k)
        currResult = decision(bmus)
        result.append(currResult)
    avgK = totalK/(len(testSet))
    rate = successRate(testSet, result)
    return rate, avgK, result

###############################################################

## 4. FUNÇÕES ADICIONAIS [ADDF] #
# Essa seção contém funções adicionais usadas ao longo do projeto. 

## printResults(trainingSet, testSet, k)
## Recebe um banco de treinamento, um banco de teste, e calcula a taxa de acerto do k-NN para os bancos

def printResults(trainingSet, testSet, k):
    print("Success Rate for %d Neighbors: %.4f" %(k,successRate(testSet, knn(trainingSet, testSet, k))))

###############################################################

## 5. MAIN [MAIN]
## Essa seção contém o main do projeto

for round in range(0,5):
    print("ROUND",round)

    ## Informações do dataset
    filename = "ecoli.csv"
    weight = local = "y"
    classCol = 8
    idCol = 0

    ## Carrega o banco de dados e normaliza.
    trainingSet = []
    testSet = []
    dataset = load(filename)
    datasetNorm = normalize(dataset[:])

    ## Pega todas as possiveis classes
    labels = []
    for i in datasetNorm:
        if not (i[classCol] in labels):
            labels.append(i[classCol])

    ## Gera os bancos de treinamento, teste, os protótipos e os folds para CV.
    trainingSet, testSet = splitShuffle(datasetNorm[:], 0.67)
    prototypes = cnn(trainingSet[:])
    folds = genFolds(datasetNorm[:], 10)

    ## Executa K-NN padrão com 10-Fold CV, o banco de treinamento inteiro e os protótipos.
    cvGlobalK = execFolds(folds[:],10,1,10)
    regularKnn, pdtL1 = loop(trainingSet[:],testSet[:],1,10)
    protKnn, pdtL2 = loop(prototypes[:],testSet[:],1,10)

    ## Calcula os valores locais para cada k com os protótipos e banco de treinamento inteiro
    ## (Isso gerarão duas taxas de acerto diferente)
    prots1 = localF(prototypes[:],testSet[:],1,10)
    prots2 = localF(trainingSet[:],testSet[:],1,10)
    rate1, avg1, sdtPROT = knnlocal(prots1[:], testSet[:])
    rate2, avg2, sdtKNN = knnlocal(prots2[:], testSet[:])

    ## Para os K-NN padrão, imprime o melhor 'K' e a taxa de acerto associada, para os K-NN com
    ## valores locais, imprime a taxa de acerto e o valor médio dos 'K' conseguido dos protótipos.
    cvGlobalK.sort(key=operator.itemgetter(1))
    regularKnn.sort(key=operator.itemgetter(1))
    protKnn.sort(key=operator.itemgetter(1))
    print("cvGlobalK (10-fold)", cvGlobalK[0])
    print("regularKnn", regularKnn[0])
    print("prototypeKnn (cnn, globalK)", protKnn[0])
    print("localKnn + cnnProts = ", rate1, "# AVG K Value:", int(avg1))
    print("localKnn + fullTrainingSet = ", rate2, "# AVG K Value:", int(avg2))
    print("")

    ## Pega classes de resultado
    classRes = []
    for i in testSet:
        classRes.append(i[classCol])

    ## Imprime COHEN
    pdtKNN = pdtL1[regularKnn[0][0]]
    pdtPROT = pdtL2[protKnn[0][0]]
    print("K pdtKNN:",cohen.computeKappa(pdtKNN, classRes, labels))
    print("K pdtPROT:",cohen.computeKappa(pdtPROT, classRes, labels))
    print("K sdtPROT:",cohen.computeKappa(sdtPROT, classRes, labels))
    print("K sdtKNN:",cohen.computeKappa(sdtKNN, classRes, labels))
    
print("###")

###############################################################

# RELATÓRIO
### DIAGRAMA DE ALGORITMO
### TABELA DE DATASETS
### RESULTADOS
