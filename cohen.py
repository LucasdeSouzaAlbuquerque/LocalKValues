##CALCULO DE COHEN

def computeKappa(y1, y2, labels):

    table = {}
    
    for i in range(0, len(y1)): 
        first = y1[i]
        second = y2[i]
            
        if not(first in table):
            table[first] = {}
        if not(second in table[first]):
            table[first][second] = 0
            
        table[first][second] += 1

    sumRows = {}
    sumCols = {}
    
    for rowN, row in table.items():
        sumRow = 0

        for key,value in row.items():
            sumRow += value
            if not(key in sumCols):
                sumCols[key] = 0
            sumCols[key] += value
            
        sumRows[rowN] = sumRow
    sumTotal = sum(sumRows.values())

    sumDiagonal = 0
    for i in labels:
        value = 0
        if(i in table) and (i in table[i]):
            value = table[i][i]
        sumDiagonal += value

    p = sumDiagonal/sumTotal
    peSum = 0
    
    for i in labels:
        if(i in sumRows) and (i in sumCols):
            peSum += sumRows[i]*sumCols[i]
            
    pe = peSum/(sumTotal*sumTotal)

    if (1-pe) == 0: return 1
    return (p-pe)/(1-pe) 
    
