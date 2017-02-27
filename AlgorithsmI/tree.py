from PIL import Image,ImageDraw


my_data=[['slashdot','USA','yes',18,'None'],
         ['google','France','yes',23,'Premium'],
         ['digg','USA','yes',24,'Basic'],
         ['kiwitobes','France','yes',23,'Basic'],
         ['google','UK','no',21,'Premium'],
         ['(direct)','New Zealand','no',12,'None'],
         ['(direct)','UK','no',21,'Basic'],
         ['google','USA','no',24,'Premium'],
         ['slashdot','France','yes',19,'None'],
         ['digg','USA','no',18,'None'],
         ['google','UK','no',18,'None'],
         ['kiwitobes','UK','no',19,'None'],
         ['digg','New Zealand','yes',12,'Basic'],
         ['slashdot','UK','no',21,'None'],
         ['google','UK','yes',18,'Basic'],
         ['kiwitobes','France','yes',19,'Basic']]

#my_data=[['slashdot','USA','None'],
#         ['google','France','Premium'],
#         ['kiwitobes','France','Basic']]

class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col
        self.value=value
        self.results=results
        self.tb=tb
        self.fb=fb

# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows,column,value, greaterEqual = 0):
    # Make a function that tells us if a row is in
    # the first group (true) or the second group (false)
    split_function=None
    if (isinstance(value,int) or isinstance(value,float) ) and greaterEqual :
        split_function=lambda row:row[column]>=value
    else:
        split_function=lambda row:row[column]==value
    # Divide the rows into two sets and return them
    set1=[row for row in rows if split_function(row)]
    set2=[row for row in rows if not split_function(row)]
    return (set1,set2)

def uniquecounts(rows):
    results={}
    for row in rows:
        # The result is the last column
        r=row[len(row)-1]
        if r not in results: results[r]=0
        results[r]+=1
    return results

# Entropy is the sum of - p(x)log(p(x)) across all
# the different possible results
def entropy(rows):
    from math import log
    log2=lambda x:log(x)/log(2)
    results=uniquecounts(rows)

    # Now calculate the entropy
    ent=0.0
    for r in results.keys():
        p=float(results[r])/len(rows)
        ent=ent-p*log2(p)
    return ent

def buildLevel(rows, level, rank):
    if(level == len(rank)):
        return decisionnode(results=rows[0][len(rows[0])-1])
    countSet = getValues(rows, rank[level])
    root = None
    remainingRows = rows
    for entry in sorted(countSet):
        entryRows, remainingRows = divideset(remainingRows,rank[level],entry,0)
        if root == None:
            # Needs to strip down rows so that we can populate results in leafs
            root = decisionnode(rank[level],entry,None,buildLevel(entryRows,level+1,rank),None)
            prevNode= root
        else:
            prevNode.fb = decisionnode(rank[level],entry,None,buildLevel(entryRows,level+1,rank),None)
            prevNode = prevNode.fb
    return root


def getValues(rows, column):
    return uniquecounts([[rows[i][column]] for i in range(len(rows))])

def buildtree(rows,scoref=entropy):
    if len(rows)==0: return decisionnode()

    # find out hierarchy of columns
    Basic_score=scoref(rows)
    print("Basic Score : ",Basic_score)
    totalRows = len(rows)
    infoGain = [0] * len(rows[0])
    for column in range(len(rows[0])):
        countSet = getValues(rows, column)
        remainingRows = rows
        finalEntropy = 0
        bestNumEntropy = bestEntry = 0
        for entry in countSet:
            # For every entry in given column, calculate part
            # of the entropy contributing towards final entropy
            entryRows, remainingRows = divideset(remainingRows,column,entry,0)
            finalEntropy += (countSet[entry]/totalRows) * entropy(entryRows)

            # If column in numerical, we can use buckets of each entry or just 2 buckets
            # For simplicity we will look into only these 2 options
            if isinstance(entry,int) or isinstance(entry,float):
                numEntropy=0
                numEntryRows, numRemainingRows = divideset(rows,column,entry,1)
                numEntropy += (len(numEntryRows)/totalRows) * entropy(numEntryRows)
                numEntropy += ((totalRows- len(numEntryRows))/totalRows) * entropy(numRemainingRows)

                # we can use use binary search on sorted array of possible numerical values
                # instead of linear search. But this is easier.
                if bestNumEntropy == 0 or bestNumEntropy > numEntropy:
                    bestNumEntropy = numEntropy
                    bestEntry = entry

        # best entropy in 2 bucket scenario
        if isinstance(entry,int) or isinstance(entry,float):
            print(bestEntry,bestNumEntropy)

        currentInfoGain = Basic_score - finalEntropy

        # insert into end of sorted array of infogain
        # and then track down its correct position
        infoGain[column] = currentInfoGain

        rank = [i for i in range(len(rows[0]))]
        for i in range(len(rank)-1):
            for j in range(i+1 ,len(rank)- 1):
                if infoGain[i] < infoGain[j] :
                    temp = rank[j]
                    rank[j] = rank[i]
                    rank[i] = temp
        print(column, finalEntropy)


    # start creating nodes based on that hierarchy
    # return root node who will have getwidth and getdepth functions


    root = buildLevel(rows, 0, rank[:len(rank)-1])


    Best_gain=0
    Best_criteria=None
    Best_sets=None

    # return root
    drawtree(root)

def drawtree(tree,jpeg='tree.jpg'):
    w=getwidth(tree)*100
    h=getdepth(tree)*100+120
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)
    drawnode(draw,tree,w/2,20)
    img.save(jpeg,'JPEG')

def getwidth(tree):
    if tree == None  or tree.results != None :
        return 1
    else:
        return getwidth(tree.tb) + getwidth(tree.fb) + 1

def getdepth(tree):
    if tree == None or tree.results != None:
        return 1
    else:
        return max(getdepth(tree.tb), getdepth(tree.fb)) + 1

def drawnode(draw,tree,x,y):
    if tree == None:
        txt='untrained'
        draw.text((x-20,y),txt,(0,0,0))
    elif tree.results== None:

        # Get the width of each branch
        w1=getwidth(tree.tb)*100
        w2=getwidth(tree.fb)*100

        # Determine the total space required by this node
        left=x-(w1+w2)/2
        right=x+(w1+w2)/2

        # Draw the condition string
        draw.text((x-20,y-10),str(tree.col)+':'+str(tree.value),(0,0,0))

        # Draw links to the branches
        draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
        draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))

        # Draw the branch nodes
        drawnode(draw,tree.tb,left+w1/2,y+100)
        drawnode(draw,tree.fb,right-w2/2,y+100)
    else:
        #txt='\n'.join(['%s:%d'%v for v in tree.results.items()])
        txt=tree.results
        draw.text((x-20,y),txt,(0,0,0))

def main():
    print(my_data[0]);
    root = buildtree(my_data)

main()
