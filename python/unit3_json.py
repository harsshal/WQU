"""
__author__ = 'harsshal'

import pandas as pd

with open('nutrients/nutrients.json', 'r') as f:
    data = f.readlines()

    data = map(lambda x: x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"

    # now, load it into pandas
    data_df = pd.read_json(data_json_str)

    df = data_df[['group','name','nutrients']]

#amino = pd.read_csv('nutrients/amino.csv')

amino = ["alanine","arginine","asparagine","aspartic acid","cysteine",
         "glutamine","glutamic acid","glycine","histidine","isoleucine",
         "leucine","lysine","methionine","phenylalanine","proline",
         "serine","threonine","tryptophan","tyrosine","valine"]

food = {}

for i in range(len(df)):
    foodStr = df.iloc[i]['name'][u'long'].encode('utf-8').lower().strip()
    for ntr in df.iloc[i]['nutrients']:
        ntrStr = ntr[u'name'].encode('utf-8').lower().strip()
        #print(" "+ntrStr)
        if ntrStr in amino:
            if ntrStr in food:
                food[ntrStr].append(foodStr)
        else:
            food[ntrStr] = [foodStr]

for f in food:
    print(f)
print(food[f])

zinc = {}
for i in range(len(df)):
    zincQty = float(ntr[u'value'].encode('utf-8'))
foodGrp = df.iloc[i]['group'].encode('utf-8').lower().strip()
for ntr in df.iloc[i]['nutrients']:
    ntrStr = ntr[u'name'].encode('utf-8').lower().strip()
#print(" "+ntrStr)
if ntrStr == "zinc, zn":
    if foodGrp in zinc:
    zinc[foodGrp].append(zincQty)
else:
zinc[foodGrp] = [zincQty]

for f in zinc:
    print(f)
median = int (len(zinc[f]) / 2)
print(len(zinc[f]), median)
zinc[f].sort()
if len(zinc[f]) == 0:
    print('None')
else:
print(zinc[f][median])

"""

import pandas as pd
import sys

def main():
    if len(sys.argv) < 2:
        print("Correct Usage : harshal-u3-mp3 jsonFile")
        exit()
    script,jsonFile = sys.argv
    with open(jsonFile, 'r') as f:
        data = f.readlines()
        data = map(lambda x: x.rstrip(), data)
        data_json_str = "[" + ','.join(data) + "]"
        # now, load it into pandas
        data_df = pd.read_json(data_json_str)
        df = data_df[['group','name','nutrients']]


if __name__ == "__main__":
    main()
