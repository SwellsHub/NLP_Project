import csv
import json
import random

linesList = []
with open('dataset_20200716.csv', newline='') as tokcsv:
    reader = csv.reader(tokcsv)
    for li, row in enumerate(reader):
        if row[4] != '[]':
            if len(row[4].split("'")) >= 4:
                linesList.append({"utterance": row[3], "emotion": row[4].split("'")[3]})



random.shuffle(linesList)

print(len(linesList))

with open('data/formattedTOKTrain.json', 'w') as tokjson:
    for li, row in enumerate(linesList):
        if li < 1413:
            tokjson.write(json.dumps(row) + "\n")

with open('data/formattedTOKValid.json', 'w') as tokjson:
    for li, row in enumerate(linesList):
        if li >= 1413 and li < 1767:
            tokjson.write(json.dumps(row) + "\n")

with open('data/formattedTOKTest.json', 'w') as tokjson:
    for li, row in enumerate(linesList):
        if li >= 1767:
            tokjson.write(json.dumps(row) + "\n")


#0-1412: training
#1413-1766: validation
#1766-2207: test


