import csv
import json
import random

linesList = []
sadness = ["Sad"]
happiness = ["Laughing", "Greeting", "Worship", "Flirt", "Drunk"]
excited = ["Injured", "Victory"]
convincing = ["Pleading", "Persuade"]
anger = ["Taunt", "Forceful", "Horror"]
neutral = ["Ready", "Inject", "Prone", "Computer", "Normal",
            "Listen", "Sleep", "Bow", "Salute", "Pause", "Scratch", "Dead"]

def insertEmotion(row):
    if any(x in row[4].split("'")[3] for x in sadness):
        linesList.append({"utterance": row[3], "emotion": "sad"})
    elif any(x in row[4].split("'")[3] for x in happiness):
        linesList.append({"utterance": row[3], "emotion": "happy"})
    elif any(x in row[4].split("'")[3] for x in excited):
        linesList.append({"utterance": row[3], "emotion": "excited"})
    elif any(x in row[4].split("'")[3] for x in anger):
        linesList.append({"utterance": row[3], "emotion": "anger"})
    elif any(x in row[4].split("'")[3] for x in convincing):
        linesList.append({"utterance": row[3], "emotion": "convincing"})
    elif any(x in row[4].split("'")[3] for x in neutral):
        linesList.append({"utterance": row[3], "emotion": "neutral"})


with open('dataset_20200716.csv', newline='') as tokcsv:
    reader = csv.reader(tokcsv)
    for li, row in enumerate(reader):
        if row[4] != '[]':
            if len(row[4].split("'")) >= 4:
                insertEmotion(row)
                

print(linesList[0:10])
print(len(linesList))

random.shuffle(linesList)

with open('data/formattedTOKTrainC.json', 'w') as tokjson:
    for li, row in enumerate(linesList):
        if li < 1413:
            tokjson.write(json.dumps(row) + "\n")

with open('data/formattedTOKValidC.json', 'w') as tokjson:
    for li, row in enumerate(linesList):
        if li >= 1413 and li < 1767:
            tokjson.write(json.dumps(row) + "\n")

with open('data/formattedTOKTestC.json', 'w') as tokjson:
    for li, row in enumerate(linesList):
        if li >= 1767:
            tokjson.write(json.dumps(row) + "\n")


#0-1412: training
#1413-1766: validation
#1766-2207: test


