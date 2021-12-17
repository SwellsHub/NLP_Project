import json
from flatten_json import flatten
from pandas import json_normalize

f = open('data/dailydialog_test.json')

data = json.load(f)


with open("data/testFormatted.json", 'w') as fi:
    for item in data:
        for li, item2 in enumerate(data[item]):
            for item3  in item2:
                fi.write(json.dumps(item3) + "\n")

#with open("formattedData.json", 'r') as fi:
    #for line in fi:
        #print(line)

