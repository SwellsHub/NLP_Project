import sys

if len(sys.argv) != 3:
    print('usage: score_types.py <output> <correct>', file=sys.stderr)
    exit(1)

total_types = match_types = 0
for oline, cline in zip(open(sys.argv[1]), open(sys.argv[2])):
    if oline.strip() == cline.strip():
        match_types += 1
    total_types += 1

print('accuracy:', match_types/total_types)

    
