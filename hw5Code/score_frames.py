import frames
import sys

if len(sys.argv) != 3:
    print('usage: score_frames.py <output> <correct>', file=sys.stderr)
    exit(1)

total_frames = match_frames = match_types = 0
output_args = correct_args = match_args = 0
for oline, cline in zip(open(sys.argv[1]), open(sys.argv[2])):
    oframe = frames.Frame.from_str(oline)
    cframe = frames.Frame.from_str(cline)
    if oframe == cframe:
        match_frames += 1
    if oframe.type == cframe.type:
        match_types += 1
    total_frames += 1
    oargs = set(oframe.args)
    cargs = set(cframe.args)
    match_args += len(oargs & cargs)
    output_args += len(oargs)
    correct_args += len(cargs)

print('exact match:        ', match_frames/total_frames)
print('frame type accuracy:', match_types/total_frames)
precision = match_args / output_args
recall = match_args / correct_args
print('argument F1:        ', 1/((1/precision+1/recall)/2))

    
