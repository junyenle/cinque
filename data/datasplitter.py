ifile = open("datawn.txt", "r")
train = open("train.txt", "w+")
test = open("test.txt", "w+")
valid = open("valid.txt", "w+")

for i, line in enumerate(ifile):
    if i % 10 == 0:
        test.write(line)
    elif i % 9 == 0:
        valid.write(line)
    else:
        train.write(line)
    