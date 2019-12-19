ifile = open("data.txt", "r")
train = open("train.txt", "w+")
test = open("test.txt", "w+")

for i, line in enumerate(ifile):
    if i % 10 == 0:
        test.write(line)
    else:
        train.write(line)
    