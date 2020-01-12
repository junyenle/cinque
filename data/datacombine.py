ofile = open("datanown.txt", "w+")

for i in range(0, 25):
    j = i + 1
    print(j)
    ifile = open("out" + str(j) + ".txt", "r")
    for line in ifile:
        ofile.write(line)
    ifile.close()