
ifile = open("holdout.txt", "r")
A = []
B = []
C = []
D = []
E = []
for i, line in enumerate(ifile):
    if i == 0:
        limit = int(line)
    elif i < 1 + limit:
        A.append(line.strip())
    elif i < 1 + 2 * limit:
        B.append(line.strip())
    elif i < 1 + 3 * limit:
        C.append(line.strip())
    elif i < 1 + 4 * limit:
        D.append(line.strip())
    elif i < 1 + 5 * limit:
        E.append(line.strip())

print(A)
print(B)
print(C)
print(D)
print(E)
        
ofile = open("testheld.txt", "w+")
for v in A:
    for w in B:
        ofile.write("dog {} {}\n".format(v, w))
    for w in C:
        ofile.write("dog {} {}\n".format(v, w))
    for w in D:
        ofile.write("dog {} {}\n".format(v, w))
    for w in E:
        ofile.write("dog {} {}\n".format(v, w))
for v in B:
    for w in C:
        ofile.write("dog {} {}\n".format(v, w))
    for w in D:
        ofile.write("dog {} {}\n".format(v, w))
    for w in E:
        ofile.write("dog {} {}\n".format(v, w))
for v in C:
    for w in D:
        ofile.write("dog {} {}\n".format(v, w))
    for w in E:
        ofile.write("dog {} {}\n".format(v, w))
for v in D:
    for w in E:
        ofile.write("dog {} {}\n".format(v, w))
ofile.close()
