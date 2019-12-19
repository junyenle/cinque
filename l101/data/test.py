ifile = open("data.txt", "r")

max = 0
for line in ifile:
    adjectives = line.split()[1:]
    leng = len(adjectives)
    if leng > max:
        max = leng
        print(adjectives)
        
print(max)