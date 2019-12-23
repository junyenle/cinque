NEGATIVE = -1
POSITIVE = 1
class PE:
    def __init__(self, element, direction):
        assert direction in (NEGATIVE, POSITIVE)
        self.element = element
        self.direction = direction

def get_permutations(list):
    numelem = len(list)
    permutation = [PE(element, NEGATIVE) for element in range(0, numelem)]
    toret = []
    for n in count(1):
        permlist = []
        for e in permutation:
            permlist.append(list[e.element])
        toret.append(permlist)
        mobile = None
        for i, e in enumerate(permutation):
            j = i + e.direction
            if (0 <= j < len(permutation)
                and e.element > permutation[j].element
                and (mobile is None or e.element > mobile_element)):
                mobile = i
                mobile_element = permutation[mobile].element
        if mobile is None:
            break
        for e in permutation:
            if e.element > mobile_element:
                e.direction *= -1
        i = mobile
        j = i + permutation[i].direction
        assert 0 <= j < len(permutation)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return toret