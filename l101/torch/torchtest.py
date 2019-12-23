import torch
# from utils import PE, get_permutations
torch.manual_seed(0)

def save_model(filename, model):
    file = open(filename, "wb")
    pickle.dump(model, file)
    file.close()
    
def load_model(filename):
    file = open(filename, "rb")
    model = pickle.load(file)
    file.close()
    return model

def get_training_example(line, reverse):    
    """ input: string line from input file
        output: adj1 embedding (1 x 300), adj2 embedding (1 x 300), correct output (scalar) """
    linearr = line.split()
    if reverse:
        adj1arr = linearr[301:]
        adj2arr = linearr[1:301]
    else:
        adj1arr = linearr[1:301]
        adj2arr = linearr[301:]
    adj1floats = []
    adj2floats = []
    for item in adj1arr:
        adj1floats.append(float(item))
    for item in adj2arr:
        adj2floats.append(float(item))
    y = []
    if reverse:
        y.append(0)
    else:
        y.append(1)       
    return torch.FloatTensor(adj1floats), torch.FloatTensor(adj2floats), torch.FloatTensor(y)

DEBUGPRINTINTERVAL = 1000 # how often to print where we are
DEBUGLIMIT = 200000 # number of inputs to consider
LR = 0.01 # learning rate
TRAIN = True
SAVEMODEL = True
MODELFILE = "wmodel"
TEST = False

# initialize weights to random numbers
w = torch.randn((300, 300), requires_grad=True)

if TRAIN:
    tfile = open("/mnt/d/trainvecs.txt", "r")
    for ti, line in enumerate(tfile):
        if ti == DEBUGLIMIT:
            break
        if ti % DEBUGPRINTINTERVAL == 0:
            print("ti = {}".format(ti))
        
        for value in [True, False]:
            adj1, adj2, y_obs = get_training_example(line, value)
            y_pred = torch.matmul(torch.matmul(torch.t(adj1), w), adj2)
            mse = torch.mean((y_pred - y_obs) ** 2)
            mse.backward()
            with torch.no_grad():
                w = w - LR * w.grad
            w.requires_grad = True
    if SAVEMODEL:
        save_model(MODELFILE, w)
            
if TEST:
    w = load_model(MODELFILE)
    w.requires_grad = False
    # testing file
    tfile = open("/mnt/d/testvecs.txt", "r")
    for ti, line in enumerate(tfile):
        if ti == DEBUGLIMIT:
            break
        if ti % DEBUGPRINTINTERVAL == 0:
            print("ti = {}".format(ti))
        for value in [True, False]:
            adj1, adj2, y_obs = get_training_example(line, value)
            y_pred = torch.matmul(torch.matmul(torch.t(adj1), w), adj2)
            mse = torch.mean((y_pred - y_obs) ** 2)
            print("correct: {}, predicted: {}".format(y_obs, y_pred))