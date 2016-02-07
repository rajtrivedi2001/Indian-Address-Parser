import json,re
from numpy import *
from logisticCostFunction import costFunction
from gradient import grad
from spellMistakes import spellMistakes

#creating regex for state names
for i in range(1,3) :
    var_edit = "regex_state_edit_"+str(i)
    var = "regex_state_"+str(i)
    vars()[var] = ""
    vars()[var_edit] = ""
    with open('states_'+str(i)+'.txt','r') as states :
        for state in states :
            state = re.sub(r"\(.*\)"," ",state)
            state_edit = ""
            state = re.sub("\n","",state)
            vars()[var] += r"\b" + state + r"\b|"
            if len(str(state)) <= 4 :
                continue
            for state_edit in spellMistakes(state) :
                if state_edit == "Nagar" or len(state_edit) <= 4:
                    continue
                vars()[var_edit] += r"\b" + state_edit + r"\b|"
    vars()[var] = vars()[var][:len(vars()[var])-1]
    vars()[var_edit] = vars()[var_edit][:len(vars()[var_edit])-1]


#location of test file
test_file_loc = "trainAddress.json"

with open(test_file_loc,'r') as file :
    all_add = json.load(file)

#declaring ml variables
#maximum no of predictions made for a given address
no_of_predictions = 5
#number of features
n = 3
#length of dataset
m = no_of_predictions*len(all_add)
#extracted features
X = mat(zeros((m,n)))
#output
y = mat(zeros((m,1)))

index = -1 
#looping through all addresses and predicting
for dictionary in all_add :

    actual_state = dictionary["state"]
    add = dictionary["address"]

    #removing utf-16 encoding from the address
    add = re.sub(r"\\u...."," ",add)

    #ml features
    states = [] #name of state
    tiers = []  #full state name - 1, short state name - 0
    edits = []  #if any spelling mistake is present in the city name

    for tier in range(1,3) :
        if re.search(eval("regex_state_"+str(tier)),add,re.I) :
            for state_match in re.finditer(eval("regex_state_"+str(tier)),add,re.I) :
                states.append(state_match.group())
                tiers.append(tier)
                edits.append(0)

        #because the regex_state_edit_2 is empty as
        if tier == 2 :
            break

        if re.search(eval("regex_state_edit_"+str(tier)),add,re.I) :
            for state_match in re.finditer(eval("regex_state_edit_"+str(tier)),add,re.I) :
                states.append(state_match.group())
                tiers.append(tier)
                edits.append(1)

    #looping through all the states found in the address
    for state_no,state in enumerate(states) :
        index += 1

        #replacing certain symbols with space to get exact word count   
        symbol_free_add = re.sub(r",|;|\(|\)"," ",add)
        total_words = len(symbol_free_add.split())

        #finding loacation of state(position) in the address
        split_phrase = re.split(r"(\b" + state + r"\b)",symbol_free_add)
        for i,phrase in enumerate(split_phrase[::-1]) :
            if phrase == state :
                state_index = len(split_phrase)-i-1
                break
        phrase_before_state =  ' '.join(split_phrase[:state_index])
        state_pos = len(phrase_before_state.split()) + 1

        #appending the features to the dataset
        X[index,0] = float(state_pos)/total_words 
        X[index,1] = edits[state_no]
        X[index,2] = tiers[state_no]

        if re.search(actual_state,state,re.I) :
            y[index,0] = 1
        else :
            y[index,0] = 0

#machine learning code
#training with the data set

#number of iteration to carry out
num_iters = 100

#choosing the parameter alpha
alpha = 1

#the cost funciton J_val
J_val = mat(zeros((num_iters,1)))

#assigning random values to theta
theta = mat(random.rand(n+1,1))

#deleting unused rows in X and y
X = X[:index+1,:]
y = y[:index+1,:]

for itr in range(num_iters) :
    J_val[itr,0] = costFunction(theta,X,y)
    gradient = grad(theta,X,y)
    theta = theta - (alpha*gradient) 

#saving required ml parameters for testing   
save("theta",theta)

    
    
