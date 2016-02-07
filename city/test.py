import json,re
import numpy
from numpy import *
from logisticCostFunction import costFunction
from gradient import grad
from predict import predict
from spellMistakes import spellMistakes

#location of test file
test_file_loc = "testAddress.json"

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

#loading ml parameters
theta = load("theta.npy")
mu = load("mu.npy")
sigma = load("sigma.npy")

#correct predictions
correct = 0
#wrong predictions
wrong = 0
#could not predict any city
neutral = 0
#no city is present in the address
not_present = 0

#creating regex for tier1-tier6 cities and storing it in regex_city_tierNo
#adding editions to tier1-tier6 cities and storing it in regex_city_edit_tierNo
for i in range(1,7) :
    var_edit = "regex_city_edit_"+str(i)
    var = "regex_city_"+str(i)
    vars()[var] = ""
    vars()[var_edit] = ""
    with open('tier_cities\\tier' + str(i) + 'cities.txt','r') as cities :
        for city in cities :
            city = re.sub(r"\(.*\)"," ",city)
            city_edit = ""
            city = re.sub("\n","",city)
            if city == "Nagar" :
                continue
            vars()[var] += r"\b" + city + r"\b|"
            if len(str(city)) <= 4 :
                continue
            for city_edit in spellMistakes(city) :
                if city_edit == "Nagar" or len(city_edit) <= 4:
                    continue
                vars()[var_edit] += r"\b" + city_edit + r"\b|"
    vars()[var] = vars()[var][:len(vars()[var])-1]
    vars()[var_edit] = vars()[var_edit][:len(vars()[var_edit])-1]

index = -1
#looping through all addresses and predicting
for dictionary in all_add :

    actual_city = dictionary["city"]
    add = dictionary["address"]

    #removing utf-16 encoding from the address
    add = re.sub(r"\\u...."," ",add)

    #ml features
    cities = [] #name of city
    tiers = []  #tier of city
    edits = []  #if any spelling mistake is present in the city name
    for tier in range(1,7) :
        if re.search(eval("regex_city_"+str(tier)),add,re.I) :
            for city_match in re.finditer(eval("regex_city_"+str(tier)),add,re.I) :
                cities.append(city_match.group())
                tiers.append(tier)
                edits.append(0)
        if re.search(eval("regex_city_edit_"+str(tier)),add,re.I) :
            for city_match in re.finditer(eval("regex_city_edit_"+str(tier)),add,re.I) :
                cities.append(city_match.group())
                tiers.append(tier)
                edits.append(1)
    correct_check = correct
    confidences = []

    #looping through all the cities found in the address
    for city_no,city in enumerate(cities) :
        index += 1

        #replacing certain symbols with space to get exact word count   
        symbol_free_add = re.sub(r",|;|\(|\)"," ",add)
        total_words = len(symbol_free_add.split())

        #finding loacation of city(position) in the address
        split_phrase = re.split(r"(\b" + city + r"\b)",symbol_free_add)
        for i,phrase in enumerate(split_phrase[::-1]) :
            if phrase == city :
                city_index = len(split_phrase)-i-1
                break
        phrase_before_city =  ' '.join(split_phrase[:city_index])
        city_pos = len(phrase_before_city.split()) + 1

        #appending the features to the dataset
        X[index,0] = float(city_pos)/total_words 
        X[index,1] = edits[city_no]
        X[index,2] = tiers[city_no]
        this_regX = (X[index,:] - mu)/sigma
        if re.search(actual_city,city,re.I) :
            y[index,0] = 1
        else :
            y[index,0] = 0

        #finding the confidence of this city being in the current address
        confidence = predict(this_regX,theta)
        confidences.append(confidence)

    #checking if the actual city is present in the address    
    if not re.search(actual_city,add,re.I) :
        not_present += 1
        continue

    #checking if any city has been found in the address
    if len(confidences) == 0 :
        neutral += 1
        continue

    #predicting the city for the current address
    max_confidence = max(confidences)
    max_confidence_index = [ position for position,conf in enumerate(confidences) if conf == max_confidence ]
    if re.search(actual_city,cities[max_confidence_index[0]],re.I) or re.search(cities[max_confidence_index[0]],actual_city,re.I) :
        correct += 1
    else :
        wrong += 1

print correct,wrong,neutral,not_present


