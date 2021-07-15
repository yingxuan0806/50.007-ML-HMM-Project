# Libraries
import sys
import os
import math
import copy

root_dir = "./"

# Function to tabularise the emission parameters
def count_emission(file):
    with open(file, "r", encoding="cp437", errors='ignore') as f:
        # readlines() returns a list containing each line in the file as a list item
        lines = f.readlines()
        
    # Track set of unique observations
    obs_all = set()
    
    # track emission count
    # key: state y, value: nested dictionary with key = obs x and value = frequency of this specific obs x
    emission_tracker = {}
    
    for line in lines:
        # split the observation and its tag
        split_line = line.strip()  # remove leading and trailing characters
        split_line = split_line.rsplit(" ")  # split a string into list

        # there are lines in the file that is just an empty line, skip these lines
        if len(split_line) == 2:
            obs = split_line[0]
            state = split_line[1]        
            obs_all.add(obs)
    
            # track the current emission
            # purpose: to update nested dictionary of specific state y in emission_tracker dict
            if state in emission_tracker:
                # get the correct nested dictionary if this state y already exists
                curr_emi = emission_tracker[state]
            else:
                # this state y does not exist yet. create new
                curr_emi = {}
                
            # update frequency of this specific obs x emitted from this specific state y
            if obs in curr_emi:
                curr_emi[obs] = curr_emi[obs] + 1
            else:
                curr_emi[obs] = 1
            
            # update nested dictionary of specific state y to overall emission tracker
            emission_tracker[state] = curr_emi
    
    # counts = a dictionary of key = state y, value = count of state y
    counts_y = {i: sum(emission_tracker[i].values()) for i in emission_tracker}
    return obs_all, emission_tracker

def emission_para(emission_tracker, obs_x, state_y):
    
    # obtain the specific nested dict of state y
    state_dict = emission_tracker[state_y]
    
    # get the value of specific state y -> obs x
    numerator = state_dict.get(obs_x, 0)
    
    # get total counts of state y
    denominator = sum(state_dict.values())
    
    return numerator / denominator

def emission_para_token(emission_tracker, obs_x, state_y, k = 0.5):
    # obtain the specific nested dict of state y
    state_dict = emission_tracker[state_y]
    
    denominator = sum(state_dict.values()) + k
    
    # word token x appears in training set
    if obs_x != "#UNK#":
        numerator = state_dict[obs_x]
    # word token x is special token
    else: 
        numerator = k
    
    return numerator / denominator

# tag prediction for a sentence (list input with string type elements)
def tag_producer(emission_tracker, sentence, obs_all):
    tag_output = []
    
    for i in sentence:
        predicted_state = ""
        highest_prob = -9999999.0
        
        # loop through all states to determine emission prob of each state, then return the highest one
        for state_y in emission_tracker:
            # if the word does not exist, assign special token
            if i not in obs_all:
                i = "#UNK#"
                
            if ((i in emission_tracker[state_y]) or (i == "#UNK#")):
                # emission probabilities here are calculated using estimator with special token
                emission_prob = emission_para_token(emission_tracker, i, state_y, 0.5)

                if emission_prob > highest_prob:
                    highest_prob = emission_prob
                    predicted_state = state_y
                    
        tag_output.append(predicted_state)
    return tag_output

if __name__ == '__main__':
    datasets = ["EN", "CN", "SG"]

    for i in datasets:
        train = root_dir + "{folder}/train".format(folder = i)
        evaluation = root_dir + "{folder}/dev.in".format(folder = i)
        
        # training
        obs_all, emission_tracker = count_emission(train)
        
        # evaluation
        with open(evaluation, "r", encoding="cp437", errors='ignore') as f:
            # readlines() returns a list containing each line in the file as a list item
            # each line is a word
            lines = f.readlines()
        
        # track each sentence's prediction labels
        # each word's prediction label will be an element of this list
        sentence = []
        
        # list containing all prediction labels
        # sentences are separated with element "\n" in between
        all_prediction = []

        print(i)

        # each line is a word
        for line in lines:        
            if line != "\n":
                line = line.strip()
                sentence.append(line)
            else:
                sentence_prediction = tag_producer(emission_tracker, sentence, obs_all)
                all_prediction = all_prediction + sentence_prediction
                all_prediction = all_prediction + ["\n"]
                sentence = []
        
        # create output file
        with open(root_dir + "{folder}/dev.p2.out".format(folder = i), "w",  encoding="cp437", errors='ignore') as g:
            
            for j in range(len(lines)):
                word = lines[j].strip()
                
                if word != "\n":
                    tag = all_prediction[j]
                    if(tag != "\n"):
                        g.write(word + " " + tag)
                        g.write("\n")
                    else:
                        g.write("\n")
                

    print("done")
