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
    
    # actually, do i even need this. maybe can remove. KIV. its more for convenience but what if i forget to update this object
    # counts = a dictionary of key = state y, value = count of state y
    counts_y = {i: sum(emission_tracker[i].values()) for i in emission_tracker}
    return obs_all, emission_tracker

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

# Function to tabularise the transition parameters
def count_transition(file):
    with open(file, "r", encoding="cp437", errors='ignore') as f:
        # readlines() returns a list containing each line in the file as a list item
        lines = f.readlines()
    
    start_state = "START"
    stop_state = "STOP"
    
    # initialise
    state_u = "START"
    
    # initialise state (u,v) transition tracker
    # key: state u (ie. previous_state)
    # value: nested dictionary of key = state v, value = frequency of state u to v
    transition_tracker = {}
    
    for line in lines:
        # split the observation and its tag
        split_line = line.strip()  # remove leading and trailing characters
        split_line = split_line.rsplit(" ")  # split a string into list
        
        # case 1: word line
        if len(split_line) == 2:
            obs = split_line[0]
            state_v = split_line[1]

            # track the current line
            # the state of the current line is state_v
            # get the specific nested dictionary of the previous state
            
            # if the specific nested dictionary of the previous state does not exist yet, create new
            if state_u not in transition_tracker:
                state_u_dict = {}
            else:
                # get the specific nested dictionary of the previous state
                state_u_dict = transition_tracker[state_u]
            
            # if key = state v already exists in this specific nested dictionary
            if state_v in state_u_dict:
                state_u_dict[state_v] += 1
            else:
                state_u_dict[state_v] = 1
                
            # update to the overall transition tracker
            transition_tracker[state_u] = state_u_dict
            
            # update the current state to become the previous state for the next line in the dataset    
            state_u = state_v
            
            
        # case 2: empty line: indicates end of current sentence, and the start of next sentence
        if len(split_line) != 2:
            # get the specific nested dictionary of the current sentence that just ended
            state_u_dict = transition_tracker[state_u]
            # set state of current sentence to stop state
            state_v = stop_state
            
            # if key = state v already exists in this specific nested dictionary
            if state_v in state_u_dict:
                state_u_dict[state_v] += 1
            else:
                state_u_dict[state_v] = 1
                
            # update to the overall transition tracker
            transition_tracker[state_u] = state_u_dict
            
            # reset state to start for the next sentence
            state_u = start_state
    
    return transition_tracker

def transition_para(transition_tracker, state_u, state_v):
    # transition: state u to state v
    # obtain the specific nested dict of state_u
    
    # if no count from training set, return zero
    if state_u not in transition_tracker:
        fraction = 0
        
    else:
        state_u_dict = transition_tracker[state_u]
    
        # numerator, 0 if not present
        numerator = state_u_dict.get(state_v, 0)
    
        # denominator
        # sum up all the counts of the specific state_u_dict
        denominator = sum(state_u_dict.values())
        fraction = numerator / denominator
    
    return fraction

def viterbi(emission_dict, transition_dict, observations, sentence):
    n = len(sentence)
    smallest = -9999

    # Set of states excluding START
    states = list(transition_dict.keys())
    states.remove("START")

    # initialize score dict
    """"
      scores = { position: {
        state_v: {
          (state_u, score)
        }
      } 
    """
    scores = {}

    # START STATE to State 1 
    # pi(position=1, state1)
    scores[0] = {}
    for state_v in states:
        # Transition Probability
        trans_frac = transition_para(transition_dict, "START", state_v)
        if trans_frac != 0:
            trans = math.log(trans_frac)
        else:
            trans = smallest
        
        # if the word does not exist, assign special token
        if sentence[0] not in observations:
            obs = "#UNK#"
        else:
            obs = sentence[0]

        # Emission Probability
        if ((obs in emission_dict[state_v]) or (obs == "#UNK#")): 
            emis_frac = emission_para_token(emission_dict, obs, state_v)
            emis = math.log(emis_frac)
        else:
            emis = smallest
        
        start = trans + emis
        scores[0][state_v] = ("START", start)
    
    
    # State 1 to n
    # pi(position=2, score) -> pi(position=n-1, score)
    for i in range(1, n):
        scores[i] = {}
        for state_v in states:
            findmax = []
            for state_u in states:
                # Transition Probability
                trans_frac = transition_para(transition_dict, state_u, state_v)
                if trans_frac != 0:
                    trans = math.log(trans_frac)
                else:
                    trans = smallest
                
                # if the word does not exist, assign special token
                if sentence[i] not in observations:
                    obs = "#UNK#"
                else:
                    obs = sentence[i]

                # Emission Probability
                if ((obs in emission_dict[state_v]) or (obs == "#UNK#")): 
                    emis_frac = emission_para_token(emission_dict, obs, state_v)
                    emis = math.log(emis_frac)
                else:
                    emis = smallest
                
                currentscore = scores[i-1][state_u][1] + trans + emis
                findmax.append(currentscore)
    
            # ARGMAX
            ans = max(findmax)
            state_ans = states[findmax.index(ans)]

            scores[i][state_v] = (state_ans, ans)
    
    # STATE N to Stop State
    scores[n] = {}
    stopmax = []
    for state_u in states:
        # Transition Probability
        trans_frac = transition_para(transition_dict, state_u, "STOP")
        if trans_frac != 0:
            trans = math.log(trans_frac)
        else:
            trans = smallest
        
        stopscore = scores[n-1][state_u][1] + trans
        stopmax.append(stopscore)
    
    # ARGMAX
    stop = max(stopmax)
    state_ans = states[stopmax.index(stop)]

    scores[n] = (state_ans, stop)

    # Backtracking path
    path = ["STOP"]
    # scores[n] = ('O', -308.32462005568965)
    last = scores[n][0]
    path.insert(0, last)
    
    for k in range(n-1, -1, -1):
        """
        scores[k=n-1] = {'B-NP': ('I-NP', -10000306.577758126), 
                     'I-NP': ('I-NP', -313.7020929214364), 
                     'B-VP': ('I-NP', -10000305.536915062), 
                     'B-ADVP': ('I-NP', -10000307.71158532), 
                     'B-ADJP': ('I-NP', -10000309.02976334), 
                     'I-ADJP': ('B-ADJP', -10000310.31315982), 
                     'B-PP': ('I-NP', -10000305.38842483), 
                     'O': ('I-NP', -307.17979888643373),  <-- scores[k][last]
                     'B-SBAR': ('I-NP', -10000308.589206912), 
                     'I-VP': ('B-VP', -20000298.54567737), 
                     'I-ADVP': ('I-NP', -20000302.53378508), 
                     'B-PRT': ('I-NP', -10000312.495499242), 
                     'I-PP': ('I-NP', -20000302.53378508), 
                     'B-CONJP': ('I-NP', -10000312.043514118), 
                     'I-CONJP': ('I-NP', -20000302.53378508), 
                     'B-INTJ': ('I-NP', -20000302.53378508), 
                     'I-INTJ': ('I-NP', -20000302.53378508), 
                     'I-SBAR': ('I-NP', -20000302.53378508), 
                     'B-UCP': ('B-NP', -10000320.498189736), 
                     'I-UCP': ('I-NP', -20000302.53378508), 
                     'B-LST': ('I-NP', -20000302.53378508)}

        """
        last = scores[k][last][0] # I-NP
        path.insert(0, last)
    return path


if __name__ == '__main__':
    datasets = ["EN", "CN", "SG"]

    for i in datasets:
        train = root_dir + "{folder}/train".format(folder = i)
        evaluation = root_dir + "{folder}/dev.in".format(folder = i)
        
        # training
        transition_tracker = count_transition(train)
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
                sentence_prediction = viterbi(emission_tracker, transition_tracker, obs_all, sentence)
                sentence_prediction.remove("START")
                sentence_prediction.remove("STOP")
                all_prediction = all_prediction + sentence_prediction
                all_prediction = all_prediction + ["\n"]
                sentence = []
        
        assert len(lines) == len(all_prediction)
        print("All words have a tag. Proceeding..")

        # create output file
        with open(root_dir + "{folder}/dev.p3.out".format(folder = i), "w", encoding="cp437", errors='ignore') as g:
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