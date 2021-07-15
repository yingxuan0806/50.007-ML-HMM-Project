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

def Nviterbi(N, emission_dict, transition_dict, observations, sentence):
    n = len(sentence)
    smallest = -9999999

    # Set of states excluding START
    states = list(transition_dict.keys())
    states.remove("START")

    """"initialize score dict
      scores = { position: {
        state_v: {
          (state_u1, score),    
          (state_u2, score),  
          (state_u3, score)
        }
      } 
    """
    scores = {}

    # Base Cases: Do not need to account
    # Reason: pi(0 , START) = 1 otherwise 0
    # when we take log, they become 0.

    # START state to state 1
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

    scores_copy = copy.deepcopy(scores)
    
    # State 1 to n
    for i in range(1, n):
        scores[i] = {}
        scores_copy[i] = {}
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
                    v = "#UNK#"
                else:
                    v = sentence[i]

                # Emission Probability
                if ((v in emission_dict[state_v]) or (v == "#UNK#")): 
                    emis_frac = emission_para_token(emission_dict, v, state_v)
                    emis = math.log(emis_frac)
                else:
                    emis = smallest
              
                if i == 1 :
                  currentscore = scores[i-1][state_u][1] + trans + emis
                  findmax.append(currentscore)
                else:
                  currentscores = [[scores[i-1][state_u][m][1] for m in range(N)][j] + trans + emis for j in range(N)] # currentscores = [bestscore, 2nd bestscore, 3rd bestscore]
                  for score in currentscores:
                    findmax.append(score)
            # findmax = [bestscore, 2nd bestscore, 3rd bestscore,bestscore, 2nd bestscore, 3rd bestscore,bestscore, 2nd bestscore, 3rd bestscore,...,bestscore, 2nd bestscore, 3rd bestscore]  
            
            # ARGMAX
            # code to find N highest scores in findmax
            # since there are nT scores, we have to argmax over ALL these scores
            # ans is a list that holds the N highest scores
            # eg. ans = [highest, 2nd highest, 3rd highest, .... , N highest]
            # state_ans is a list that holds the N best states, with decrementing "bests state" from left to right
            # eg. state_ans = [best state, 2nd best state, ..... , N best state]
            ans = [] 
            state_ans = []
            findmax_copy = copy.deepcopy(findmax)
            for m in range(N):
                ans.append(max(findmax_copy))
                state_ans.append(states[findmax.index(ans[m]) // N])
                findmax_copy[findmax.index(ans[m])] = -999999999.999
            
            # store nested tuple of N best states ((best state, score),(2nd best state, score),(3rd best state, score))
            scores[i][state_v] = tuple((state_ans[m], ans[m]) for m in range(N))
            

    # STOP STATE
    scores[n] = {}
    scores_copy[n] = {}
    stopmax = []

    for state_u in states:
        # Transition Probability
        trans_frac = transition_para(transition_dict, state_u, "STOP")
        if trans_frac != 0:
            trans = math.log(trans_frac)
        else:
            trans = smallest

        stopscore = [[scores[n-1][state_u][m][1] for m in range(N)][j] + trans + emis for j in range(N)]
        for score in stopscore:
          stopmax.append(score)
    
    # ARGMAX
    # code to find n highest scores in stopmax
    # stop is a list that holds the N highest scores to stop
    # eg. stop = [highest, 2nd highest, 3rd highest, ... , N highest]
    # state_ans is a list that holds the best N bests state
    # state_ans = [best state, 2nd best state, ... , N , best state]
    stop = []
    state_ans = []
    stopmax_copy = copy.deepcopy(stopmax)
    for i in range(N):
        stop.append(max(stopmax_copy))
        state_ans.append(states[stopmax.index(stop[i]) // N])
        stopmax_copy[stopmax.index(stop[i])] = -999999999.999
    scores[n][state_u] = tuple((state_ans[m], stop[m]) for m in range(N))
    
      
    # Backtracking path
    # N_bestPaths is list of N lists, that holds N best paths in decreasing order.
    # eg. N_bestPaths = [best path, 2nd best path, ... , N best path]
    # lasts is a list to trac k the last state of each of the N best paths. 
    # eg. lasts = [best path's last, 2nd best path's last, ... , N best path's last]
    N_bestPaths = []
    lasts = [] 
    for i in range(N):
      path = ["STOP"]
      last = list(scores[n].values())[0][i][0]
      lasts.append(last)
      path.insert(0, last)
      N_bestPaths.append(path)
    
    for i in range(N):
        for k in range(n-1, -1, -1):
            if k == 0:
                last = scores[k][N_bestPaths[i][0]][0] 
            else:
                last = scores[k][N_bestPaths[i][0]][0][0]
            N_bestPaths[i].insert(0, last)
    
    
    return N_bestPaths[N-1]

if __name__ == '__main__':
    part4_dataset = ["EN"]

    for i in part4_dataset:
        """
        train = "Data/{folder}/train".format(folder = i)
        evaluation = "Data/{folder}/dev.in".format(folder = i)
        """

        train = root_dir + "{folder}/train".format(folder = i)
        evaluation = root_dir + "{folder}/dev.in".format(folder = i)
        
        # training
        transition_tracker = count_transition(train)

        obs_all, emission_tracker = count_emission(train)
        
        # evaluation
        with open(evaluation, "r") as f:
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
        # initialise N
        N = 3 
        # each line is a word
        for line in lines:        
            if line != "\n":
                line = line.strip()
                sentence.append(line)
            else:
                sentence_prediction = Nviterbi(N, emission_tracker, transition_tracker, obs_all, sentence)
                sentence_prediction.remove("START")
                sentence_prediction.remove("STOP")
                all_prediction = all_prediction + sentence_prediction
                all_prediction = all_prediction + ["\n"]
                sentence = []
        
        assert len(lines) == len(all_prediction)
        # create output file
        with open(root_dir + "{folder}/dev.p4.out".format(folder = i), "w") as g:
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