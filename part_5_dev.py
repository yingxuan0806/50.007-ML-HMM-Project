# Libraries
import sys
import os
import math
import copy
from operator import itemgetter

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

def HMM2_count_transition(file):
  with open(file, 'r') as f:
    lines = f.readlines()

  start_state = 'START'
  stop_state = 'STOP'

  # initialise
  state_u = 'START'
  state_v = 'START' # How do I handle START -> START -> first word? 
                    # Maybe implement a check at the end, and remove the START-> START when calculating transition prob.

  ''' 
  initialise state(u,v,w) transition tracker
  key state u (ie. 2nd previous state) 
  value: nested dictionary of key = state v,  value = nested dictionary of key = state w, value = freq of u to v to w
  
  transition tracker = {
    state_u : {
      state_v : {
        state_w : freq of u->v->w
      }
    }
  } 
  '''
  
  transition_tracker = {}
  

  for line in lines:
    # split the obs and its tag.
    split_line = line.strip() # removes trailing characters
    split_line = split_line.rsplit(' ') # splits a string into a list
    
    # case 1: word line
    if len(split_line) == 2:
      obs = split_line[0]
      state_w = split_line[1]

      # track the current line
      # the state of the current line is state_w
      # get the specific nested dictionary of the prev state

      # if the specified nested dictionary of prev state does not yet exist, create new dict
      if state_u not in transition_tracker:
        transition_tracker[state_u] = {}
      
      if state_v not in transition_tracker[state_u]:
        state_v_dict = {}
      else:
        # get the specifed 'double' nested dictionary of the 2 prev states
        state_v_dict = transition_tracker[state_u][state_v]
          
      # if key = state w already exists in this specifed nested dictionary
      if state_w in state_v_dict:
        state_v_dict[state_w] += 1
      else:
        state_v_dict[state_w] = 1

      # update overall transition tracker
      transition_tracker[state_u][state_v] = state_v_dict

      # update the current state, prev state to become the prev state, 2nd prev state for the next line in the dataset
      state_u = state_v
      state_v = state_w

    # case 2: empty line; indicates end of current sentence, and the start of next sequence
    if len(split_line) != 2:

      if state_u not in transition_tracker:
        transition_tracker[state_u] = {}
      
      if state_v not in transition_tracker[state_u]:
        state_v_dict = {}
      else:
        # get the specified nested dictionary of the current sentence that just ended
        state_v_dict = transition_tracker[state_u][state_v]

      # set state of current sentence to stop state
      state_w = stop_state

      # if key = state w already exists in this specific 'double' nested dictionary
      if state_w in state_v_dict:
        state_v_dict[state_w] += 1
      else:
        state_v_dict[state_w] = 1


      # update the overall transition tracker
      transition_tracker[state_u][state_v] = state_v_dict 

      # reset state to start for next sentence
      state_u = start_state
      state_v = start_state

  return transition_tracker

def HMM2_transition_para(transition_tracker, state_u, state_v, state_w):
  # transition: state_u -> state_v -> state_w 
  # obtain the specified 'double' nested dict of state_v
  
  # if no count from training set, return zero
  if state_u not in transition_tracker:
    fraction = 0
  elif state_v not in transition_tracker[state_u]:
    fraction = 0
  else:
    state_v_dict = transition_tracker[state_u][state_v]

    # numerator, 0 if not present
    numerator = state_v_dict.get(state_w, 0)

    # denominator
    # sum up all counts of specified state_v_dict
    denominator = sum(state_v_dict.values())
    fraction = numerator / denominator

  return fraction

# Viterbi: 2nd Order-ed HMM 
def HMM2_viterbi(emission_dict, transition_dict, observations, sentence):
  n = len(sentence)
  smallest = -9999
  TOKEN = "#UNK#"

  # set of states excluding START
  states = list(transition_dict.keys())
  states.remove('START')
  
  scores = {}

  """ 
  initialise score dict
    scores = {
      position : {
        state_w : {
          state_u : {
            state_v : score
          }
        }
      }
    }
  """

  # Base Cases: Do not need to account
  # Reason: pi(-1, START) = 1 otherwise 0
  #         pi(0 , START) = 1 otherwise 0
  # when we take log, they become 0. 


  # POSITION 0
  # (START -> START -> STATE W)
  scores[0] = {}

  for state_w in states:
    # Transition Probability
    trans_frac = HMM2_transition_para(transition_dict, 'START', 'START', state_w)

    if trans_frac != 0:
      trans = math.log(trans_frac)
    else:
      trans = smallest
    
    # if the word does not exist, assign the special token
    if sentence[0] not in observations:
      obs = TOKEN
    else:
      obs = sentence[0]

    # Emission Probability
    if ((obs in emission_dict[state_w]) or (obs == TOKEN)):
      emis_frac = emission_para_token(emission_dict, obs, state_w)
      emis = math.log(emis_frac)
    else:
      emis = smallest

    start_score = trans + emis
    scores[0][state_w] = {}
    scores[0][state_w]['START'] = {"START": start_score}

  # Case: 1 Word Sentence
  if n == 1:
    # POSITION 0 to STOP STATE
    # (START -> POSITION 0 -> STOP)
    scores[n] = {}
    scores[n]["STOP"] = {}

    scores[n]["STOP"]["START"] = {}
    for state_v in states:
      # Transition Probability
      trans_frac = HMM2_transition_para(transition_dict, "START", state_v, "STOP")
      if trans_frac != 0:
        trans = math.log(trans_frac)
      else:
        trans = smallest
        
      best_score_v = scores[0][state_v]["START"]["START"]
      current_stop_score = best_score_v + trans
      scores[n]["STOP"]["START"][state_v] = current_stop_score
    
    # Backtracking path
    path = ["STOP"]
    stop_lst = []

    for state_v in scores[n]["STOP"]["START"]:
      stop_lst.append((state_v, scores[n]["STOP"]["START"][state_v]))
    
    max_state_v_for_start_state = max(stop_lst, key=itemgetter(1))
    
    # Insert state v
    path.insert(0, max_state_v_for_start_state[0])
    # Insert start state
    path.insert(0, "START")

    # Return [START, state_v, STOP]
    return path

  # For n > 1
  # Position 1
  # START -> STATE V -> STATE W
  else:
    scores[1] = {}

    for state_w in states:
      scores[1][state_w] = {}
      scores[1][state_w]["START"] = {}
      for state_v in states:
        # Transition Probability
        trans_frac = HMM2_transition_para(transition_dict, 'START', state_v, state_w)

        if trans_frac != 0:
          trans = math.log(trans_frac)
        else:
          trans = smallest
        
        # if the word does not exist, assign the special token
        if sentence[1] not in observations:
          obs = TOKEN
        else:
          obs = sentence[1]

        # Emission Probability
        if ((obs in emission_dict[state_w]) or (obs == TOKEN)):
          emis_frac = emission_para_token(emission_dict, obs, state_w)
          emis = math.log(emis_frac)
        else:
          emis = smallest

        # Previous position: START -> START -> STATE W(Now STATE V)
        current_score = scores[0][state_v]['START']['START'] + trans + emis
        scores[1][state_w]['START'][state_v] = current_score

  if n == 2:
    # Sentence has two letters
    # state 2 to 'STOP' state
    scores[2] = {}
    scores[2]["STOP"] = {}

    for state_u in states:
      scores[n]["STOP"][state_u] = {}
      for state_v in states:
        # Transition Probability
        trans_frac = HMM2_transition_para(transition_dict, state_u, state_v, 'STOP')
        if trans_frac != 0:
          trans = math.log(trans_frac)
        else:
          trans = smallest

        state_v_arr = []
        
        # compute the current best score of state_u -> state_v -> STOP over all previous "OLD STATE U"
        for old_state_u in scores[n-1][state_v]:
          state_v_arr.append(scores[n-1][state_v][old_state_u][state_u])
        
        best_score_v = max(state_v_arr)
        current_stop_score = best_score_v + trans
        scores[n]["STOP"][state_u][state_v] = current_stop_score

    # Backtracking path
    path = ["STOP"]
    stop_lst = []

    for state_u in scores[n]["STOP"]:
      state_u_lst = []
      for state_v in scores[n]["STOP"][state_u]:
        state_u_lst.append((state_v, scores[n]["STOP"][state_u][state_v]))
      
      # Finding best score for a specific STATE U among all possible STATE V
      max_state_v_for_state_u = max(state_u_lst, key=itemgetter(1))
      
      # Tuple of (state u, state v, score)
      max_tuple = (state_u, max_state_v_for_state_u[0], max_state_v_for_state_u[1])
      stop_lst.append(max_tuple)

    # maximum score for among all previous STATE U and all previous STATE V
    max_stop_tuple = max(stop_lst, key=itemgetter(2))
    
    # Insert state v
    path.insert(0, max_stop_tuple[1])
    # Insert state u
    path.insert(0, max_stop_tuple[0])

    # To refer to the second last STATE
    prev = -2

    for k in range(n-1, 0, -1):
      state_u = scores[k][path[prev]] # k = 1 always
      state_v_lst = []
      
      # Find all STATE U that results in the specific STATE V pointed by prev
      for i in state_u.keys():
        if path[prev-1] in state_u[i]:
          state_v_lst.append((i, state_u[i][path[prev-1]]))
      
      # Obtain the ARGMAX of STATE U -> STATE V pointed by prev
      max_score = max(state_v_lst, key=itemgetter(1))
      
      # Shift STATE being looked at by -1
      prev = prev - 1
      
      # Insert max STATE U into PATH
      path.insert(0, max_score[0])
  
  # For n > 2
  # Position 2
  # STATE U -> STATE V -> STATE W
  # However, OLD STATE U in position 1 is "START"
  elif n > 2:
    scores[2] = {}
    for state_w in states:
      scores[2][state_w] = {}
      for state_u in states:
        scores[2][state_w][state_u] = {}
        for state_v in states:
          # Transition Probability
          trans_frac = HMM2_transition_para(transition_dict, state_u, state_v, state_w)

          if trans_frac != 0:
            trans = math.log(trans_frac)
          else:
            trans = smallest
          
          # if the word does not exist, assign the special token
          if sentence[2] not in observations:
            obs = TOKEN
          else:
            obs = sentence[2]

          # Emission Probability
          if ((obs in emission_dict[state_w]) or (obs == TOKEN)):
            emis_frac = emission_para_token(emission_dict, obs, state_w)
            emis = math.log(emis_frac)
          else:
            emis = smallest

          # Previous position: START -> STATE V(Now STATE U) -> STATE W(Now STATE V)
          current_score = scores[1][state_v]['START'][state_u] + trans + emis
          scores[2][state_w][state_u][state_v] = current_score


    # Position 3 to n
    # STATE U -> STATE V -> STATE W
    # OLD STATE U in position 1 will never be "START"
    for i in range(3,n):
      scores[i] = {}
      for state_w in states:
        scores[i][state_w] = {}
        for state_u in states:
          scores[i][state_w][state_u] = {}
          for state_v in states:
            # Transition Probability
            trans_frac = HMM2_transition_para(transition_dict, state_u, state_v, state_w)
            if trans_frac != 0:
              trans = math.log(trans_frac)
            else:
              trans = smallest

            # if the word does not exist, assign the special token
            if sentence[i] not in observations:
              obs = TOKEN
            else:
              obs = sentence[i]

            # Emission Probability
            if ((obs in emission_dict[state_w]) or (obs == TOKEN)):
              emis_frac = emission_para_token(emission_dict, obs, state_w)
              emis = math.log(emis_frac)
            else:
              emis = smallest

            # compute the current best score of state_u -> state_v -> state_w over all previous "OLD STATE U"
            state_v_arr = []
            for old_state_u in scores[i-1][state_v]:
                state_v_arr.append(scores[i-1][state_v][old_state_u][state_u])
            best_score_v = max(state_v_arr)

            current_score = best_score_v + trans + emis
            scores[i][state_w][state_u][state_v] = current_score

    # POSITION n+1 (outside the sentence)
    # Mainly transition probability to STOP state
    scores[n] = {}
    scores[n]["STOP"] = {}

    for state_u in states:
      scores[n]["STOP"][state_u] = {}
      for state_v in states:
        # Transition Probability
        trans_frac = HMM2_transition_para(transition_dict, state_u, state_v, 'STOP')
        if trans_frac != 0:
          trans = math.log(trans_frac)
        else:
          trans = smallest
        
        # compute the current best score of state_u -> state_v -> STOP over all previous "OLD STATE U"
        state_v_arr = []
        for old_state_u in scores[n-1][state_v]:
          state_v_arr.append(scores[n-1][state_v][old_state_u][state_u])
        
        best_score_v = max(state_v_arr)
        current_stop_score = best_score_v + trans
        scores[n]["STOP"][state_u][state_v] = current_stop_score


    # Backtracking path
    path = ["STOP"]
    stop_lst = []

    for state_u in scores[n]["STOP"]:
      state_u_lst = []
      for state_v in scores[n]["STOP"][state_u]:
        state_u_lst.append((state_v, scores[n]["STOP"][state_u][state_v]))
      
      # Finding best score for a specific STATE U among all possible STATE V
      max_state_v_for_state_u = max(state_u_lst, key=itemgetter(1))
      
      # Tuple of (state u, state v, score)
      max_tuple = (state_u, max_state_v_for_state_u[0], max_state_v_for_state_u[1])
      stop_lst.append(max_tuple)
    
    # maximum score for among all previous STATE U and all previous STATE V
    max_stop_tuple = max(stop_lst, key=itemgetter(2))
    
    # Insert STATE V
    path.insert(0, max_stop_tuple[1])
    # Insert STATE U
    path.insert(0, max_stop_tuple[0])

    # To refer to the second last STATE
    prev = -2

    for k in range(n-1, 0, -1):
      state_u = scores[k][path[prev]]
      state_v_lst = []

      # Find all STATE U that results in the specific STATE V pointed by prev
      for i in state_u.keys():
        if path[prev-1] in state_u[i]:
          state_v_lst.append((i, state_u[i][path[prev-1]]))
      
      # Obtain the ARGMAX of STATE U -> STATE V pointed by prev
      max_score = max(state_v_lst, key=itemgetter(1))
      
      # Shift STATE being looked at by -1
      prev = prev - 1
      
      # Insert max STATE U into PATH
      path.insert(0, max_score[0])


  return path


if __name__ == '__main__':
    design_dataset = ["EN"]

    for i in design_dataset:
        train = root_dir + "{folder}/train".format(folder = i)
        evaluation = root_dir + "{folder}/dev.in".format(folder = i)
        
        # training
        transition_tracker = HMM2_count_transition(train)

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
        
        # each line is a word
        for line in lines:        
            if line != "\n":
                line = line.strip()
                sentence.append(line)
            else:
                sentence_prediction = HMM2_viterbi(emission_tracker, transition_tracker, obs_all, sentence)
                sentence_prediction.remove("START")
                sentence_prediction.remove("STOP")
                all_prediction = all_prediction + sentence_prediction
                all_prediction = all_prediction + ["\n"]
                sentence = []
        
        assert len(lines) == len(all_prediction)
        # create output file
        with open(root_dir + "{folder}/dev.p5.out".format(folder = i), "w") as g:
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