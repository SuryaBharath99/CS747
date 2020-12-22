import numpy as np
import math
import matplotlib.pyplot as plt



#######################################################################################
########### Grid given in example 6.5

number_of_rows = 7
number_of_columns = 10
Start_column = 0
Start_row = 3
End_column = 7
End_row = 3
wind_weights = np.zeros(number_of_columns)
wind_weights = [0, 0 ,0, 1, 1, 1, 2, 2, 1, 0]
######### Parameters
epsilon = 0.1 
alpha = 0.5
gamma = 1
total_time_steps = 8000

number_of_states = number_of_rows * number_of_columns
Start = Start_column + Start_row * number_of_columns
End = End_column + End_row * number_of_columns 
##### can be changed as per requirement for other grids.
#########################################################################################

def state_update(current_state,current_action,wind_weights,number_of_columns, number_of_rows,number_of_states,End , number_of_actions,RandomSeed ,stochasticity ):
    # np.random.seed(RandomSeed)
    current_column = current_state %number_of_columns
    current_row = current_state / number_of_columns
    current_row = np.int(current_row)
    next_row = current_row
    next_column = current_column
    wind_effect = wind_weights[current_column]
    ### Stochastic
    if stochasticity == True :
        stochastic_shift = np.random.randint(-1,2)
        wind_effect = wind_effect+stochastic_shift

    ## normal
    if current_action == 0 :
        next_row = current_row -1 - wind_effect
    if current_action == 1:
        next_row = current_row +1  - wind_effect
    if current_action == 2:
        next_row = current_row -wind_effect
        next_column =current_column +1
    if current_action ==3:
        next_row = current_row -wind_effect
        next_column = current_column -1             

    ## including king
    
    if current_action == 4:
        next_row = current_row - 1 - wind_effect
        next_column = current_column + 1
    
    if current_action == 5:
        next_row = current_row + 1 - wind_effect
        next_column = current_column+1
    
    if current_action == 6:
        next_column =current_column + 1
        next_row =current_row -1 - wind_effect
    
    if current_action == 7:
        next_column = current_column -1
        next_row = current_row +1 - wind_effect

    ###### Boundary cases of crossing wall 

    if next_column <0 :
        next_column = 0
    if next_row <0 :
        next_row = 0
    if next_row >= number_of_rows:
        next_row = number_of_rows - 1 
    if next_column >= number_of_columns:
        next_column = number_of_columns -1

    ## next state evaluation
    # print(next_column, next_row)
    next_state = next_column + (next_row * number_of_columns )
    next_state = np.int(next_state)
    return next_state



#####################  Sarsa(0)
def sarsa(total_time_steps,Start, End, number_of_actions,wind_weights,number_of_columns, number_of_rows,number_of_states, alpha , gamma ,RandomSeed ,stochasticity ):
    np.random.seed(RandomSeed)
    Q = np.zeros((number_of_states, number_of_actions))
    current_time = 0
    total_episodes = 0
    episodes = []
    

    while current_time < total_time_steps:

        ##### Initial
        ## s 
        current_state = Start
        
        ## action epsilon greedy
        Trig = np.random.binomial(1,epsilon,1)
        Trig = np.int(Trig)
        if Trig == 1 :
            a = np.random.randint(number_of_actions)
        if Trig == 0 :
            Q1 = Q[current_state]
            Q1  = Q1[:]
            a = np.argmax(Q1)  

        while (current_state != End and current_time < total_time_steps) :

            next_state = state_update(current_state,a,wind_weights,number_of_columns, number_of_rows,number_of_states,End , number_of_actions,RandomSeed ,stochasticity)
            
            ## next action epsilon greedy
            Trig = np.random.binomial(1,epsilon,1)
            Trig = np.int(Trig)
            if Trig == 1 :
                a1 = np.random.randint(number_of_actions)
            if Trig == 0 :
                Q1 = Q[next_state]
                Q1  = Q1[:]
                a1 = np.argmax(Q1)    

            ### reward as mentioned in example
            if next_state == End :
                r = 0
            else :
                r = -1

            ### Q update 
            Q[current_state][a] = Q[current_state][a] + alpha * (r + (gamma*Q[next_state][a1])  - Q[current_state][a] )
                       
            if next_state == End :
                total_episodes = total_episodes +1
                        
            episodes.append(total_episodes)
            
            a = a1
            current_state = next_state
            current_time =current_time +1

        
    return episodes



################### Q - learning



def QLearning(total_time_steps,Start, End, number_of_actions,wind_weights,number_of_columns, number_of_rows,number_of_states, alpha , gamma ,RandomSeed ,stochasticity):
    np.random.seed(RandomSeed)
    Q = np.zeros((number_of_states, number_of_actions))
    current_time = 0
    total_episodes = 0
    episodes = []
    

    while current_time < total_time_steps:

        ##### Initial
        ## s 
        current_state = Start
        
        ## action epsilon greedy
        Trig = np.random.binomial(1,epsilon,1)
        Trig = np.int(Trig)
        if Trig == 1 :
            a = np.random.randint(number_of_actions)
        if Trig == 0 :
            Q1 = Q[current_state]
            Q1  = Q1[:]
            a = np.argmax(Q1)  

        while (current_state != End and current_time < total_time_steps) :

            next_state = state_update(current_state,a,wind_weights,number_of_columns, number_of_rows,number_of_states,End , number_of_actions,RandomSeed ,stochasticity)
            # print(next_state)

            
            ### reward as mentioned in example
            if next_state == End :
                r = 0
            else :
                r = -1

            ### Q update 
            Q[current_state][a] = Q[current_state][a] + alpha * (r + (gamma*max(Q[next_state][:]))  - Q[current_state][a] )
            
           

            ## next action epsilon greedy
            Trig = np.random.binomial(1,epsilon,1)
            Trig = np.int(Trig)
            if Trig == 1 :
                a1 = np.random.randint(number_of_actions)
            if Trig == 0 :
                Q1 = Q[next_state]
                Q1  = Q1[:]
                a1 = np.argmax(Q1)    


            if next_state == End :
                total_episodes = total_episodes +1
            
            
            episodes.append(total_episodes)
            
            a = a1
            current_state = next_state
            current_time =current_time +1

        
    return episodes







################### Expected Sarsa

def Expected_Sarsa(total_time_steps,Start, End, number_of_actions,wind_weights,number_of_columns, number_of_rows,number_of_states, alpha , gamma ,RandomSeed ,stochasticity ):
    np.random.seed(RandomSeed)
    Q = np.zeros((number_of_states, number_of_actions))
    current_time = 0
    total_episodes = 0
    episodes = []
    

    while current_time < total_time_steps:

        ##### Initial
        ## s 
        current_state = Start
        
        ## action epsilon greedy
        Trig = np.random.binomial(1,epsilon,1)
        Trig = np.int(Trig)
        if Trig == 1 :
            a = np.random.randint(number_of_actions)
        if Trig == 0 :
            Q1 = Q[current_state]
            Q1  = Q1[:]
            a = np.argmax(Q1)  

        while (current_state != End and current_time < total_time_steps) :

            next_state = state_update(current_state,a,wind_weights,number_of_columns, number_of_rows,number_of_states,End , number_of_actions,RandomSeed ,stochasticity)
            # print(next_state)

            
            ### reward as mentioned in example
            if next_state == End :
                r = 0
            else :
                r = -1

            prob_mat = np.zeros(number_of_actions)
            act = np.float(number_of_actions)
            prob = epsilon / act
            for i in range(number_of_actions):
                prob_mat[i] = prob 
            Q1 = Q[next_state]
            Q1  = Q1[:]
            b = np.argmax(Q1)
            prob_mat[b]  = prob + 1.0-epsilon
            prob_mat = np.array(prob_mat)   
            targ = np.matmul(prob_mat,np.array(Q[next_state][:]) )
            ### Q update 
            Q[current_state][a] = Q[current_state][a] + alpha * (r + (gamma*targ)  - Q[current_state][a] )
            
           

            ## next action epsilon greedy
            Trig = np.random.binomial(1,epsilon,1)
            Trig = np.int(Trig)
            if Trig == 1 :
                a1 = np.random.randint(number_of_actions)
            if Trig == 0 :
                Q1 = Q[next_state]
                Q1  = Q1[:]
                a1 = np.argmax(Q1)    


            if next_state == End :
                total_episodes = total_episodes +1
            
            
            episodes.append(total_episodes)
            
            a = a1
            current_state = next_state
            current_time =current_time +1

        
    return episodes

















############# plotting
number_of_actions = 4
sarsa_episodes = np.zeros(total_time_steps)
for rs in range(10):
    sarsa_episodes = sarsa_episodes + sarsa(total_time_steps,Start, End, number_of_actions,wind_weights,number_of_columns, number_of_rows,number_of_states, alpha , gamma,rs,stochasticity = False)
sarsa_episodes = sarsa_episodes*(1/10.0)
# print(sarsa_episodes[-20:])
plt.plot(sarsa_episodes, label ="Sarsa(0) without KING")



number_of_actions = 8
sarsa_episodes = np.zeros(total_time_steps)
for randomseed in range(10):
    sarsa_episodes = sarsa_episodes + sarsa(total_time_steps,Start, End, number_of_actions,wind_weights,number_of_columns, number_of_rows,number_of_states, alpha , gamma,randomseed ,stochasticity =False)
sarsa_episodes = sarsa_episodes*(1/10.0)
plt.plot(sarsa_episodes, label = "Sarsa(0) with KING")


number_of_actions = 8
sarsa_episodes = np.zeros(total_time_steps)
for randomseed in range(10):
    sarsa_episodes = sarsa_episodes + sarsa(total_time_steps,Start, End, number_of_actions,wind_weights,number_of_columns, number_of_rows,number_of_states, alpha , gamma, randomseed ,stochasticity = True )
sarsa_episodes = sarsa_episodes*(1/10.0)
plt.plot(sarsa_episodes, label ="Sarsa(0) with Stochastic & KING")


number_of_actions = 4
q_episodes = np.zeros(total_time_steps)
for randomseed in range(10):
    q_episodes = q_episodes + QLearning(total_time_steps,Start, End, number_of_actions,wind_weights,number_of_columns, number_of_rows,number_of_states, alpha , gamma, randomseed ,stochasticity = False)
q_episodes = q_episodes *( 1/ 10.0)
# print(q_episodes[-20:])
plt.plot(q_episodes, label ="Q-Learning")

number_of_actions = 4
ExpSarsa_episodes = np.zeros(total_time_steps)
for randomseed in range(10):
    ExpSarsa_episodes = ExpSarsa_episodes + Expected_Sarsa(total_time_steps,Start, End, number_of_actions,wind_weights,number_of_columns, number_of_rows,number_of_states, alpha , gamma, randomseed ,stochasticity = False)
ExpSarsa_episodes = ExpSarsa_episodes * (1/10.0)
# print(ExpSarsa_episodes[-20:])
plt.plot(ExpSarsa_episodes, label ="Expected Sarsa")





# plt.xlim(0,1500)
# plt.ylim(0,15)
# plt.title (" ")
plt.xlabel("Time Steps")
plt.ylabel("Episodes")
plt.legend()
plt.show()
# plt.savefig("wind_grid.png")
