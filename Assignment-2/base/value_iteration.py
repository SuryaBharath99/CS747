import numpy as np


with open("data\mdp\continuing-mdp-10-5.txt") as inp:
    mdp_data = inp.readlines()
    
mdp_data = [x.strip() for x in mdp_data] 
a = mdp_data[1].split()
no_of_actions = a[1]

# number of states
a = mdp_data[0].split()
no_of_states = a[1]

#start state ID
a = mdp_data[2].split()
start_state = a[1]

#terminal states
terminal_states = []
a = mdp_data[3].split()
dum = np.int(a[1])
l = np.int(len(a))
if dum != -1 :
    i = 1
    while i < l :
        terminal_states.append(a[i])
        i = i+1  

# Intial states
initial_state = []
l1 = np.int(len(mdp_data))
k = 4
while k < l1-2 :
    a = mdp_data[k].split()
    initial_state.append(a[1])
    k = k+1


#actions
actions = []
l1 = np.int(len(mdp_data))
k = 4
while k < l1-2 :
    a = mdp_data[k].split()
    actions.append(a[2])
    k = k+1

#final state
final_state = []
l1 = np.int(len(mdp_data))
k = 4
while k < l1-2 :
    a = mdp_data[k].split()
    final_state.append(a[3])
    k = k+1

#reward
reward = []
l1 = np.int(len(mdp_data))
k = 4
while k < l1-2 :
    a = mdp_data[k].split()
    reward.append(a[4])
    k = k+1    

#transition prob
transition = []
l1 = np.int(len(mdp_data))
k = 4
while k < l1-2 :
    a = mdp_data[k].split()
    transition.append(a[5])
    k = k+1

##gamma
gamma = mdp_data[l1-1].split()
gamma = gamma[1]
gamma = float(gamma)

no_of_actions = np.int(no_of_actions)
no_of_states = np.int(no_of_states)


###################################################


empt = np.zeros((no_of_states,no_of_actions))

transition_big_mat = []
Reward_big_mat = []
i  = 0
while i<no_of_states:
    transition_big_mat.append(empt)
    Reward_big_mat.append(empt)
    i = i+1

j = 0 

while j < np.int(len(initial_state)) :
    s = np.int(initial_state[j])
    a = np.int(actions[j])
    s1 = np.int(final_state[j]) 
    trans = np.array(transition_big_mat[s])
    # print(trans.shape)
    trans[s1,a] = transition[j]
    rew = np.array(Reward_big_mat[s])
    rew [s1,a] = reward[j]
    transition_big_mat[s] = trans
    Reward_big_mat[s] = rew
    j = j+1


######################################## successfully generated T and R



i = 0
Vt1 = np.zeros(no_of_states)

a = np.ones(no_of_states)
a =gamma*a
Vt1 = np.zeros(no_of_states)
Vt1= gamma*Vt1
Vt = np.ones(no_of_states)
Vt = gamma*Vt

while 0.000001 < max(abs(Vt1-Vt)):
    for i in range(no_of_states):
        mul = np.matmul((np.array(transition_big_mat[i]).T),(np.array(Reward_big_mat[i])))
        diag = mul.diagonal()
        vect = np.matmul((np.array(transition_big_mat[i]).T),(gamma*Vt)) 
        sumed = diag + vect
        Vt[i] = Vt1[i]
        Vt1[i] = max(sumed) 


V1 =np.round(Vt1, decimals = 6)

policy =1* np.zeros(no_of_states)

for i in range(no_of_states):
        mul = np.matmul((np.array(transition_big_mat[i]).T),(np.array(Reward_big_mat[i])))
        diag = mul.diagonal()
        Optimal_vect = np.matmul((np.array(transition_big_mat[i]).T),(gamma*Vt1))
        sumed = diag + Optimal_vect
        policy[i]= np.int(np.argmax(sumed))

print(V1)







