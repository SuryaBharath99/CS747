import numpy as np

    
with open("data\mdp\continuing-mdp-10-5.txt") as inp:
    mdp_data = inp.readlines()
    
mdp_data = [x.strip() for x in mdp_data] 

# number of actions
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

# print(transition[0])

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



def Q(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma,value_mat):
    Q_big = []
    for i in range(no_of_states):
        mul = np.matmul((np.array(transition_big_mat[i]).T),(np.array(Reward_big_mat[i])))
        diag = mul.diagonal()
        V = np.matmul((np.array(transition_big_mat[i]).T),value_mat)
        sums = diag + gamma*V
        Q_big.append(sums)
    return Q_big


policy = np.zeros(no_of_states)

## initial policy
i = 0
while i< no_of_states:
    j = 0
    while j < np.int(len(initial_state)):
        if i == np.int(initial_state[j]):
            policy[i] = actions[j]
        j = j+1
    i = i+1

# print(policy)


####value function states for this policy


def value_fn(policy,no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma):
    lhs_mat = []
    rhs_mat = []
    for i in range(no_of_states):
        mul = np.matmul((np.array(transition_big_mat[i]).T),(np.array(Reward_big_mat[i])))
        diag = mul.diagonal()
        vect =  (np.array(transition_big_mat[i]))
        g = vect.T
        h = np.int(policy[i])
        g1 = gamma*g[h]  
        rhs = diag[h]
        rhs_mat.append(rhs)
        g1 = -1.0*g1
        g1[i] = 1.0 + g1[i]
        lhs_mat.append(g1)  
    val = np.linalg.solve(np.array(lhs_mat),np.array(rhs_mat))    
    return val




######### generation of IA and IS
IA_big= []
IS = []
k = 0

while k < no_of_states:
    Val = value_fn(policy,no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)
    Q_mat_big = Q(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma,Val)
    f = 0
    IA =[]
    IS = []
    while f<no_of_actions:
        Q_mat = np.array(Q_mat_big[k])
        if Q_mat[f] > Val[k]:
            IA.append(f)
        f = f+1
    if np.int(len(IA))>= 1:
        IS.append(k)
    IA_big.append(IA)     
    k = k+1

# print(IS)

u = 0
l = np.int(len(IS))
dummy = np.zeros(no_of_states)
#### policy update
while u< l:
    y = np.int(IS[u])
    if np.int(len(IA_big[y])) > 0:
        policy[y] = np.int(IA_big[y][0])
        dummy[y] = np.int(IA_big[y][0])
    u = u+1
# print(policy)


#### hwi implementation

IA = []
j = np.int(len(IS))

policy_old = []
policy_old.append(dummy)


while j > 0.0 :
    # print("hi")
    IA_big= []
    diff_big = []
    k = 0
    IS = []
    for k in range(no_of_states):
        Val = value_fn(policy,no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)
        Q_mat_big = Q(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma,Val)
        f = 0
        IA =[]
        diff = []
        for f in range(no_of_actions):
            Q_mat = np.array(Q_mat_big[k])
            if Q_mat[f] > Val[k]:
                IA.append(f)
                diff.append(Q_mat[f] - Val[k])
        if np.int(len(IA)) > 0:
            IS.append(k)
        IA_big.append(IA)
        diff_big.append(diff)     
    u = 0
    l = np.int(len(IS))
    j = np.int(len(IS))
    #### policy update
    z = 0 
    for u in range(l):
        y = np.int(IS[u])
        if np.int(len(IA_big[y])) >= 1.0:
            p = np.argmax(np.array(diff_big[y]))
            if policy[y] == np.int(IA_big[y][p]) :
                z = z+1
            policy[y] = np.int(IA_big[y][p])

    if z == np.int(len(IS)):
        j = 0
    else:
        j = 1
    # w = np.int(len(policy_old))        
    # if list(np.array(policy_old[w-1])) == list(policy):
    #     j = 0
    #     print("hello")
    # else:
    #     j = 1
    #     print("bye")
    #     policy_old.append(policy)   
    # print(policy)    


print(policy)    
Value = value_fn(policy,no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)
V1 =np.round(Value, decimals = 6)
print(V1)