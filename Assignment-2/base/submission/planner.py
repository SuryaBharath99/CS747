import numpy as np
import sys
import pulp as p
from pulp import *

def vi(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma): 
    i = 0
    Vt1 = np.zeros(no_of_states)
    a = np.ones(no_of_states)
    a =gamma*a
    Vt1 = np.zeros(no_of_states)
    Vt1= gamma*Vt1
    #initialization
    Vt = np.ones(no_of_states)
    Vt = gamma*Vt
    while 0.00000001 < max(abs(Vt1-Vt)):
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
    return V1, policy

def Q(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma,value_mat):
    Q_big = []
    for i in range(no_of_states):
        mul = np.matmul((np.array(transition_big_mat[i]).T),(np.array(Reward_big_mat[i])))
        diag = mul.diagonal()
        V = np.matmul((np.array(transition_big_mat[i]).T),value_mat)
        # print(value_mat.shape)
        sums = diag + gamma*V
        Q_big.append(sums)
    return Q_big

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

def hpi(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma):
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
    

    #### generating IS & IA for all states
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
        if np.int(len(IA))  >= 1:
            IS.append(k)
        IA_big.append(IA)     
        k = k+1
    u = 0
    l = np.int(len(IS))

    #### policy update
    while u< l:
        y = np.int(IS[u])
        if np.int(len(IA_big[y])) > 0:
            policy[y] = np.int(IA_big[y][0])
        u = u+1 
    IA = []
    j = np.int(len(IS))
    policy_old = np.zeros(no_of_states)
    while j > 0.0 :
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
        z = 0 
        #### policy update
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
    # print(policy)    
    Value = value_fn(policy,no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)
    V1 =np.round(Value, decimals = 6)
    # print(V1)

    return V1,policy







def lp(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma):    
    i = 0
    Vt1 = np.zeros(no_of_states)
    a = np.ones(no_of_states)
    a =gamma*a
    Vt1 = np.zeros(no_of_states)
    Vt1= gamma*Vt1
    Vt = np.ones(no_of_states)
    Vt = gamma*Vt

    ########## Linear Programming 
    prob = pulp.LpProblem('OptimalPolicy', LpMinimize)
    ## decision variables
    decision_value_fns = []
    for i in range(no_of_states):
        variable = str("Vs" + str(i))
        variable = pulp.LpVariable(str(variable),0)
        decision_value_fns.append(variable)
    summation = ""

    ## objective function
    for i in range(no_of_states):
        summation = summation+1.0*decision_value_fns[i]
    prob += summation
    sums = []
    constr = []
    for i in range(no_of_states):
        mul = np.matmul((np.array(transition_big_mat[i]).T),(np.array(Reward_big_mat[i])))
        diag = mul.diagonal()
        p = 0
        while p<no_of_actions:
            sums.append(diag[p])
            p = p+1
        vect =  (np.array(transition_big_mat[i]))
        g = vect.T
        b = np.zeros(no_of_states)
        h = 0
        while h< no_of_actions:
            v = g[h]
            const = ""
            m = 0
            while m<no_of_states:
                f = gamma*v[m]*decision_value_fns[m]
                const = const + f
                m = m+1                
            constr.append(const)
            h = h+1
    i = 0
    l = np.int(len(constr))
    k = 0
    while i<no_of_states:
        m = 0
        while m<no_of_actions:
            prob += (decision_value_fns[i]-constr[m+i*no_of_actions]>=sums[m+i*no_of_actions])
            m = m+1
        i = i+1
    pulp.PULP_CBC_CMD(msg=0).solve(prob)
    Vt1 = np.zeros(no_of_states)
    for v in prob.variables():
        i = 0
        while i< no_of_states:
            if v.name == str("Vs"+str(i)):
                Vt1[i] = v.varValue
            i = i + 1 

    V1 =np.round(Vt1, decimals = 6)

    policy =1* np.zeros(no_of_states)

    for i in range(no_of_states):
            mul = np.matmul((np.array(transition_big_mat[i]).T),(np.array(Reward_big_mat[i])))
            diag = mul.diagonal()
            Optimal_vect = np.matmul((np.array(transition_big_mat[i]).T),(gamma*Vt1))
            sumed = diag + Optimal_vect
            policy[i]= np.int(np.argmax(sumed))         

    return V1, policy



















no_of_args = len(sys.argv)
for i in range(1,no_of_args):
    if sys.argv[i] == "--mdp":
        i = i + 1
        input_mdp_path  = sys.argv[i]
    elif sys.argv[i] == "--algorithm":
        i = i+1
        algo =sys.argv[i]




with open(input_mdp_path) as inp:
    mdp_data = inp.readlines()
    
mdp_data = [x.strip() for x in mdp_data] 


# number of states
a = mdp_data[0].split()
no_of_states = a[1]
no_of_states = np.int(no_of_states)
#number of actions
a = mdp_data[1].split()
no_of_actions = a[1]
no_of_actions = np.int(no_of_actions)

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
#actions
actions = []
#final state
final_state = []
#reward
reward = []
#transition prob
transition = []



l1 = np.int(len(mdp_data))
k = 4
while k < l1-2 :
    a = mdp_data[k].split()
    initial_state.append(a[1])
    actions.append(a[2])
    final_state.append(a[3])
    reward.append(a[4])
    transition.append(a[5])
    k = k+1



##gamma
gamma = mdp_data[l1-1].split()
gamma = gamma[1]
gamma = float(gamma)

no_of_actions = np.int(no_of_actions)
no_of_states = np.int(no_of_states)

empt = np.zeros((no_of_states,no_of_actions))

transition_big_mat = []
Reward_big_mat = []
empt1 = np.zeros((no_of_states,no_of_actions))
empt1[:] = -10*no_of_states

i  = 0
while i<no_of_states:
    transition_big_mat.append(empt)
    Reward_big_mat.append(empt1)
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

##### Value iteration
if algo == "vi":
    values = vi(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)[0]
    policy = vi(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)[1]
###### Howard's Policy Iteration
elif algo == "hpi":
    values = hpi(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)[0]
    policy = hpi(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)[1]
##### Linear programming
elif algo == "lp":
    values = lp(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)[0]
    policy = lp(no_of_actions,no_of_states,transition_big_mat,Reward_big_mat,gamma)[1]

i = 0
l = np.int(len(values))
while i<l:
    print(str(values[i])+" "+str(policy[i])+"\n")
    i = i+1