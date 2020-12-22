import numpy as np
import sys


# input is epsilon, number of arms.
def epsilon_greedy(bandit_instance,RandomSeed,no_of_arms,epsilon,horizon):
    np.random.seed(RandomSeed)
    
    emp_mean = []
    pulls_per_arms = []
    i = 0
    while i<no_of_arms:
        emp_mean.append(0)
        i = i+1

    i = 0
    while i<no_of_arms:
        pulls_per_arms.append(0)
        i = i+1


    expected_reward =0
    t = 0
    while t<horizon:
        Trig = np.random.binomial(1,epsilon,1)
        Trig = np.int(Trig)
        #Exploit
        # print(Trig)
        if Trig == 1 :
            # arm_id = np.random.randint(no_of_arms)
            arm_id = np.argmax(emp_mean)
            # print("exploit")
        #Explore
        else: 
            # arm_id = np.argmax(emp_mean)
            arm_id = np.random.randint(no_of_arms)
            # print("explore")

        arm_prob = bandit_instance[arm_id]
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)
        
        #updating the empirical mean and pulls per arm
        emp_mean[arm_id] = (((emp_mean[arm_id])*pulls_per_arms[arm_id]*1.0)+arm_reward+0.0)/(pulls_per_arms[arm_id]+1.0)
        
        # print(emp_mean[arm_id]*(np.float(pulls_per_arms[arm_id])))
        # print(emp_mean[arm_id]*((pulls_per_arms[arm_id])))
        pulls_per_arms[arm_id] = pulls_per_arms[arm_id]+1
        
        expected_reward = expected_reward + arm_reward

        t = t+1


    return expected_reward





def ucb(bandit_instance,RandomSeed,no_of_arms,horizon):
    np.random.seed(RandomSeed)
    emp_mean = []
    pulls_per_arms = []
    i = 0
    while i<no_of_arms:
        emp_mean.append(0)
        i = i+1

    i = 0
    while i<no_of_arms:
        pulls_per_arms.append(0)
        i = i+1
    t = 0
    ####################----> Pulling each arm  <-------####################################################
    expected_reward = 0
    a = 0
    while a<no_of_arms:
        arm_prob = bandit_instance[a]
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)

        emp_mean[a] = (((emp_mean[a])*np.float(pulls_per_arms[a]))+arm_reward+0.0)/(pulls_per_arms[a]+1.0)
        
        pulls_per_arms[a] = pulls_per_arms[a]+1
            
        expected_reward = expected_reward + arm_reward
        a  = a+1

    # print(pulls_per_arms)
    a = 0
    while a<no_of_arms:
        arm_prob = bandit_instance[a]
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)

        emp_mean[a] = (((emp_mean[a])*np.float(pulls_per_arms[a]))+arm_reward+0.0)/(pulls_per_arms[a]+1.0)
        
        pulls_per_arms[a] = pulls_per_arms[a]+1
            
        expected_reward = expected_reward + arm_reward
        a  = a+1    

    
    t = 2*no_of_arms
    while t < horizon:
        
        ####--> ucb to determine good arm to pull at t pull <--###
        ucb = []
        k = 0
        while k<no_of_arms:
            value = emp_mean[k]+ np.sqrt((2.0 *(np.log(t))*1.0/(pulls_per_arms[k]+0.0)))
            value = np.float(value)
            ucb.append(value)
            k = k+1
        arm_id = np.argmax(ucb)
        # extracting arm probability from given bandit instance
        arm_prob = bandit_instance[arm_id]
        ##pulling an arm to get reward
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)
        
        #updating the empirical mean and pulls per arm
        emp_mean[arm_id] = (((emp_mean[arm_id])*pulls_per_arms[arm_id]*1.0)+arm_reward+0.0)/(pulls_per_arms[arm_id]+1.0)
        
        pulls_per_arms[arm_id] = pulls_per_arms[arm_id]+1
        
        expected_reward = expected_reward + arm_reward

        t = t+1

    
    return expected_reward

















def kl_ucb(bandit_instance,RandomSeed,no_of_arms,horizon):
    np.random.seed(RandomSeed)
    emp_mean = []
    pulls_per_arms = []
    i = 0
    while i<no_of_arms:
        emp_mean.append(0)
        i = i+1

    i = 0
    while i<no_of_arms:
        pulls_per_arms.append(0)
        i = i+1
    t = 0
    ####################----> Pulling each arm  <-------####################################################
    expected_reward = 0
    a = 0
    while a<no_of_arms:
        arm_prob = bandit_instance[a]
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)

        emp_mean[a] = (((emp_mean[a])*np.float(pulls_per_arms[a]))+arm_reward+0.0)/(pulls_per_arms[a]+1.0)
        
        pulls_per_arms[a] = pulls_per_arms[a]+1
            
        expected_reward = expected_reward + arm_reward
        a  = a+1

    # print(pulls_per_arms)
    a = 0
    while a<no_of_arms:
        arm_prob = bandit_instance[a]
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)

        emp_mean[a] = (((emp_mean[a])*np.float(pulls_per_arms[a]))+arm_reward+0.0)/(pulls_per_arms[a]+1.0)
        
        pulls_per_arms[a] = pulls_per_arms[a]+1
            
        expected_reward = expected_reward + arm_reward
        a  = a+1    

    
    t = 2*no_of_arms
    while t < horizon:
        
        ####--> ucb to determine good arm to pull at t pull <--###
        kl_ucb = []
        k = 0
        while k<no_of_arms:
            # value = emp_mean[k]+ np.sqrt((2.0 *(np.log(t))*1.0/(pulls_per_arms[k]+0.0)))
            # value = np.float(value)
            
            U = np.log(t) + 3*(np.log(np.log(t))) 
            
            q = emp_mean[k]
            q = np.float(q)
           
            ######### making compatible with division
            if q == 1.0:
                q = q-0.0001
            if q ==0.0 :
                q = 0.0001


            maxq = emp_mean[k]
            emp_mean[k] = np.float(emp_mean[k])    
            while q<1.0:
                if emp_mean[k] == 1.0:
                    emp_mean[k] = emp_mean[k]-0.0001
                if emp_mean[k] == 0.0:
                    emp_mean[k] == 0.0001    

                kl_div = emp_mean[k]*(np.log((emp_mean[k]+0.0)/q))  +  (1.0-emp_mean[k])*(np.log((1.0-emp_mean[k])/(1.0-q)))
                
                if kl_div < U/(pulls_per_arms[k]+0.0):
                    maxq = q
                q = q+0.005

            value =maxq
            value = np.float(value)


            
            
            kl_ucb.append(value)
            k = k+1
        
        arm_id = np.argmax(kl_ucb)
        # extracting arm probability from given bandit instance
        arm_prob = bandit_instance[arm_id]
        ##pulling an arm to get reward
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)
        




        #updating the empirical mean and pulls per arm
        emp_mean[arm_id] = (((emp_mean[arm_id])*pulls_per_arms[arm_id]*1.0)+arm_reward+0.0)/(pulls_per_arms[arm_id]+1.0)
        
        pulls_per_arms[arm_id] = pulls_per_arms[arm_id]+1
        
        expected_reward = expected_reward + arm_reward

        t = t+1

    
    return expected_reward





    
def thompson_sampling(bandit_instance,RandomSeed,no_of_arms,horizon):
    np.random.seed(RandomSeed)
    emp_mean = []
    pulls_per_arms = []
    success  = []
    i = 0
    emp_mean = np.zeros(no_of_arms)

    pulls_per_arms = np.zeros(no_of_arms)
    
    success = np.zeros(no_of_arms)
    
    failure = np.zeros(no_of_arms)

    t = 0

    ####################----> Pulling each arm  <-------####################################################
    expected_reward = 0
    a = 0
    while a<no_of_arms:
        arm_prob = bandit_instance[a]
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)
        if arm_reward == 1:
            success[a] = success[a]+ 1

        emp_mean[a] = (((emp_mean[a])*np.float(pulls_per_arms[a]))+arm_reward+0.0)/(pulls_per_arms[a]+1.0)
        
        pulls_per_arms[a] = pulls_per_arms[a]+1


        expected_reward = expected_reward + arm_reward
        a  = a+1
    

    
    t = no_of_arms
    while t < horizon:
        failure = pulls_per_arms-success
        ####--> thompson sampling to determine good arm to pull at t pull <--###

        beta = np.zeros(no_of_arms)

        a = 0
        while a < no_of_arms:
            beta[a] = np.random.beta((success[a]+1), (failure[a]+1))
            a = a+1


        arm_id = np.argmax(beta)
        # extracting arm probability from given bandit instance
        arm_prob = bandit_instance[arm_id]
        ##pulling an arm to get reward
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)
        
        if arm_reward == 1:
            success[arm_id] = success[arm_id]+ 1

        #updating the empirical mean and pulls per arm
        emp_mean[arm_id] = (((emp_mean[arm_id])*pulls_per_arms[arm_id]*1.0)+arm_reward+0.0)/(pulls_per_arms[arm_id]+1.0)
        
        pulls_per_arms[arm_id] = pulls_per_arms[arm_id]+1
        
        expected_reward = expected_reward + arm_reward

        t = t+1

    
    # print(success)
    # print(failure)
    return expected_reward


    
def thompson_sampling_with_hint(bandit_instance,RandomSeed,no_of_arms,horizon):
    np.random.seed(RandomSeed)
    emp_mean = []
    pulls_per_arms = []
    success  = []
    emp_mean = np.zeros(no_of_arms)
    distance_from_optimal = np.zeros(no_of_arms)
    thompson_confidence = np.zeros(no_of_arms)

    pulls_per_arms = np.zeros(no_of_arms)
    
    success = np.zeros(no_of_arms)
    
    failure = np.zeros(no_of_arms)

    t = 0
    hint_ls = np.sort(bandit_instance)

    # scaling = abs(emp_mean-max(hint_ls))

    ####################----> Pulling each arm  <-------####################################################
    expected_reward = 0
    a = 0
    while a<no_of_arms:
        arm_prob = bandit_instance[a]
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)
        if arm_reward == 1:
            success[a] = success[a]+ 1

        emp_mean[a] = (((emp_mean[a])*np.float(pulls_per_arms[a]))+arm_reward+0.0)/(pulls_per_arms[a]+1.0)
        
        pulls_per_arms[a] = pulls_per_arms[a]+1


        expected_reward = expected_reward + arm_reward
        a  = a+1

    
    t = no_of_arms
    


    while t < horizon/2:

        failure = pulls_per_arms-success
            ####--> thompson sampling to determine good arm to pull at t pull <--###

        beta = np.zeros(no_of_arms)

        a = 0
        while a < no_of_arms:
            beta[a] = np.random.beta((success[a]+1), (failure[a]+1))
            a = a+1


        arm_id = np.argmax(beta)
            # extracting arm probability from given bandit instance
        arm_prob = bandit_instance[arm_id]
            ##pulling an arm to get reward
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)
            
        if arm_reward == 1:
            success[arm_id] = success[arm_id]+ 1

            #updating the empirical mean and pulls per arm
        emp_mean[arm_id] = (((emp_mean[arm_id])*pulls_per_arms[arm_id]*1.0)+arm_reward+0.0)/(pulls_per_arms[arm_id]+1.0)
            
        pulls_per_arms[arm_id] = pulls_per_arms[arm_id]+1
            
        expected_reward = expected_reward + arm_reward
        t = t +1 

    while t < horizon:
        failure = pulls_per_arms-success
        ####--> thompson sampling to determine good arm to pull at t pull <--###

        beta = np.zeros(no_of_arms)

        distance_from_optimal = np.absolute(emp_mean-(np.amax(hint_ls)))
        

        a = 0
        while a < no_of_arms:
            beta[a] = np.random.beta((success[a]+1), (failure[a]+1))
            a = a+1
       
        thompson_confidence = beta
        
        # ranking = np.sort(thompson_confidence)

        dummy = -1*thompson_confidence
        rank1 = np.zeros(no_of_arms)        
        dum = dummy.argsort()
        rank1 = np.arange(len(dummy))[dum.argsort()]
        


        rank2 = np.zeros(no_of_arms)
        dum = distance_from_optimal.argsort()
        rank2 = np.arange(len(distance_from_optimal))[dum.argsort()]




        rank12 = np.zeros(no_of_arms)
        rank12 = rank1+rank2    
        


        ranking = np.zeros(no_of_arms)
        dum = rank12.argsort()
        ranking = np.arange(len(rank12))[dum.argsort()]
        
        comparision_distance = np.zeros(no_of_arms)
        indices = []
        j = 0
        while j<no_of_arms:
            index = ranking[j]
            comparision_distance[j] = np.absolute(emp_mean[j]-hint_ls[index]) 
            indices.append(index)
            j = j+1 


        #normalization
        dist_sum = np.sum(comparision_distance)
        comparision_distance = comparision_distance*(1/(dist_sum+0.0))


        # print(distance_from_optimal,comparision_distance)

        new_confidence = np.zeros(no_of_arms)
        # print(indices)
        a = 0
        
        while a < no_of_arms:
            if (rank1[a])<=(np.int(np.log2(no_of_arms))):
                # if comparision_distance[a] > 0.3 :
                # new_confidence[a] = beta[a]/(comparision_distance[a]*distance_from_optimal[a]*distance_from_optimal[a])
                new_confidence[a] = beta[a]*(comparision_distance[a])/(distance_from_optimal[a]*distance_from_optimal[a])
            # else:
            #     new_confidence[a] = beta[a]*comparision_distance[a]
            a = a+1   

      


        # print(emp_mean,hint_ls)
        # print(emp_mean)
        scaling = np.absolute(emp_mean-(np.amax(hint_ls)))
        # print("distance:",scaling)
        scaling = np.reciprocal(scaling)
        # print("Scaling:",scaling)
        # print("Beta confidence :",beta)
        # print("Modified confidence:",beta*scaling)



        arm_id = np.argmax(new_confidence)
        # arm_id = np.argmax(beta*scaling)
        
        # print(arm_id)
        # extracting arm probability from given bandit instance
        arm_prob = bandit_instance[arm_id]
        ##pulling an arm to get reward
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)
        
        if arm_reward == 1:
            success[arm_id] = success[arm_id]+ 1

        #updating the empirical mean and pulls per arm
        emp_mean[arm_id] = (((emp_mean[arm_id])*pulls_per_arms[arm_id]*1.0)+arm_reward+0.0)/(pulls_per_arms[arm_id]+1.0)
        
        pulls_per_arms[arm_id] = pulls_per_arms[arm_id]+1
        
        expected_reward = expected_reward + arm_reward

        t = t+1


    # print(success)
    # print(failure)
    return expected_reward



no_of_args = len(sys.argv)

for i in range(1,no_of_args):
    if sys.argv[i] == "--instance":
        i = i + 1
        bandit_instance_path  = sys.argv[i]
    elif sys.argv[i] == "--algorithm":
        i = i+1
        algo =sys.argv[i]
    elif sys.argv[i] == "--randomSeed":
        i = i+1
        RandomSeed = sys.argv[i]
    elif sys.argv[i] == "--epsilon":
        i = i+1
        epsilon = sys.argv[i]
        epsilon = np.float(epsilon)
    elif sys.argv[i] == "--horizon":
        i = i+1
        horizon = sys.argv[i]



bandit_instance = np.loadtxt(bandit_instance_path)
no_of_arms = len(bandit_instance)



RandomSeed = np.int(RandomSeed)

horizon = np.int(horizon)



if algo == "epsilon-greedy":
    expected_reward = epsilon_greedy(bandit_instance,RandomSeed,no_of_arms,1-epsilon,horizon)
elif algo == "ucb":
    expected_reward = ucb(bandit_instance,RandomSeed,no_of_arms,horizon)
elif algo == "kl-ucb":
    expected_reward = kl_ucb(bandit_instance,RandomSeed,no_of_arms,horizon)
elif algo == "thompson-sampling":
    expected_reward = thompson_sampling(bandit_instance,RandomSeed,no_of_arms,horizon)        
elif algo == "thompson-sampling-with-hint":
    expected_reward = thompson_sampling_with_hint(bandit_instance,RandomSeed,no_of_arms,horizon)  



Maximum_reward = max(bandit_instance)*horizon



Regret = Maximum_reward -expected_reward



# print(bandit_instance_path + ", " + algo +", "+RandomSeed+", " +epsilon+", "+horizon+", "+ Regret  )




# file1 = open("outputDataT2.txt","a")
# file1.write(bandit_instance_path + ", " + algo +", "+str(RandomSeed)+", " +str(epsilon)+", "+str(horizon)+", "+ str(Regret)+"\n") 
# file1.close() 

print(bandit_instance_path + ", " + algo +", "+str(RandomSeed)+", " +str(epsilon)+", "+str(horizon)+", "+ str(Regret)+"\n")