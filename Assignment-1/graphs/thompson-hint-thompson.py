import numpy as np




   
def thompson_sampling(bandit_instance,RandomSeed,no_of_arms,horizon):
    reward = np.zeros(1600)
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
        reward[a] = expected_reward
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
        reward[t] = expected_reward

        t = t+1

    
    # print(success)
    # print(failure)
    return reward


































































    
def thompson_sampling_with_hint(bandit_instance,RandomSeed,no_of_arms,horizon):
    reward = np.zeros(1600)
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
        reward[a] = expected_reward
        a  = a+1
    
    # while a<no_of_arms:
    #     arm_prob = bandit_instance[a]
    #     arm_reward = np.random.binomial(1,arm_prob,1)
    #     arm_reward = np.int(arm_reward)
    #     if arm_reward == 1:
    #         success[a] = success[a]+ 1

    #     emp_mean[a] = (((emp_mean[a])*np.float(pulls_per_arms[a]))+arm_reward+0.0)/(pulls_per_arms[a]+1.0)
        
    #     pulls_per_arms[a] = pulls_per_arms[a]+1


    #     expected_reward = expected_reward + arm_reward
    #     a  = a+1
    
    # while a<no_of_arms:
    #     arm_prob = bandit_instance[a]
    #     arm_reward = np.random.binomial(1,arm_prob,1)
    #     arm_reward = np.int(arm_reward)
    #     if arm_reward == 1:
    #         success[a] = success[a]+ 1

    #     emp_mean[a] = (((emp_mean[a])*np.float(pulls_per_arms[a]))+arm_reward+0.0)/(pulls_per_arms[a]+1.0)
        
    #     pulls_per_arms[a] = pulls_per_arms[a]+1


    #     expected_reward = expected_reward + arm_reward
    #     a  = a+1
    
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
        reward[t] = expected_reward
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
        # while a < no_of_arms:
        #     if comparision_distance[a]>0.6:
        #         new_confidence[a] = beta[a]*comparision_distance[a]
        #     else:
        #         new_confidence[a] = beta[a]/comparision_distance[a]
        #     a = a+1     
        # print(comparision_distance)
        while a < no_of_arms:
            if (rank1[a])<=(np.int(np.log2(no_of_arms))):
                # if comparision_distance[a] > 0.3 :
                # new_confidence[a] = beta[a]/(comparision_distance[a]*distance_from_optimal[a]*distance_from_optimal[a])
                new_confidence[a] = beta[a]*(comparision_distance[a])/(distance_from_optimal[a]*distance_from_optimal[a])
            # else:
            #     new_confidence[a] = beta[a]*comparision_distance[a]
            a = a+1   

        
        # i = 0
        # while i<no_of_arms:
        #     j = 0
        #     while j<no_of_arms:
        #         if thompson_confidence[i] == 
        #         j = j+1

        #     i = i+1


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
        
        expected_reward= expected_reward + arm_reward
        reward[t]= expected_reward
        t = t+1


    # print(success)
    # print(failure)
    return reward



horizons = np.zeros(1600)
i = 0
while i<1600:
    horizons[i] = i+1
    i = i+1






instance1 = [0.4,0.8]
instance2 = [0.4,0.3,0.5,0.2,0.1]
instance3 = [0.15,0.23,0.37,0.44,0.50,0.32,0.78,0.21,0.82,0.56,0.34,0.56,0.84,0.76,0.43,0.65,0.73,0.92,0.10,0.89,0.48,0.96,0.60,0.54,0.49]










bandit_instance =instance1
#######epsilon greedy 

avg_reward_ucb = np.zeros(1600)
i = 0
while i<50:
    avg_reward_ucb = avg_reward_ucb + thompson_sampling_with_hint(bandit_instance,i,2,1600)
    i = i+1

avg_reward_ucb = avg_reward_ucb*(1.0/50.0)

avg_reward_thompson = np.zeros(1600)
i = 0
while i<50:
    avg_reward_thompson = avg_reward_thompson + thompson_sampling(bandit_instance,i,2,1600)
    i = i+1
avg_reward_thompson = avg_reward_thompson*(1.0/50.0)

 
np.savetxt('thompson_with_hint_1_1_test.txt',avg_reward_ucb)  
np.savetxt('thompson_orig_1_1_test.txt',avg_reward_thompson)  























bandit_instance =instance2
#######epsilon greedy 

avg_reward_ucb = np.zeros(1600)
i = 0

i = 0
while i<50:
    avg_reward_ucb = avg_reward_ucb + thompson_sampling_with_hint(bandit_instance,i,5,1600)
    i = i+1

avg_reward_ucb = avg_reward_ucb*(1.0/50.0)

avg_reward_thompson = np.zeros(1600)
i = 0
while i<50:
    avg_reward_thompson = avg_reward_thompson + thompson_sampling(bandit_instance,i,5,1600)
    i = i+1
avg_reward_thompson = avg_reward_thompson*(1.0/50.0)






  
np.savetxt('thompson_with_hint_2_2_test.txt',avg_reward_ucb)  
np.savetxt('thompson_orig_2_2_test.txt',avg_reward_thompson)  

bandit_instance =instance3
#######epsilon greedy 



avg_reward_ucb = np.zeros(1600)
i = 0
while i<50:
    avg_reward_ucb = avg_reward_ucb + thompson_sampling_with_hint(bandit_instance,i,np.int(len(bandit_instance)),1600)
    i = i+1

avg_reward_ucb = avg_reward_ucb*(1.0/50.0)

avg_reward_thompson = np.zeros(1600)
i = 0
while i<50:
    avg_reward_thompson = avg_reward_thompson + thompson_sampling(bandit_instance,i,np.int(len(bandit_instance)),1600)
    i = i+1
avg_reward_thompson = avg_reward_thompson*(1.0/50.0)





  
np.savetxt('thompson_with_hint_3_3_test.txt',avg_reward_ucb)  
np.savetxt('thompson_orig_3_3_test.txt',avg_reward_thompson)  