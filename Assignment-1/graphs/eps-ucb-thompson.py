import numpy as np
import sys
np.set_printoptions(threshold=np.inf)


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
    horizons = np.zeros(horizon)
    reward = np.zeros(horizon)
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
        horizons[t] = t+1
        reward[t] = expected_reward 
        t = t+1


    return reward 





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
    reward = np.zeros(horizon)
    while a<no_of_arms:
        arm_prob = bandit_instance[a]
        arm_reward = np.random.binomial(1,arm_prob,1)
        arm_reward = np.int(arm_reward)

        emp_mean[a] = (((emp_mean[a])*np.float(pulls_per_arms[a]))+arm_reward+0.0)/(pulls_per_arms[a]+1.0)
        
        pulls_per_arms[a] = pulls_per_arms[a]+1
            
        expected_reward = expected_reward + arm_reward
        reward[a] = expected_reward
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
        reward[a+no_of_arms] =expected_reward
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
        reward[t] = expected_reward
        t = t+1

    
    return reward





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
    reward = np.zeros(horizon)
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
    
    # print(pulls_per_arms)
    # a = 0
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


horizons = np.zeros(102400)
i = 0
while i<102400:
    horizons[i] = i+1
    i = i+1






instance1 = [0.4,0.8]
instance2 = [0.4,0.3,0.5,0.2,0.1]
instance3 = [0.15,0.23,0.37,0.44,0.50,0.32,0.78,0.21,0.82,0.56,0.34,0.56,0.84,0.76,0.43,0.65,0.73,0.92,0.10,0.89,0.48,0.96,0.60,0.54,0.49]

bandit_instance =instance1
#######epsilon greedy 

avg_reward_epsilon = np.zeros(102400)
i = 0
while i<50:
    avg_reward_epsilon = avg_reward_epsilon + epsilon_greedy(bandit_instance,i,2,0.98,102400)
    i = i+1


avg_reward_epsilon = avg_reward_epsilon*(1.0/50.0)




avg_reward_ucb = np.zeros(102400)
i = 0
while i<50:
    avg_reward_ucb = avg_reward_ucb + ucb(bandit_instance,i,2,102400)
    i = i+1

avg_reward_ucb = avg_reward_ucb*(1.0/50.0)






avg_reward_thompson = np.zeros(102400)
i = 0
while i<50:
    avg_reward_thompson = avg_reward_thompson + thompson_sampling(bandit_instance,i,2,102400)
    i = i+1
avg_reward_thompson = avg_reward_thompson*(1.0/50.0)








file1 = open("plots.txt","a")
file1.write("Order is epsilon, ucb, thompson"+"\n") 
file1.close() 


file1 = open("plots.txt","a")
file1.write("For instance 3:"+"\n") 
file1.close() 

file1 = open("plots.txt","a")
file1.write("Epsilon :") 
file1.close()

file1 = open("plots.txt","a")
file1.write(str(avg_reward_epsilon)) 
file1.close() 





file1 = open("plots.txt","a")
file1.write("\n") 
file1.close()

file1 = open("plots.txt","a")
file1.write("UCB :") 
file1.close()
file1 = open("plots.txt","a")
file1.write(str(avg_reward_ucb)) 
file1.close() 





file1 = open("plots.txt","a")
file1.write("\n") 
file1.close()
file1 = open("plots.txt","a")
file1.write("Thompson :") 
file1.close()

file1 = open("plots.txt","a")
file1.write(str(avg_reward_thompson)) 
file1.close() 


np.savetxt('epsilon_1.txt',avg_reward_epsilon)  
np.savetxt('ucb_1.txt',avg_reward_ucb)  
np.savetxt('thompson_1.txt',avg_reward_thompson)  























bandit_instance =instance2
#######epsilon greedy 

avg_reward_epsilon = np.zeros(102400)
i = 0
while i<50:
    avg_reward_epsilon = avg_reward_epsilon + epsilon_greedy(bandit_instance,i,5,0.98,102400)
    i = i+1


avg_reward_epsilon = avg_reward_epsilon*(1.0/50.0)


avg_reward_ucb = np.zeros(102400)
i = 0
while i<50:
    avg_reward_ucb = avg_reward_ucb + ucb(bandit_instance,i,5,102400)
    i = i+1

avg_reward_ucb = avg_reward_ucb*(1.0/50.0)

avg_reward_thompson = np.zeros(102400)
i = 0
while i<50:
    avg_reward_thompson = avg_reward_thompson + thompson_sampling(bandit_instance,i,5,102400)
    i = i+1
avg_reward_thompson = avg_reward_thompson*(1.0/50.0)









file1 = open("plots.txt","a")
file1.write("Order is epsilon, ucb, thompson"+"\n") 
file1.close() 


file1 = open("plots.txt","a")
file1.write("For instance 3:"+"\n") 
file1.close() 

file1 = open("plots.txt","a")
file1.write("Epsilon :") 
file1.close()

file1 = open("plots.txt","a")
file1.write(str(avg_reward_epsilon)) 
file1.close() 

file1 = open("plots.txt","a")
file1.write("\n") 
file1.close()

file1 = open("plots.txt","a")
file1.write("UCB :") 
file1.close()
file1 = open("plots.txt","a")
file1.write(str(avg_reward_ucb)) 
file1.close() 

file1 = open("plots.txt","a")
file1.write("\n") 
file1.close()
file1 = open("plots.txt","a")
file1.write("Thompson :") 
file1.close()

file1 = open("plots.txt","a")
file1.write(str(avg_reward_thompson)) 
file1.close() 




np.savetxt('epsilon_2.txt',avg_reward_epsilon)  
np.savetxt('ucb_2.txt',avg_reward_ucb)  
np.savetxt('thompson_2.txt',avg_reward_thompson)  

bandit_instance =instance3
#######epsilon greedy 

avg_reward_epsilon = np.zeros(102400)
i = 0
while i<50:
    avg_reward_epsilon = avg_reward_epsilon + epsilon_greedy(bandit_instance,i,np.int(len(bandit_instance)),0.98,102400)
    i = i+1


avg_reward_epsilon = avg_reward_epsilon*(1.0/50.0)


avg_reward_ucb = np.zeros(102400)
i = 0
while i<50:
    avg_reward_ucb = avg_reward_ucb + ucb(bandit_instance,i,np.int(len(bandit_instance)),102400)
    i = i+1

avg_reward_ucb = avg_reward_ucb*(1.0/50.0)

avg_reward_thompson = np.zeros(102400)
i = 0
while i<50:
    avg_reward_thompson = avg_reward_thompson + thompson_sampling(bandit_instance,i,np.int(len(bandit_instance)),102400)
    i = i+1
avg_reward_thompson = avg_reward_thompson*(1.0/50.0)





file1 = open("plots.txt","a")
file1.write("Order is epsilon, ucb, thompson"+"\n") 
file1.close() 


file1 = open("plots.txt","a")
file1.write("For instance 3:"+"\n") 
file1.close() 

file1 = open("plots.txt","a")
file1.write("Epsilon :") 
file1.close()

file1 = open("plots.txt","a")
file1.write(str(avg_reward_epsilon)) 
file1.close() 

file1 = open("plots.txt","a")
file1.write("\n") 
file1.close()

file1 = open("plots.txt","a")
file1.write("UCB :") 
file1.close()
file1 = open("plots.txt","a")
file1.write(str(avg_reward_ucb)) 
file1.close() 

file1 = open("plots.txt","a")
file1.write("\n") 
file1.close()
file1 = open("plots.txt","a")
file1.write("Thompson :") 
file1.close()

file1 = open("plots.txt","a")
file1.write(str(avg_reward_thompson)) 
file1.close() 


np.savetxt('epsilon_3.txt',avg_reward_epsilon)  
np.savetxt('ucb_3.txt',avg_reward_ucb)  
np.savetxt('thompson_3.txt',avg_reward_thompson)  