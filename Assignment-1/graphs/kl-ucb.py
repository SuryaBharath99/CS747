import numpy as np

np.set_printoptions(threshold=np.inf)
def kl_ucb(bandit_instance,RandomSeed,no_of_arms,horizon):
    reward = np.zeros(horizon)
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
        reward[a+no_of_arms] = expected_reward
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
        reward[t] = expected_reward
        t = t+1

    
    return reward



instance1 = [0.4,0.8]
instance2 = [0.4,0.3,0.5,0.2,0.1]
instance3 = [0.15,0.23,0.37,0.44,0.50,0.32,0.78,0.21,0.82,0.56,0.34,0.56,0.84,0.76,0.43,0.65,0.73,0.92,0.10,0.89,0.48,0.96,0.60,0.54,0.49]

bandit_instance =instance1
#######epsilon greedy 

avg_reward_kl_ucb = np.zeros(100)
i = 0
while i<50:
    avg_reward_kl_ucb = avg_reward_kl_ucb + kl_ucb(bandit_instance,i,2,100)
    i = i+1



avg_reward_kl_ucb= avg_reward_kl_ucb*(1.0/50.0)


file1 = open("kl_plots.txt","a")
file1.write("Thompson :") 
file1.close()
file1 = open("kl_plots.txt","a")
file1.write("\n") 
file1.close()
file1 = open("kl_plots.txt","a")
file1.write(str(avg_reward_kl_ucb)) 
file1.close() 


np.savetxt('kl_ucb_1_test.txt',avg_reward_kl_ucb)    



