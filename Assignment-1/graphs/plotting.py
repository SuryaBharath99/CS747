import numpy as np 
from matplotlib import pyplot as plt 







instance1 = [0.4,0.8]
instance2 = [0.4,0.3,0.5,0.2,0.1]
instance3 = [0.15,0.23,0.37,0.44,0.50,0.32,0.78,0.21,0.82,0.56,0.34,0.56,0.84,0.76,0.43,0.65,0.73,0.92,0.10,0.89,0.48,0.96,0.60,0.54,0.49]


max1 = max(instance1)
print(max1)
max2 = max(instance2)
max3 = max(instance3)
print(max3)


ucb1 = np.loadtxt('ucb_1.txt')
epsilon1 = np.loadtxt('epsilon_1.txt')
kl_ucb1 = np.loadtxt('kl_ucb_1.txt')
thompson1 = np.loadtxt('thompson_1.txt')

# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max1
#     horizons[i] = i+1
#     i = i+1

# Reget_calc = Reget_calc-epsilon1


# plt.label("UCB") 
# # plt.subplot(1,3,1)
# plt.xlabel("Horizon") 
# plt.ylabel("Average regret") 
# print(len(Reget_calc))
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label= 'Epsilon Greedy') 
# # plt.show()










# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max1
#     horizons[i] = i+1
#     i = i+1

# np.savetxt('testing.txt',Reget_calc)  
# Reget_calc = Reget_calc-ucb1
# # np.savetxt('testing.txt',thompson1)  


# plt.title("Instance 1") 
# plt.xlabel("Horizon") 
# plt.ylabel("Average regret") 
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label = 'UCB')
# # plt.legend() 



# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max1
#     horizons[i] = i+1
#     i = i+1

# np.savetxt('testing.txt',Reget_calc)  
# Reget_calc = Reget_calc-kl_ucb1
# # np.savetxt('testing.txt',thompson1)  


# plt.title("Instance 1") 
# plt.xlabel("Horizon") 
# plt.ylabel("Average regret") 
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label = 'KL UCB')
# plt.legend() 















# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max1
#     horizons[i] = i+1
#     i = i+1

# Reget_calc = Reget_calc-thompson1


# # plt.label("UCB") 
# # plt.subplot(1,3,2)
# plt.xlabel("Horizon (log scale base-10)") 
# plt.ylabel("Average regret (log scale base-10) ") 
# print(len(Reget_calc))
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label= 'Thompson Sampling') 
# # plt.show()

# plt.legend() 










# ucb1 = np.loadtxt('ucb_2.txt')
# epsilon1 = np.loadtxt('epsilon_2.txt')
# kl_ucb1 = np.loadtxt('kl_ucb_2.txt')
# thompson1 = np.loadtxt('thompson_2.txt')





# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max2
#     horizons[i] = i+1
#     i = i+1

# Reget_calc = Reget_calc-epsilon1


# # plt.label("UCB") 
# # plt.subplot(1,3,2)
# plt.xlabel("Horizon") 
# plt.ylabel("Average regret") 
# print(len(Reget_calc))
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label= 'Epsilon Greedy') 
# # plt.show()










# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max2
#     horizons[i] = i+1
#     i = i+1

# np.savetxt('testing.txt',Reget_calc)  
# Reget_calc = Reget_calc-ucb1
# # np.savetxt('testing.txt',thompson1)  


# plt.title("Instance 1") 
# plt.xlabel("Horizon") 
# plt.ylabel("Average regret") 
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label = 'UCB')
# plt.legend() 



# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max2
#     horizons[i] = i+1
#     i = i+1

# np.savetxt('testing.txt',Reget_calc)  
# Reget_calc = Reget_calc-kl_ucb1
# # np.savetxt('testing.txt',thompson1)  


# plt.title("Instance 2") 
# plt.xlabel("Horizon") 
# plt.ylabel("Average regret") 
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label = 'KL UCB')
# # plt.legend() 







# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max2
#     horizons[i] = i+1
#     i = i+1

# Reget_calc = Reget_calc-thompson1


# plt.xlabel("Horizon (log scale base-10)") 
# plt.ylabel("Average regret (log scale base-10) ") 
# print(len(Reget_calc))
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label= 'Thompson Sampling') 
# # plt.show()
# plt.legend() 



















# ucb1 = np.loadtxt('ucb_3.txt')
# epsilon1 = np.loadtxt('epsilon_3.txt')





# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max3
#     horizons[i] = i+1
#     i = i+1

# Reget_calc = Reget_calc-epsilon1


# # plt.label("UCB") 
# # plt.subplot(1,3,3)
# plt.xlabel("Horizon") 
# plt.ylabel("Average regret") 
# print(len(Reget_calc))
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label= 'Epsilon Greedy') 
# # plt.show()










# Reget_calc = np.zeros(1600)
# horizons = np.zeros(1600)
# i = 0 

# while i< 1600:
#     Reget_calc[i] = (i+1)*max3
#     horizons[i] = i+1
#     i = i+1

# np.savetxt('testing.txt',Reget_calc)  
# Reget_calc = Reget_calc-ucb1
# # np.savetxt('testing.txt',thompson1)  


# plt.title("Instance 1") 
# plt.xlabel("Horizon") 
# plt.ylabel("Average regret") 
# plt.plot(np.log10(horizons),np.log10(Reget_calc),label = 'UCB')
# plt.legend() 








kl_ucb1 = np.loadtxt('thompson_with_hint_1_1_test.txt')
thompson1 = np.loadtxt('thompson_orig_1_1_test.txt')
Reget_calc = np.zeros(1600)
horizons = np.zeros(1600)
i = 0 

while i< 1600:
    Reget_calc[i] = (i+1)*max1
    horizons[i] = i+1
    i = i+1

np.savetxt('testing.txt',Reget_calc)  
Reget_calc = Reget_calc-kl_ucb1
# np.savetxt('testing.txt',thompson1)  

plt.subplot(1,3,1)
plt.title("Instance 1") 
plt.xlabel("Horizon") 
plt.ylabel("Average regret") 
plt.plot(horizons,Reget_calc,label = 'Thompson Sampling with hint')
# plt.legend() 





Reget_calc = np.zeros(1600)
horizons = np.zeros(1600)
i = 0 

while i< 1600:
    Reget_calc[i] = (i+1)*max1
    horizons[i] = i+1
    i = i+1

Reget_calc = Reget_calc-thompson1


# plt.label("UCB") 
# plt.subplot(1,3,2)
plt.xlabel("Horizon ") 
plt.ylabel("Average regret ") 
print(len(Reget_calc))
plt.plot(horizons,Reget_calc,label= 'Thompson Sampling') 
# plt.show()
plt.legend() 


















kl_ucb1 = np.loadtxt('thompson_with_hint_2_2_test.txt')
thompson1 = np.loadtxt('thompson_orig_2_2_test.txt')
Reget_calc = np.zeros(1600)
horizons = np.zeros(1600)
i = 0 

while i< 1600:
    Reget_calc[i] = (i+1)*max2
    horizons[i] = i+1
    i = i+1

np.savetxt('testing.txt',Reget_calc)  
Reget_calc = Reget_calc-kl_ucb1
# np.savetxt('testing.txt',thompson1)  

plt.subplot(1,3,2)
plt.title("Instance 2") 
plt.xlabel("Horizon") 
plt.ylabel("Average regret") 
plt.plot(horizons,Reget_calc,label = 'Thompson Sampling with hint')
# plt.legend() 





Reget_calc = np.zeros(1600)
horizons = np.zeros(1600)
i = 0 

while i< 1600:
    Reget_calc[i] = (i+1)*max2
    horizons[i] = i+1
    i = i+1

Reget_calc = Reget_calc-thompson1


# plt.label("UCB") 
# plt.subplot(1,3,2)
plt.xlabel("Horizon ") 
plt.ylabel("Average regret ") 
print(len(Reget_calc))
plt.plot(horizons,Reget_calc,label= 'Thompson Sampling') 
# plt.show()
plt.legend() 







































kl_ucb1 = np.loadtxt('thompson_with_hint_3_3_test.txt')
thompson1 = np.loadtxt('thompson_orig_3_3_test.txt')
Reget_calc = np.zeros(1600)
horizons = np.zeros(1600)
i = 0 

while i< 1600:
    Reget_calc[i] = (i+1)*max3
    horizons[i] = i+1
    i = i+1

np.savetxt('testing.txt',Reget_calc)  
Reget_calc = Reget_calc-kl_ucb1
# np.savetxt('testing.txt',thompson1)  
plt.subplot(1,3,3)

plt.title("Instance 3") 
plt.xlabel("Horizon") 
plt.ylabel("Average regret") 
plt.plot(horizons,Reget_calc,label = 'Thompson Sampling with hint')
# plt.legend() 





Reget_calc = np.zeros(1600)
horizons = np.zeros(1600)
i = 0 

while i< 1600:
    Reget_calc[i] = (i+1)*max3
    horizons[i] = i+1
    i = i+1

Reget_calc = Reget_calc-thompson1


# plt.label("UCB") 
# plt.subplot(1,3,2)
plt.xlabel("Horizon ") 
plt.ylabel("Average regret ") 
print(len(Reget_calc))
plt.plot(horizons,Reget_calc,label= 'Thompson Sampling') 
# plt.show()
plt.legend() 


plt.show()




