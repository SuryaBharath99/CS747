import numpy as np
import sys




no_of_args = len(sys.argv)
for i in range(1,no_of_args):
    if sys.argv[i] == "--grid":
        i = i + 1
        input_grid_path  = sys.argv[i]
    elif sys.argv[i] == "--value_policy":
        i = i+1
        policy =sys.argv[i]    




with open(input_grid_path) as inp:
    maze_data = inp.readlines()    
maze_data = [x.strip() for x in maze_data]

grid_row = np.int(len(maze_data))
a = maze_data[1].split()
grid_col = np.int(len(a))

i = 0
j = 0

grid = np.zeros((grid_row,grid_col))

for i in range(grid_row):
    b = maze_data[i].split()
    for j in range(grid_col):
        grid[i,j] = b[j] 
        j = j+1
    i = i +1


start = 0
end = 0
no_of_states = np.int((grid_row)*(grid_col))

for k in range(no_of_states):
    row =np.int(k / (grid_col))
    col = k-row*(grid_col) 
    if (1<=row<=grid_row-2 and 1<=col<=grid_col-2):
        if grid[row,col] == 2:
            start = k            
        if grid[row,col] == 3:
            end = k             











with open(policy) as inp:
    policy_data = inp.readlines()
    
policy_data = [x.strip() for x in policy_data]


# print(policy_data[0])
# print(policy_data[1])


path = []

# print(policy_data)

b = 0
a1 = 0
b1 = 0
s = start
s = np.int(s)

end = np.int(end)
# print(start)
out = str('')

while s!= end:
    f =2*(s)
    a1 = policy_data[f].split()
    # print(a1)
    b2 = a1[1]
    #North
    # print(type(b1))
    b2 =b2[0]
    b1 = int(b2)

    if b1 == 0:
        s= s-grid_col
        path.append('N')
        out = out+str('N ')
        # print(s)
    if b1 == 1:
        s = s+grid_col
        path.append('S')
        out = out+str('S ')
        # print(s)
    if b1 == 2:
        s = s+1
        path.append('E')
        out = out+str('E ')
        # print(s)
    if b1 == 3:
        s = s-1
        # print(s)
        path.append('W')
        out = out+str('W ')
    s =np.int(s)

    # a = policy_data[k].split()
    # b = a[1]
# i = np.int(len(out))    
# out = out - str(out[i-1])
print(out[:-1])
