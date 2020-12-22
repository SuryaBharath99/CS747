import numpy as np
import sys




no_of_args = len(sys.argv)
for i in range(1,no_of_args):
    if sys.argv[i] == "--grid":
        i = i + 1
        input_grid_path  = sys.argv[i]


with open(input_grid_path) as inp:
    maze_data = inp.readlines()
    
maze_data = [x.strip() for x in maze_data] 
grid_row = np.int(len(maze_data))
a = maze_data[1].split()
grid_col = np.int(len(a))


# print(grid_row)
# print(grid_col)
i = 0
j = 0

grid = np.zeros((grid_row,grid_col))

for i in range(grid_row):
    b = maze_data[i].split()
    for j in range(grid_col):
        grid[i,j] = b[j] 
        j = j+1
    i = i +1

# print(grid)

########## transition and reward 

no_of_states = np.int((grid_row)*(grid_col))


print("numStates" , " " ,no_of_states)

print("numActions" , " " ,4)

start = 0
end = 0

for k in range(no_of_states):
    row =np.int(k / (grid_col))
    col = k-row*(grid_col) 
    if (1<=row<=grid_row-2 and 1<=col<=grid_col-2):
        if grid[row,col] == 2:
            start = k            
            print("start" , " " ,k)
        if grid[row,col] == 3:
            end = k          
            print("end" , " " ,k)    


  




for k in range(no_of_states):
    # transition_mat = np.zeros((grid_row,grid_col))
    # reward_mat = np.zeros((grid_row,grid_col))
    row =np.int(k / (grid_col))
    col = k-(row*(grid_col))
    # print(row,col)
    #North = same column, 1 row less
    #South = same column, 1 row more
    #East = same row , 1 column more
    #West = same row, 1 column less

    if (1<=row<=grid_row-2 and 1<=col<=grid_col-2):
        en = 0
        num  = 0
        if grid[row-1,col] == 3:
            num = (row-1)*(grid_col)+col
            print("transition" + " " +str(k)+" " +str(0)+" "+str(num)+" "+ str(no_of_states)+" "+ str(1))
        if grid[row,col-1] == 3:
            num = row*(grid_col)+col-1 
            print("transition" + " " +str(k)+" " +str(3)+" "+str(num)+" "+ str(no_of_states)+" "+ str(1))
        if grid[row+1,col] == 3:
            num = (row+1)*(grid_col)+col
            print("transition" + " " +str(k)+" " +str(1)+" "+str(num)+" "+ str(no_of_states)+" "+ str(1))
        if grid[row,col+1] == 3:
            num = row*(grid_col)+col+1
            print("transition" + " " +str(k)+" " +str(2)+" "+str(num)+" "+ str(no_of_states)+" "+ str(1))

        if grid[row,col] != 3:

            if grid[row-1,col] == 0:
                num = (row-1)*(grid_col)+col
                print("transition" + " " +str(k)+" " +str(0)+" "+str(num)+" "+ str(-1.0/no_of_states)+" "+ str(1))
           
            if grid[row,col-1] == 0:
                num = row*(grid_col)+col-1
                print("transition" + " " +str(k)+" " +str(3)+" "+str(num)+" "+ str(-1.0/no_of_states)+" "+ str(1))


            if grid[row+1,col] == 0:
                num = (row+1)*(grid_col)+col 
                print("transition" + " " +str(k)+" " +str(1)+" "+str(num)+" "+ str(-1.0/no_of_states)+" "+ str(1))
            
            if grid[row,col+1] == 0:
                num = row*(grid_col)+col+1
                print("transition" + " " +str(k)+" " +str(2)+" "+str(num)+" "+ str(-1.0/no_of_states)+" "+ str(1)) 


            if grid[row-1,col] == 2:
                num = (row-1)*(grid_col)+col
                print("transition" + " " +str(k)+" " +str(0)+" "+str(num)+" "+ str(-1)+" "+ str(1))
           
            if grid[row,col-1] == 2:
                num = row*(grid_col)+col-1
                print("transition" + " " +str(k)+" " +str(3)+" "+str(num)+" "+ str(-1)+" "+ str(1))

            if grid[row+1,col] == 2:
                num = (row+1)*(grid_col)+col
                print("transition" + " " +str(k)+" " +str(1)+" "+str(num)+" "+ str(-1)+" "+ str(1))

            if grid[row,col+1] == 2:
                num = row*(grid_col)+col+1 
                print("transition" + " " +str(k)+" " +str(2)+" "+str(num)+" "+ str(-1)+" "+ str(1))

 
        # North
        
        # South

        # East

        # West

print("mdptype" + " " +"episodic")
print("discount" + " " +str(0.9999))


