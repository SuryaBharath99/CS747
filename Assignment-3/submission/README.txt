The only script to run is "windy_gridworld.py"

In the top section ( first few lines) of this "windy_gridworld.py" there are following tunable parameters :

gamma ( default is set to 1.0)
epsilon ( given as 0.1 in example 6.5) 
alpha ( given as 0.5 in example 6.5) -> Learning rate

One can change these above values to their required values while experimenting .

Describing the grid  : 
Number of rows in the grid ( set to 7)
Number of column in the grid ( set to 10)

Details of starting position
Start_column = column number of the start cell
Start_row = row number of the start cell

Details of ending position
End_column = column number of the end cell
End_row = row number of the end cell

Details of wind effect on grid ( wind_weights )
wind_weights is a vector conatining values of wind strength for each column.


The last part of "windy_gridworld.py" is plotting part.
This contains the code snippets to plot Episodes  vs Time stamps for each agent . 
So, one can get any combination of plots by very minimum and suitable modification (commenting) those lines .


