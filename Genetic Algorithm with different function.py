import random
import pandas as pd
import matplotlib.pyplot as plt
import math


def eqn(x,y):
    return -7*(x**2) + 3*x*(math.sin(y))-786*y+989
    # return (11*x) - (7.59*y)

def list_initialze():
    # Initialize a 3x10 list of zeros
    zero_matrix = []
    for _ in range(rows):
        row = [0] * cols
        zero_matrix.append(row)
    
    return zero_matrix
    



# Function to find the best fitness
def best_fit(lis):
    global df
    temp_list=[]
    for i in range(rows):
        temp_list.append(lis[i][2])
#     print(temp_list)
    best=max(temp_list)
    next_gen = len(df) + 1
    df.loc[next_gen] = [next_gen, best]
    # print(best)
    return best


# Check Mutated value if it grater or lesser than value
def mutated_y(value):
    if value>y_upper:
        return y_upper
    elif value<y_lower:
        return y_lower
    else:
        return value

    
# Check Mutated value if it grater or lesser than value
def mutated_x(value):
    if value>x_upper:
        return x_upper
    elif value<x_lower:
        return x_lower
    else:
        return value
    

def sort_des(I,C):
    new_list=I+C
    sorted_list = sorted(new_list, key=lambda x: x[2], reverse=True)
    I[:10]=sorted_list[:10]
    # print(I)
    return I

def print_x_y():    
    # Specify the column for which you want to access the last value
    column_name = "Best Fitness"

    # Access the last value of the specified column
    target_value =  df[column_name].iloc[-1]

    matching_sublist = None
    for sublist in C:
        if sublist[-1] == target_value:
            matching_sublist = sublist
            break

    if matching_sublist is not None:
        print("Matching sublist found:", matching_sublist)
    else:
        print("No matching sublist found.")    

    
def plot_graph():    
    x_column = 'Generation'
    y_column = 'Best Fitness'

    fig, ax = plt.subplots(figsize=(15, 7))
    # Plot a line graph
    df.plot(x=x_column, y=y_column, kind='line', ax=ax)

    # Add labels and title
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Line Graph between {x_column} and {y_column}')


    # Show the plot
    plt.show()
def I_initialze(I):
    for i in range(rows):
        for j in range(cols):
            if j==0:
                # Generate a random floating-point number between 1.0 and 10.0 (inclusive of 1.0, exclusive of 10.0)
                rand_x = random.uniform(x_lower,x_upper +1)
                I[i][j]=rand_x
            elif j==1:
                rand_y = random.uniform(y_lower,y_upper+1)
                I[i][j] = rand_y
            else:
                rand_x = I[i][0]
                rand_y = I[i][1]
                I[i][j] = eqn(rand_x,rand_y)
    # print(I)
                


def main_working(I, C):
    global df
    best_val=1
    cols = len(C[0])  
    count=0
    pre_aveg=0
    while best_val > 0.001:
    # for _ in range(60):
        for i in range(0, 10):  # Updated to cover the entire range
            index_i = random.randint(0, 9)
            index_j = random.randint(0, 9)
            # print(index_i,index_j)
            for j in range(cols):
                if j == 0:
                    avr_val=(I[index_i][j]+I[index_j][j])/2
                    C[i][j] = avr_val
                    # C[i + 1][j] = I[index_j][j]
                elif j == 1:
                    avr_val=(I[index_i][j]+I[index_j][j])/2
                    C[i][j] = avr_val
                else:
                    mutate_pos = 0.15
                    mutate_neg = -0.15
                    mutate_if = random.randint(0, 100)
                    if mutate_if >= 50:  # Mutation Possible
                        mutate_pos_neg = random.randint(0, 100)
                        if mutate_pos_neg >= 50:  # Positive Mutation
                            mutate_x_y = random.randint(0, 100)
                            if mutate_x_y >= 50:  # y mutate
                                C[i][1] = mutated_y(C[i][1] + mutate_pos)
                            else:  # x Mutate
                                C[i][0] = mutated_x(C[i][0] + mutate_pos)
                        else:  # Negative mutate
                            mutate_x_y = random.randint(0, 100)
                            if mutate_x_y >= 50:  # y mutate
                                C[i][1] = mutated_y(C[i][1] + mutate_neg)
                            else:  # x Mutate
                                C[i][0] = mutated_x(C[i][0] + mutate_neg)
                    val_i_x = C[i][0]
                    val_i_y = C[i][1]
                    C[i][2] = eqn(val_i_x, val_i_y)

        best_fit(C)
        I = sort_des(I, C)
        count+=1
        if count==10:
            last_10 = sum(df['Best Fitness'].tail(10).tolist())
            best_val=last_10-pre_aveg
            pre_aveg=last_10
            count=0
            print(last_10)


        
        # print(I)
        
# Setting a seed
random.seed(650) #650
x_lower= -15
x_upper= 20
y_lower= -20
y_upper= 25
rows = 10
cols = 3  
columns = ['Generation', 'Best Fitness']
df= pd.DataFrame(columns=columns)     
I=list_initialze()
C=list_initialze()    
I_initialze(I)
main_working(I,C)
print_x_y()
plot_graph()
df.to_csv('output_file.csv', index=False)