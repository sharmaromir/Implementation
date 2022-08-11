import numpy as np
import sys
import copy
import pandas as p
import matplotlib.pyplot as plt
import random


arena = [[-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1], 
    [-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1],
    [-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1,-.1],
    [-.1, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 100]] #rewards of the environment

move = { #mapping coordinates to actions
    '[-1, 0]' : 0,
    '[1, 0]' : 1,
    '[0, -1]' : 2,
    '[0, 1]' : 3
}

inv_move = { #reverse mapping actions to coordinates
    0 : [-1,0],
    1 : [1,0],
    2 : [0,-1],
    3 : [0, 1]
}

def Dr(noise:float) -> tuple: #since this is a simulation, we can skip the data collection phases and manually input state distributions
    return [1-(noise*3/4), noise*1/4, noise*1/4, noise*1/4], \
            [noise*1/4, 1-(noise*3/4), noise*1/4, noise*1/4], \
            [noise*1/4, noise*1/4, 1-(noise*3/4), noise*1/4], \
            [noise*1/4, noise*1/4, noise*1/4, 1-(noise*3/4)]

def Ds() -> tuple: #deterministic environment
    return [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]

def f_g(s:list, a:int, noise:float) -> list: #returns the most likely state as gat would
    distribution = Dr(noise)
    chance:tuple = distribution[a]
    choice = chance.index(max(chance))
    action = inv_move[choice]
    return [s[0]+action[0], s[1]+action[1]]

def f_inv_g(s:list, s_:list) -> int: #gets the best action to map to the new state
    action = [s_[0]-s[0], s_[1]-s[1]]
    choice = move[str(action)]
    return choice

def g_g(s:list, a:int, noise:float) -> list: #essentially serves as a map for the policy
    mainPoint = f_inv_g(s, f_g(s, a, noise))
    res = [0]*4
    res[mainPoint] = 1
    return res

def f_s(s:list, a:int, noise:float) -> list: #returns a distribution of all possible future states
    distribution = Dr(noise)
    dist = []
    for i in range(4):
        action = inv_move[i]
        chance = distribution[a][i]
        dist.append([s[0]+action[0], s[1]+action[1], chance])
    return dist

def f_inv_s(s:list, future:list) -> list: #returns a distribution of possible actions
    res = [0]*4
    for i in range(len(future)):
        action = move[str([future[i][0]-s[0], future[i][1]-s[1]])]
        res[action]+=future[i][2]
    return res

def g_s(s:list, a:int, noise:float) -> list: #gets the distribution of future states when choosing an action at a state
    return f_inv_s(s, f_s(s, a, noise))

def optimize(noise:float, stochastic:bool, gamma:float) -> list: #uses policy iteration with bounds of 1000 for 
    p_steps:int=100000 #termination point for the most times the policy can change
    v_steps:int=100000 #termination point for the most times the value van be updated
    delta:int = 0.000001
    fPar = len(arena) #first parameter, top to bottom of the arena
    sPar = len(arena[0]) #second parameter, <-> of the arena
    #pi:list=[[1,1,1,1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1,1,1,1], [3,3,3,3,3,3,3,3,3,3,3,3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] #starting policy, set all actions to up
    pi:list = [[0 for i in range(sPar)] for j in range(fPar)]
    V:list=copy.deepcopy(arena) #starting values set to 0
    

    for i in range(p_steps): #repeat as long as the policy hasn't exceeded its limit
        optimal = True

        for j in range(v_steps): #repeat as long as the value hasn't exceeded its limit, policy evaluation
            max_diff = 0
            for r in range(fPar):
                for c in range(sPar):

                    val = arena[r][c]
                    if(stochastic): dist = g_s([r,c], pi[r][c], noise) #use sgat to find a distribution of states
                    else: dist = g_g([r,c], pi[r][c], noise)  #gat for a deterministic state 

                    for actions in range(len(dist)): #map through all actions in a spot
                        actTran = inv_move[actions] #convert the move to coordinates

                        s_ = [r+actTran[0], c+actTran[1]] #next state 

                        if(s_[0]<0): s_[0]=0 #out of bounds corrections
                        if(s_[0]>=fPar): s_[0] = fPar-1
                        if(s_[1]<0): s_[1]=0
                        if(s_[1]>=sPar): s_[1] = sPar-1
                        #print(str(s_[0]) + " " + str(s_[1]) + " " + str(len(V[0])) + " " + str(len(V)))
                        val+=dist[actions]*gamma*V[s_[0]][s_[1]]
                    max_diff = max(max_diff, abs(val - V[r][c]))
                    V[r][c] = val
            if max_diff<delta:
                break
        
        for r in range(fPar): #policy improvemnet
            for c in range(sPar):
                
                maxVal = V[r][c]

                for actionList in range(len(dist)):
                    if(stochastic): dist = g_s([r,c], actionList, noise) #use sgat to find a distribution of states
                    else: dist = g_g([r,c], actionList, noise)  #gat for a deterministic state 
                    
                    val = arena[r][c]
                    for actions in range(len(dist)): #map through all actions in a spot
                        actTran = inv_move[actions] #convert the move to coordinates

                        s_ = [r+actTran[0], c+actTran[1]] #next state 

                        if(s_[0]<0): s_[0]=0 #out of bounds corrections
                        if(s_[0]>=fPar): s_[0] = fPar-1
                        if(s_[1]<0): s_[1]=0
                        if(s_[1]>=sPar): s_[1] = sPar-1
                        val+=dist[actions]*gamma*V[s_[0]][s_[1]]

                    if val > maxVal and pi[r][c] != actionList: #update the policy at the spot
                        pi[r][c] = actionList
                        maxVal = val
                        optimal = False
        #print("finished iteration " + str(i)) #counter
        if optimal:
            break
    
    return pi #return optimal policy

def play(l, d):
    sgat_x = []
    sgat_y = []
    gat_x = []
    gat_y = []

    gamma = 0.9
    for i in range(l+1):
        print(i)
        rewardS = 0
        rewardG = 0
        #gamma = max(gamma, i/(l+1))
        oPo = optimize(i/l, False, gamma)
        nPo = optimize(i/l, True, gamma)
        for j in range(d): #repeat d iterations
            rewardS += test(nPo, 3, 0, 0, i/l) #add the reward for sgat
            rewardG += test(oPo, 3, 0, 0, i/l) #add the reward for gat
        
        sgat_x.append(rewardS/d)
        sgat_y.append(i/(l))
        gat_x.append(rewardG/d)
        gat_y.append(i/(l))
    display(sgat_x, sgat_y, gat_x, gat_y)
    return

def test(policy, x, y, reward, noise): #carries out a trajectory
    action = policy[x][y]
    if(np.random.random_sample()<noise):
        action = random.choice([2,3,0,1])
    if(action==2):
        y = y-1 if y>0 else y
    elif(action==3):
        y = y+1 if y<len(policy[0])-1 else y
    elif(action==0):
        x = x-1 if x>0 else x
    elif(action==1):
        x = x+1 if x<len(policy)-1 else x
    reward += arena[x][y]
    if arena[x][y]==-10 or arena[x][y]==100 or reward<-50:
        return reward
    return test(policy, x, y, reward, noise)

def display(y1, x1, y2, x2):

    plt.plot(x2, y2, label = "GAT")
    plt.plot(x1, y1, label = "SGAT")

    plt.xlabel('Environment Noise')

    plt.ylabel('Expected Return')
    
    plt.title('Comparing Distributional and Deterministic Algorithms')
    plt.legend()
    plt.show()
    return

sys.setrecursionlimit(2000)
play(int(sys.argv[1]), int(sys.argv[2]))
#print(str(p.DataFrame(optimize(0.0, True, 0.9))) + "\n\n")
#print(str(p.DataFrame(optimize(0.0, False, 0.9))) + "\n\n")
#print(str(p.DataFrame(optimize(0.3, True, 0.61))) + "\n\n")
#print(str(p.DataFrame(optimize(0.3, False, 0.61))) + "\n\n")

