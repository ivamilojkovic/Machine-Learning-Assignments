import numpy as np
import copy
       
def learning_rate(alpha, episode, adaptive = False):
    
    if adaptive==True:
        new_alpha = np.log(episode + 1)/(episode + 1)
    else:
        new_alpha = alpha
        
    return new_alpha

       
class Environment():
    
   def __init__(self, transition_prob):
       
       # Dimenzije prostora
       self.nrows = 3
       self.ncolumns = 4
       
       # Verovatnoce prelaza za jedno stanje
       self.prob = transition_prob
       
       # Prostor kretanja  
       self.space = np.ones((self.nrows, self.ncolumns))
       self.space[1,1] = 0
       
       # Lista stanja 
       self.states = []
       for i in range(self.nrows):
           for j in range(self.ncolumns):
               if (i,j) != (1,1):
                   self.states.append((i, j))
       
       # Lista svih akcija 
       self.all_actions = ['up', 'right', 'down', 'left']
       
       # Matrica nagrada
       self.rewards = -0.04*self.space
       self.rewards[0,3] = 1
       self.rewards[1,3] = -1
       
       # Cuvanje frejmova za V i Q vrednosti i za politiku
       self.frames = []
       self.all_pi = []
       self.Q_frames = []
       
       # Matrica V vrednosti
       self.V = np.zeros((self.nrows, self.ncolumns))
       self.V[1,1] = 2
       
       # 3D Matrica Q vrednosti
       self.Q = np.zeros((self.nrows, self.ncolumns, len(self.all_actions)))
       self.Q[1,1,:] = 2
       
       
   def sample(self, action):
        
        ind = self.all_actions.index(action)
        
        prob = np.random.rand()
        if prob <= self.prob[0]:
            action = action  
            P = self.prob[0]
            
        elif prob <= self.prob[0]+(1-self.prob[0])/2:
            action = self.all_actions[ind-1] 
            P = self.prob[1]
            
        elif prob <= 1:
            action = self.all_actions[(ind+1) % len(self.all_actions)]
            P = self.prob[2]
            
        return action, P    


class Agent():
    
    def __init__(self, env, start_state):
        
        self.env = env
        self.current_state = start_state
        self.current_reward = -0.04
        self.end = False
        
        self.num_states = sum(self.env.space)
        self.num_actions = len(self.env.all_actions)
        
    
    
    def next_state(self, state, action):
        
        current_state = self.current_state

        if action=='up' and state[0]!=0 and state!=(2,1):
            current_state = (state[0]-1, state[1])
            
        elif action=='down' and state[0]!=2 and state!=(0,1):
            current_state = (state[0]+1, state[1])
            
        elif action=='left' and state[1]!=0 and state!=(1,2):
            current_state = (state[0], state[1]-1)
            
        elif action=='right' and state[1]!=3 and state!=(1,0):
            current_state = (state[0], state[1]+1)
                  
        return current_state
    
    
    def probability(self, state, real_action, new_state):
        
        # Na pocetku nije pronadjena akcija kojom se stize u novo stanje
        exists = False
        
        # Stanja nakon akcija ['up', 'right', 'down', 'left'] redom      
        for a in self.env.all_actions:
            
            if new_state == self.next_state(state, a):
                
                # Moguce je stici sa akcijom a u novo stanje
                possible_action = a
                exists = True 
                break
         
        # Provera da li je uopste moguce stici u novo stanje    
        if exists == False:
            return 0
        
        # Pozicija zadate (prave) akcije u listi akcija
        indx_real = self.env.all_actions.index(real_action)
        
        # Poklapanje akcija 
        if possible_action == real_action:
            return self.env.prob[0]
        
        elif possible_action == self.env.all_actions[(indx_real+1) % self.num_actions]:
            return self.env.prob[1]
        
        elif possible_action == self.env.all_actions[(indx_real-1) % self.num_actions]:
            return self.env.prob[2]
        
        elif possible_action == self.env.all_actions[(indx_real-2) % self.num_actions]:
            return 0

     
    def Q_iteration(self, gamma, tol):
        
        self.env.all_pi = []
        self.env.frames = []
        self.env.Q_frames = []     
        self.env.gamma = gamma

        self.env.epsilon = None
        self.env.adapt = None
        self.env.alpha = None
        
        
        Q_new = copy.deepcopy(self.env.Q)
        Q_old = copy.deepcopy(self.env.Q)
        R = self.env.rewards
        
        i = 0
        while(1):
            
            for s in self.env.states:
                
                for a_ind, a in enumerate(self.env.all_actions):
                    
                    Q_new[s[0], s[1], a_ind] = R[s]
                    
                    if s == (0,3) or s == (1,3):
                        continue
                    
                    temp1 = 0
                    for new_s in self.env.states:
                        
                        P_sa = self.probability(s, a, new_s)
                        temp1 += P_sa * np.max(Q_old[new_s[0], new_s[1], :])
                        
                    Q_new[s[0], s[1], a_ind] += gamma*temp1
                    
            self.env.Q_frames.append(copy.deepcopy(Q_new))
            self.env.frames.append(self.V_from_Q(Q_new))
            self.env.all_pi.append(copy.deepcopy(self.Q_policy(Q_new)))
                 
            i += 1
            if np.max(abs(Q_new - Q_old)) < tol:
                
                print('Q je konvergiralo u ' + str(i) + '. iteraciji!' )
                self.Q_opt = copy.deepcopy(Q_new)
                self.V_opt = copy.deepcopy(self.V_from_Q(Q_new))
                self.opt_policy = copy.deepcopy(self.V_policy(self.V_from_Q(Q_new)))
                break
            
            Q_old = copy.deepcopy(Q_new)       
                    
           
    def compute_V(self, gamma, tol):
          
        self.env.frames=[]
        self.env.all_pi=[]
        
        
        V_new = copy.deepcopy(self.env.V)
        V_old = copy.deepcopy(self.env.V)
        R = self.env.rewards
        
        self.env.frames.append(copy.deepcopy(V_new))
        self.env.all_pi.append(copy.deepcopy(self.V_policy(V_new)))
         
        i = 0
        while (1):

            for s in self.env.states:
                
                V_new[s] = R[s]
                
                if s == (0,3) or s == (1,3):
                    continue
                
                temp_a = []
                for a in self.env.all_actions:
                    
                    temp1 = 0
                    for new_s in self.env.states:
                        
                        P_sa = self.probability(s, a, new_s)
                        temp1 += P_sa * V_old[new_s]
                    
                    temp_a.append(temp1)
                
                V_new[s] += gamma*max(temp_a)
                
            self.env.frames.append(copy.deepcopy(V_new))
            self.env.all_pi.append(copy.deepcopy(self.V_policy(V_new)))
            
            i+=1
                
            if np.max(abs(V_new - V_old)) < tol:
                
                print('V je konvergiralo u ' + str(i) + '. iteraciji!' )
                self.V_opt = copy.deepcopy(V_new)
                self.opt_policy = copy.deepcopy(self.V_policy(V_new))
                break
            
            V_old = copy.deepcopy(V_new)
            
        
    def V_policy(self, V):
        
        pi = np.zeros((self.env.nrows, self.env.ncolumns), object)
        pi[1,1] = ' '
        
        for s in self.env.states:
            
            if s == (0,3) or s == (1,3):
                pi[s] = 'exit'
                continue
            
            temp_a = []
            for a in self.env.all_actions:
                
                temp1 = 0
                for new_s in self.env.states:
                    
                    P_sa = self.probability(s, a, new_s)
                    temp1 += P_sa * V[new_s]
                
                temp_a.append(temp1)
                
            pi[s] = self.env.all_actions[np.argmax(temp_a)]
            
        return pi



    def Q_policy(self, Q):
        
        pi = np.zeros((self.env.nrows, self.env.ncolumns), object)
        pi[1,1] = ' '
        
        for s in self.env.states:
            
            if s == (0,3) or s == (1,3):
                pi[s] = 'exit'
                continue
                
            pi[s] = self.env.all_actions[np.argmax(Q[s[0], s[1]])]
            
        return pi
    


    def V_from_Q(self, Q):      
        
        V = np.zeros((self.env.nrows, self.env.ncolumns))
        V[1,1] = 2
        
        # Prvo nadji optimalno V za dobijeno optimalno Q
        for s in self.env.states:

            V[s] = np.max(Q[s[0], s[1], :])
            
        return V
    
    def eps_greedy_action(self, Q):
        
        prob = np.random.rand()
        if prob <= self.epsilon:
            action = np.random.choice(self.env.all_actions)
            
        else:
            s = self.current_state
            indx_a = np.argmax(Q[s[0], s[1], :])
            action = self.env.all_actions[indx_a]
            
        return action       
        
    
    def Q_learning(self, eps, alpha, gamma, tol, adapt):
        
        self.env.all_pi = []
        self.env.frames = []
        self.env.Q_frames = []        
        
        self.epsilon = eps
        self.alpha = alpha
        self.gamma = gamma
        self.tol = tol
        self.env.gamma = gamma
        self.env.epsilon = eps
        self.env.adapt = adapt
        self.env.alpha = []
        self.env.alpha.append(alpha)
        
        Q_old = copy.deepcopy(self.env.Q)
        Q = copy.deepcopy(self.env.Q)
        episode = 0
      
        while(1):
            
            # Trenutno stanje i nagrada 
            s = self.current_state
            R_s = self.current_reward
            q = R_s

            if not (s == (0,3) or s == (1,3)):

                # Agent generise zeljenu akciju 
                action = self.eps_greedy_action(Q)
                
                # Simulator vraca akciju koja se desava sa nekom verovatnocom 
                action_p, P = self.env.sample(action)
                indx_a = self.env.all_actions.index(action_p) 
                
                # Agent prelazi u novo stanje
                new_s = self.next_state(self.current_state, action_p)
                
            else:  
                
                Q[s[0], s[1], : ] = q

                # print("Epizoda " + str(episode))
                # print("\n V vrednosti : \n", self.V_from_Q(Q))
                # print("\n GORE : \n", Q[ : , : , 0])
                # print("\n DESNO : \n", Q[ : , : , 1])
                # print("\n DOLE : \n", Q[ : , : , 2])
                # print("\n LEVO : \n", Q[ : , : , 3])
                    
                # Da li je Q konvergiralo?
                if np.max(abs(Q-Q_old)) < self.tol:
                    
                    print('Q je konvergiralo u ' + str(episode) +'. epizodi!')
                    
                    self.Q_opt = copy.deepcopy(Q)
                    self.V_opt = copy.deepcopy(self.V_from_Q(Q))            
                    self.opt_policy = copy.deepcopy(self.Q_policy(Q))               
                    self.last_episode = episode
                    
                    break
                
                # Cuvanje svih Q i V vrednosti i svih politika
                self.env.Q_frames.append(copy.deepcopy(Q))
                self.env.frames.append(copy.deepcopy(self.V_from_Q(Q)))
                self.env.all_pi.append(copy.deepcopy(self.Q_policy(Q)))
                               
                # Q_old je sada Q, restartuju se pozicija i nagrada
                self.alpha = learning_rate(self.alpha, episode+1, adaptive = adapt)
                #print("\n Alpha :", self.alpha)
                self.env.alpha.append(self.alpha)
                self.current_state = (2, 0)
                self.current_reward = -0.04
                Q_old = copy.deepcopy(Q)                
                episode += 1
                
                continue
            
            # Racunaju se q(s, a) i Q(s, a)
            q += gamma * np.max(Q[new_s[0], new_s[1], :])
            Q[s[0], s[1], indx_a] += self.alpha * (q - Q[s[0], s[1], indx_a])
          
            # Azuriranje trenutnog stanja i nagrade
            self.current_state = new_s
            self.current_reward = self.env.rewards[new_s]
        
 
            
# %% Main
           
        
# env = Environment([0.8, 0.1, 0.1])    
# agent = Agent(env, (2,0))  

# agent.Q_learning(eps = 0.8, alpha = 0.5, gamma = 0.9, tol = 0.01, adapt = True)      


# agent.Q_iteration(gamma = 0.9, tol = 0.01)
# print()
# print(agent.opt_policy)
# print()

# agent.Q_learning(eps = 0.8, alpha = 0.5 , gamma = 1, tol = 0.01)
# print()
# print(agent.opt_policy)
# print()

