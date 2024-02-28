import subprocess
from cmath import cos, exp, inf
import numpy as np
from numpy import linalg as LA
import random
import os
import ast
import sys
import json
from datetime import datetime
from xml.dom import minidom
from math import exp, sqrt, sin, cos, tan
from sklearn import svm

from automode.architecture.finite_state_machine import State, Transition, FSM
from automode.modules.tuttifrutti import Behavior, Condition

"""
Features vectors phi is dim=40 and each features is relative to the distance from the center of white/black floor patches

ALGO:

input: User demos, eps
1) Generate 1 random initial PFSM P_0 and compute mu_0, i=1
2) Compute t_i and w_i with SVM or projection algorithm
3) if t_i < eps, return PFSM
4) Run irace with R_i = w_i . phi_i to compute new PFSM P_i
5) Compute mu_i = mu(P_i)
6) i = i+1 and go back to 2)
"""

class AUTODEMO:

    def __init__(self, experience, extra, nb_missions):
        self.experience = experience
        self.extra = extra
        self.nb_missions = int(nb_missions)
        self.mission = self.experience.split("_")[0]
        self.folder = f"/home/jszpirer/Memoirepaper/codeDemo-Fruit/TuttiFrutti/local_code/ManualDesign/irace/{experience}/{experience}{extra}" #Change with your path
        self.demoFile = f"{self.folder}/mission-folder/{self.mission}.argos"
        self.arenaD = 1.73205
        self.demos = self.computeDemo(self.demoFile) # self.other_info when time and second mission
        self.patches, self.obstacles = self.retrievePatches(self.demoFile)

        self.muE = self.computeMuE()
        self.muHistory = []
        self.labelHistory = []
        for i in range(self.nb_missions):
            self.muHistory.append([self.muE[i]])
            self.labelHistory.append([])
            self.labelHistory[-1].append(1)

        self.PFSM = self.generate_rand_PFSM()
        self.BestPFSM = self.PFSM
        self.mu,_ = self.computeMu()
        for i in range(self.nb_missions):
            self.muHistory[i].append(self.mu[i])
            self.labelHistory[i].append(-1)

        # Initialize mu history file
        with open(f"{self.folder}/muHistory",'w+') as f:
            f.write(f"{self.muE};1\n")
            f.write(f"{self.mu};-1")

    def computeDemo(self, argosFile):
        """
        Function that extracts the positions of the robots in demo argos file. 
        
        When there are more than one mission, the length of the positions list determines the number of different missions.
        """
        # parse an xml file by name
        print(f'{argosFile}')
        file = minidom.parse(f'{argosFile}')

        #use getElementsByTagName() to get tag
        models = file.getElementsByTagName('demo')

        positions = []

        # Comstruction of the final positions list
        for i in range(self.nb_missions):
            positions.append([])

        for m in models:
            i = 0
            epucks = m.getElementsByTagName("epuck") # List of epucks in demo m
            pos = []
            for e in epucks:
                if len(pos)!= 0 and e.getAttribute("id") == "Epuck_0":
                    positions[i].append(pos)
                    i+=1
                    pos=[]
                pos.append(ast.literal_eval("[" + e.getAttribute("position") + "]"))
            positions[i].append(pos)
        print(f"Demo pos: {positions}")
        return positions

    def retrievePatches(self, argosFile):
        # Identifie quels sont les cercles et obstacles presents sur le plateau
        patches = []

        # parse an xml file by name
        file = minidom.parse(f'{argosFile}')

        #retriving circle patches
        circles = file.getElementsByTagName('circle')
        for c in circles:
            if(c.getAttribute("color") == "white"):
                patches.append(ast.literal_eval("[" + c.getAttribute("position") + "," + c.getAttribute("radius") + "]"))
            else:
                patches.append(ast.literal_eval("[" + c.getAttribute("position") + "," + c.getAttribute("radius") + "]"))

        #retriving rect patches
        rectangles = file.getElementsByTagName('rectangle')
        for r in rectangles:
            if(r.getAttribute("color") == "white"):
                patches.append(ast.literal_eval("[" + r.getAttribute("center") + "," + r.getAttribute("width") + "," + r.getAttribute("height") + "]"))
            else:
                patches.append(ast.literal_eval("[" + r.getAttribute("center") + "," + r.getAttribute("width") + "," + r.getAttribute("height") + "]"))

        obstacles = []
                
        return patches, obstacles

    def distToCircle(self, circle, pos):
        # Distance entre pos et le cercle sauf si obstacle
        c_x = circle[0]
        c_y = circle[1]
        r = circle[2]
        for obs in self.obstacles:
            if(self.intersect(pos,circle,obs[0], obs[1])):
                return self.arenaD
        return max(0, sqrt((pos[0]-c_x)**2 + (pos[1] - c_y)**2) - r)

    def distToRect(self, rect, pos):
        # Pareil que cercle mais avec un rectangle
        x_min = rect[0] - rect[2]/2
        x_max = rect[0] + rect[2]/2
        y_min = rect[1] - rect[3]/2
        y_max = rect[1] + rect[3]/2

        dx = max(x_min - pos[0], 0, pos[0] - x_max)
        dy = max(y_min - pos[1], 0, pos[1] - y_max)
        
        for obs in self.obstacles:
            if(self.intersect(pos,[x_min,pos[1]],obs[0], obs[1]) or
               self.intersect(pos,[x_max,pos[1]],obs[0], obs[1]) or
               self.intersect(pos,[pos[0],y_min],obs[0], obs[1]) or
               self.intersect(pos,[pos[0],y_max],obs[0], obs[1])):
               return self.arenaD
        return sqrt(dx**2 + dy**2)

    def ccw(self, a, b, c):
        return (c[0] - a[0])*(b[1] - a[1]) > (b[0] - a[0])*(c[1] - a[1])

    # Return true if segments AB and CD intersect
    def intersect(self, a, b, c, d):
        return (self.ccw(a,c,d) != self.ccw(b,c,d)) and (self.ccw(a,b,c) != self.ccw(a,b,d))

    def computeMuE(self):
        """
        Function that computes the average of all phi vectors at the beginning with the values of the demonstrations.
        """
        phi_list = []
        for sub_mission in self.demos:
            phi_list.append([])
            for demo in sub_mission:
                phi = self.computePhiE(demo)
                phi_list[-1].append(phi)

        mu = []
        for sub_mission in phi_list:
            mu.append([])
            for j in range(len(sub_mission[0])):
                avg = 0
                for i in range(len(sub_mission)):
                    avg += sub_mission[i][j]
                mu[-1].append(avg/len(sub_mission))
        
        print(f"muE = {mu}")

        return mu

    def computePhiE(self, demo):
        phiE = []
        # distance to pathces features
        for p in self.patches:
            phi = []
            patch = p.copy()

            for pos in demo:
                if(len(patch) == 3):
                    distance = self.distToCircle(patch, pos)
                else:
                    distance = self.distToRect(patch, pos)
                phi.append(distance)

            h = (2*np.log(10))/(self.arenaD**2)
            phi = [exp(- h * self.arenaD * pos) for pos in phi]
            phi.sort(reverse=True) 

            for e in phi: phiE.append(e)

        
        phi = []
        for i in range(len(demo)):
            neighbors = demo.copy()
            neighbors.pop(i)
            distance = min([LA.norm(np.array(demo[i]) - np.array(n), ord=2) for n in neighbors])
            phi.append(distance)

        h = (2*np.log(10))/(self.arenaD**2)
        phi = [exp(- h * self.arenaD * pos) for pos in phi]
        phi.sort(reverse=True) 

        for e in phi: phiE.append(e)

        return phiE
    
    def computeMu(self, exp=10):
        phi_list = []
        for i in range(self.nb_missions):
            phi_list.append([])
        for _ in range(exp):
            phi = self.computePhi()
            print(f"phi: {phi}")
            for i in range(self.nb_missions):
                phi_list[i].append(phi[i])

        mu = []
        for sub_mission in phi_list:
            mu.append([])
            for j in range(len(sub_mission[0])):
                avg = 0
                for i in range(len(sub_mission)):
                    avg += sub_mission[i][j]
                mu[-1].append(avg/len(sub_mission))

        return mu, phi_list

    def computePhi(self):
        """From the computed current PFSM,
        compute the features expectations mu
        in form of a list of position of all robots"""
        # Same as computePhiE but with the values after an iteration
        # pfsm=" ".join(self.PFSM.convert_to_commandline_args())

        self.run_Argos(sleep=3)
        # extract robots position  from argos output file
        # computes features expectation with demo positions

        swarm_pos = [[],[]]
        with open(f"{self.folder}/mission-folder/first_pos.mu", "r") as f:
            nb_pos = 0
            for pos in f.readlines():
                # if nb_pos%20 == 0: # Suppose that we always have 20 robots in a simulation
                    # swarm_pos.append([])
                swarm_pos[0].append(ast.literal_eval("[" + pos[:-2] + "]"))
                nb_pos += 1
        with open(f"{self.folder}/mission-folder/final_pos.mu", "r") as f:
            nb_pos = 0
            for pos in f.readlines():
                # if nb_pos%20 == 0: # Suppose that we always have 20 robots in a simulation
                    # swarm_pos.append([])
                swarm_pos[1].append(ast.literal_eval("[" + pos[:-2] + "]"))
                nb_pos += 1    
            
        print(f"swarm pos: {swarm_pos}")

        # Ligne commentee pour voir le resultat de l'output du file argos
        # os.remove(f"{self.folder}/mission-folder/pos.mu")

        phiTot = []
        for sub_mission in swarm_pos:
            phiTot.append([])
            for p in self.patches:
                phi = []
                patch = p.copy()

                for pos in sub_mission:
                    if(len(patch) == 3):
                        distance = self.distToCircle(patch, pos)
                    else:
                        distance = self.distToRect(patch, pos)
                    phi.append(distance)

                h = (2*np.log(10))/(self.arenaD**2)
                phi = [exp(- h * self.arenaD * pos) for pos in phi]
                phi.sort(reverse=True) 

                for e in phi: phiTot[-1].append(e)

        
        
        for j in range(len(swarm_pos)):
            phi = []
            for i in range(len(swarm_pos[j])):
                neighbors = swarm_pos[j].copy()
                neighbors.pop(i)
                distance = min([LA.norm(np.array(swarm_pos[j][i]) - np.array(n), ord=2) for n in neighbors])
                phi.append(distance)

            h = (2*np.log(10))/(self.arenaD**2)
            phi = [exp(- h * self.arenaD * pos) for pos in phi]
            phi.sort(reverse=True) 

            for e in phi: phiTot[j].append(e)

        return phiTot

    def run_Argos(self, pfsm="--fsm-config --nstates 1 --s0 1 --cle0 4", sleep=1):
        # Run argos to get features and create the fils with the positions after the iteration
        print("About to run the Argos")
        print(pfsm)
        subprocess.run(f"cd {self.folder}/mission-folder; /home/jszpirer/TuttiFrutti/AutoMoDe-tuttifrutti/bin/automode_main -n -c {self.mission}.argos {pfsm}", shell=True)       

    def generate_rand_PFSM(self):
        # Generates one first random PFSM
        PFSM = FSM()
        PFSM.states.clear()
        n = random.randint(1,4)	# Number of states

        for i in range(n):
            behavior = Behavior.get_by_name(random.choice([b for b in Behavior.behavior_list]))  # b is just the name
            # set the behavior
            s = State(behavior)
            s.ext_id = i
            PFSM.states.append(s)

        for s in PFSM.states:
            m = random.randint(1,4)	# Number of states
            from_state = s
            for j in range(m):
                transition_id = s.ext_id
                transition_ext_id = transition_id	
                possible_states = [s for s in PFSM.states if s != from_state]
                if(possible_states):
                    to_state = random.choice(possible_states)
                    transition_condition = Condition.get_by_name(random.choice([c for c in Condition.condition_list]))
                    t = Transition(from_state, to_state, transition_condition)
                    t.ext_id = transition_ext_id
                    PFSM.transitions.append(t)

        PFSM.initial_state = [s for s in PFSM.states if s.ext_id == 0][0]
            
        return PFSM

    def computeMargin(self, init=False):
        # Implement 2) with SVM or projection algoritm
        
        # svm algoritm
        list_w = []
        list_t = []
        list_distance = []
        list_coefs = []
        list_supportvectors = []
        for j in range(self.nb_missions):
            svm = self.compute_SVM(self.muHistory[j], self.labelHistory[j])
            svmCoef = svm.coef_[0]
            SVs = svm.support_vectors_
            list_coefs.append(svmCoef)
            list_supportvectors.append(SVs)

            w = (np.array(svm.coef_[0])/LA.norm(svm.coef_[0], ord=2)).tolist()
            list_w.append(w) 

            t = inf
            for i in range(1, len(self.muHistory[j])):
                mu = self.muHistory[j][i]
                tCand = np.dot(w, np.array(self.muE[j]) - np.array(mu))
                if(tCand < t):
                    t = tCand
            list_t.append(t)
            distance = LA.norm(np.array(self.muE[j]) - np.array(self.mu[j]))
            list_distance.append(distance)

        return list_w, list_t, list_distance, list_coefs, list_supportvectors

    def compute_SVM(self, X, y):
        print("Debut de svm")
        print(X)
        print(y)
        clf = svm.SVC(kernel="linear", class_weight='balanced')
        clf.fit(X, y)
        return clf

    def initiateJson(self):
        # Perform the algorithm to generate the desirable PFSM
        pfsm = " ".join(self.PFSM.convert_to_commandline_args())
        bpfsm = " ".join(self.BestPFSM.convert_to_commandline_args())
        w, t, distance, svm, SVs = self.computeMargin(init=True) 
        now = datetime.now()
        dt = now.strftime("%d_%m_%Y_%H_%M_%S")

        with open(f"{self.folder}/{self.experience}{self.extra}.json", "w") as file:
            json.dump([], file)

        dico = {"iter" : "0",
                "begin time" : dt,
                "muE" : self.muE,
                "fsm" : pfsm,
                "best fsm": bpfsm,
                "best fsm distance": bpfsm,
                "mu" : [[round(e,6) for e in self.mu[0]],[round(e,6) for e in self.mu[1]]],
                "w" : [[round(e,6) for e in w[0]], [round(e,6) for e in w[1]]],
                "svm coeff" : [[round(e,6) for e in svm[0]],[round(e,6) for e in svm[1]]],
                "support vectors": [SVs[0].tolist(),SVs[1].tolist()],
                "t" : [round(e,6) for e in t],
                "distance" : [round(e,6) for e in distance],
                "Norm" : "L2"
                }

        with open(f"{self.folder}/{self.experience}{self.extra}.json", "r") as file:
            info = json.load(file)
        with open(f"{self.folder}/{self.experience}{self.extra}.json", "w") as file:
            info.append(dico)
            json.dump(info, file, indent=4)
        
        # Initial upload of irl information
        with open(f"{self.folder}/mission-folder/irl.txt",'w+') as f:
            f.write(f"{w}\n")
            f.write(f"{self.folder}/mission-folder/{self.mission}.argos")
        
if __name__ == '__main__':
    experience = sys.argv[1]
    extra = sys.argv[2]
    nb_missions = sys.argv[3]
    autodemo = AUTODEMO(experience, extra, nb_missions)
    autodemo.initiateJson()
