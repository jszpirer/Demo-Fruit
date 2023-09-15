import subprocess
from cmath import cos, exp, inf
import numpy as np
from numpy import linalg as LA
import os
import ast
import sys
import json
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

    def __init__(self, experience, iteration, nb_missions):
        self.nb_missions = int(nb_missions)
        self.iteration = iteration
        self.experience = experience
        self.mission = self.experience.split("_")[0]
        self.folder = f"/home/jszpirer/Mandarina/autodemo/irace/{experience}"
        self.demoFile = f"{self.folder}/mission-folder/{self.mission}.argos"
        self.arenaD = 1.73205
        self.patches, self.obstacles = self.retrievePatches(self.demoFile)

        # parse mu history for svm from muHistory.txt
        self.muHistory = []
        self.labelHistory = []
        for i in range(self.nb_missions):
            self.muHistory.append([])
            self.labelHistory.append([])
        with open(f"{self.folder}/muHistory", "r") as f:
            lines = f.readlines()
            for line in lines:
                mu = ast.literal_eval(line.split(";")[0])
                label = int(line.split(";")[1])
                for i in range(len(mu)):
                    self.muHistory[i].append(mu[i])
                    self.labelHistory[i].append(label)
        self.muE = [self.muHistory[0][0],self.muHistory[1][0]]

    def retrievePatches(self, argosFile):
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
        whiteRectangles = []
        blackRectangles = []
        for r in rectangles:
            if(r.getAttribute("color") == "white"):
                patches.append(ast.literal_eval("[" + r.getAttribute("center") + "," + r.getAttribute("width") + "," + r.getAttribute("height") + "]"))
            else:
                patches.append(ast.literal_eval("[" + r.getAttribute("center") + "," + r.getAttribute("width") + "," + r.getAttribute("height") + "]"))

        obstacles = []

        return patches, obstacles

    def distToCircle(self, circle, pos):
        c_x = circle[0]
        c_y = circle[1]
        r = circle[2]
        for obs in self.obstacles:
            if(self.intersect(pos,circle,obs[0], obs[1])):
                return self.arenaD
        return max(0, sqrt((pos[0]-c_x)**2 + (pos[1] - c_y)**2) - r)

    def distToRect(self, rect, pos):
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
        pfsm=" ".join(self.PFSM.convert_to_commandline_args())

        self.run_Argos(pfsm=pfsm, sleep=3)
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

        
        phi = []
        for i in range(len(swarm_pos)):
            neighbors = swarm_pos.copy()
            neighbors.pop(i)
            distance = min([LA.norm(np.array(swarm_pos[i]) - np.array(n), ord=2) for n in neighbors])
            phi.append(distance)

        h = (2*np.log(10))/(self.arenaD**2)
        phi = [exp(- h * self.arenaD * pos) for pos in phi]
        phi.sort(reverse=True) 

        for e in phi: phiTot.append(e)
        
        
        return phiTot

    def run_Argos(self, pfsm="--fsm-config --nstates 1 --s0 1 --cle0 4", sleep=1):
        # Run argos to get features
        subprocess.run(f"cd {self.folder}/mission-folder; /home/jszpirer/Mandarina/AutoMoDe-tuttifrutti/bin/automode_main -n -c {self.mission}.argos {pfsm}", shell=True)

    def computeMargin(self, init=False):
        """Implement 2) with SVM or projection algoritm
        """
        # svm algoritm
        list_w = []
        list_t = []
        list_distance = []
        list_coefs = []
        list_supportvectors = []
        new_t = 0
        new_distance = 0
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
            new_t += t
            distance = LA.norm(np.array(self.muE[j]) - np.array(self.mu[j]))
            list_distance.append(distance)
            new_distance += distance
        
        new_t = new_t/self.nb_missions
        new_distance = new_distance/self.nb_missions

        improved = False
        with open(f"{self.folder}/{self.experience}.json", "r") as file:
            info = json.load(file)
            previous_t = 0.5*info[-1]["t"][0]+0.5*info[-1]["t"][1] #TODO: Generaliser cette ligne a plus de deux missions
            previous_best = info[-1]["best fsm"]
            if(round(new_t,6) < previous_t):
                improved = True
            
        improved_distance = False
        with open(f"{self.folder}/{self.experience}.json", "r") as file:
            info = json.load(file)
            previous_distance = 0.5*info[-1]["distance"][0] + 0.5*info[-1]["distance"][1]
            previous_best_distance = info[-1]["best fsm distance"]
            if(round(new_distance,6) < previous_distance):
                improved_distance = True
        
        return list_w, list_t, list_distance, list_coefs, list_supportvectors, improved, improved_distance, previous_best, previous_best_distance 

    def compute_SVM(self, X, y):
        clf = svm.SVC(kernel="linear", class_weight='balanced')
        clf.fit(X, y)
        return clf

    def getFSM(self):
        with open(f"{self.folder}/mission-folder/fsm.txt",'r') as f:
            line = f.readline()
            PFSM = FSM.parse_from_commandline_args(line.split(" "))
        return PFSM

    def computePFSM(self):
        """Perform the algorithm to generate the desirable PFSM"""
        self.PFSM = self.getFSM()
        self.mu, phi_list = self.computeMu()
        for i in range(self.nb_missions):
            self.muHistory[i].append(self.mu[i])
            self.labelHistory[i].append(-1)     
        w, t, distance, svm, SVs, improved, improved_distance, previous_best, previous_best_distance = self.computeMargin() 
        pfsm = " ".join(self.PFSM.convert_to_commandline_args())
        if(improved): 
            best_pfsm = " ".join(self.PFSM.convert_to_commandline_args())
        else:
            best_pfsm = previous_best

        if(improved_distance):
            best_pfsm_distance = " ".join(self.PFSM.convert_to_commandline_args())
        else:
            best_pfsm_distance = previous_best_distance

        dico = {"iter" : self.iteration,
                "fsm" : pfsm,
                "best fsm" : best_pfsm,
                "best fsm distance": best_pfsm_distance,
                "mu" : [[round(e,6) for e in mu] for mu in self.mu],
                "w" : [[round(e,6) for e in weights] for weights in w],
                "svm coeff" : [[round(e,6) for e in coeffs] for coeffs in svm],
                "support vector(s)": [SVs[0].tolist(),SVs[1].tolist()], 
                "t" : [round(e,6) for e in t],
                "distance":[round(e,6) for e in distance]} 

        # Append JSON file history of produced PFSM
        with open(f"{self.folder}/{self.experience}.json", "r") as file:
            info = json.load(file)
        with open(f"{self.folder}/{self.experience}.json", "w") as file:
            info.append(dico)
            json.dump(info, file, indent=4)

        # Update mu history file
        with open(f"{self.folder}/muHistory",'a') as f:
            f.write(f"\n{self.mu};-1")

        # Update info for irace
        with open(f"{self.folder}/mission-folder/irl.txt",'w+') as f:
            f.write(f"{w}\n")
            f.write(f"{self.folder}/mission-folder/{self.mission}.argos")
    
if __name__ == '__main__':
    experience = sys.argv[1]
    iteration = sys.argv[2]
    nb_missions = sys.argv[3]
    autodemo = AUTODEMO(experience, iteration, nb_missions)
    PFSM = autodemo.computePFSM()
