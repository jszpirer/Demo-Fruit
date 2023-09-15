from cmath import cos, exp, inf
import numpy as np
from numpy import linalg as LA
import random
import time
import os
import ast
import paramiko
import sys
import json
from datetime import datetime
from xml.dom import minidom
from math import exp, sqrt, sin, cos
# from sklearn import svm
import getpass

# from automode.architecture.finite_state_machine import State, Transition, FSM
# from automode.modules.chocolate import Behavior, Condition

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

    def __init__(self, experience):
        self.password = getpass.getpass(prompt='Enter password: ')
        self.mission = experience.split("_")[0]
        self.remoteFolder = f"/home/jszpirer/Mandarina/autodemo/irace/{experience}" # Path in cluster
        self.sshCreateFolder()
        print("sshCreateFolder done")
        self.sshCopyFolder()
        print("sshCopyFolder done")

        self.demoFile = f"/home/students/jszpirer/Mandarina/local_code/ArgosFiles/{self.mission}.argos" # Path in local code
        remotepath =  f"{self.remoteFolder}/mission-folder/{self.mission}.argos"
        remotepath_sequence1 = f"{self.remoteFolder}/mission-folder/sequence-1.argos"
        remotepath_sequence2 = f"{self.remoteFolder}/mission-folder/sequence-2.argos"
        localpath = self.demoFile
        self.sshPut(localpath, remotepath, remotepath_sequence1, remotepath_sequence2)

    def sshPut(self, localpath, remotepath, remotepath_sequence1, remotepath_sequence2):
        failed = True
        while(failed):
            try:
                host = "majorana.ulb.be"
                password = self.password  
                username = "jszpirer"
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname = host,username = username, password = password)
                sftp = ssh.open_sftp()
                sftp.put(localpath, remotepath)
                sftp.put(localpath, remotepath_sequence1)
                sftp.put(localpath, remotepath_sequence2)
                sftp.put(localpath, remotepath)
                sftp.close()
                ssh.close()
                failed = False
            except:
                print(f"Failed to ssh put: \"{localpath}\" in {remotepath}")
                time.sleep(1)

    def sshCreateFolder(self):
        failed = True
        while(failed):
            try:
                host = "majorana.ulb.be"
                password = self.password  
                username = "jszpirer"
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname = host,username = username, password = password)
                sftp = ssh.open_sftp()
                stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {self.remoteFolder}")
                sftp.close()
                ssh.close()
                failed = False
            except:
                print(f"SSH create folder failed")
                time.sleep(1)

    def sshCopyFolder(self):
        failed = True
        while(failed):
            try:
                host = "majorana.ulb.be"
                password = self.password  
                username = "jszpirer"
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname = host,username = username, password = password)
                sftp = ssh.open_sftp()
                print(self.remoteFolder)
                stdin, stdout, stderr = ssh.exec_command(f"cp -r /home/jszpirer/Mandarina/autodemo/irace/example/* {self.remoteFolder}")
                sftp.close()
                ssh.close()
                if(len(stderr.readlines()) > 0):
                    failed = True
                else:
                    failed = False
            except:
                print(f"SSH copy folder failed")
                time.sleep(1)

        return stdout

    def sshCmd(self, cmd):
        print("hey")
        failed = True
        while(failed):
            try:
                host = "majorana.ulb.be"
                password = self.password  
                username = "jszpirer"
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname = host,username = username, password = password)
                sftp = ssh.open_sftp()
                print(cmd)
                stdin, stdout, stderr = ssh.exec_command(cmd)
                print(stderr.readlines())
                sftp.close()
                ssh.close()
                if(len(stderr.readlines()) > 0):
                    failed = True
                else:
                    failed = False
            except:
                print(f"SSH cmd: \"{cmd}\" failed")
                time.sleep(1)
        return stdout
        
if __name__ == '__main__':
    experience = sys.argv[1]
    steps = sys.argv[2]
    number = sys.argv[3]
    iterations = sys.argv[4]
    for i in range(int(number)):
        autodemo = AUTODEMO(f"{experience}_{i+1}")
