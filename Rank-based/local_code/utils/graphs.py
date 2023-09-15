import json
import sys
import math
import os
from numpy import linalg as LA
from numpy import array as arr
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns; sns.set_theme()

if __name__ == "__main__":
    mission = sys.argv[1]
    date = sys.argv[2]
    nb_designs = int(sys.argv[3])
    
    # Every json is processed and put in "distance" folder to only keep some information
    info_final_plot = []
    exp = 1
    os.mkdir(f"../JSON/{mission}_{date}/distance")
    for file in os.listdir(f"../JSON/{mission}_{date}"):
        if file.split(".")[-1] == "json":
            info_to_write = []
            f = os.path.join(f"../JSON/{mission}_{date}", file)
            print(f"opening {f}")
            j = file.split(".")[0].split("_")[-1]
            with open(f, "r") as file:
                info = json.load(file)
            muE = arr(info[0]["muE"])
            for i in range(0, len(info)):
                t = arr(info[i]["t"])
                best_fsm = info[i]["best fsm"]
                fsm = info[i]["fsm"]
                distance = arr(info[i]["distance"])
                info_to_write.append({"iter":i, "t part1":t.tolist()[0], "t part2":t.tolist()[1],"distance part1":distance.tolist()[0], "distance part2":distance.tolist()[1], "fsm":fsm, "best fsm":best_fsm})
                if i == len(info)-1:
                    info_final_plot.append({"exp":exp,"distance part1":distance.tolist()[0], "distance part2":distance.tolist()[1], "fsm":fsm})
                    exp+=1
            with open(f"../JSON/{mission}_{date}/distance/{mission}_{date}_{j}.json", "w+") as jsonFile:
                json.dump(info_to_write, jsonFile)
    with open(f"../JSON/{mission}_{date}/last_produced.json", "w+") as jsonFile:
                json.dump(info_final_plot, jsonFile)
    
    # Every json is processed and put in "best_distance" folder to only keep the best distance so far in the file
    info_final_plot = []
    exp = 1
    os.mkdir(f"../JSON/{mission}_{date}/best_distance")
    for file in os.listdir(f"../JSON/{mission}_{date}/distance"):
        if file.split(".")[-1] == "json":
            info_to_write = []
            f = os.path.join(f"../JSON/{mission}_{date}/distance", file)
            print(f"opening {f}")
            j = file.split(".")[0].split("_")[-1]
            with open(f, "r") as file:
                info = json.load(file)
            # best_distance = arr(info[0]["distance"])
            best_distance_part1 = round(info[0]["distance part1"],6)
            best_distance_part2 = round(info[0]["distance part2"],6)
            distance_part1_mean = round(info[0]["distance part1"],6)
            distance_part2_mean = round(info[0]["distance part2"],6)
            best_mean_distance = 0.5*best_distance_part1+0.5*best_distance_part2
            best_fsm_part1 = info[0]["fsm"]
            best_fsm_part2 = info[0]["fsm"]
            best_fsm_mean = info[0]["fsm"]
            info_to_write.append({"iter":0, "distance part1":best_distance_part1, "distance part2":best_distance_part2, "distance mean":best_mean_distance, "fsm part1":best_fsm_part1, "fsm part2":best_fsm_part2, "fsm mean":best_fsm_mean})
            print("la longueur de info est ")
            print(len(info))
            for i in range(1, len(info)):
                # distance = arr(info[i]["distance"])
                distance = [info[i]["distance part1"],info[i]["distance part2"]]
                # See if distance is better for part 1
                if(round(distance[0],6) < best_distance_part1):
                    best_distance_part1 = round(distance[0],6)
                    best_fsm_part1 = info[i]["fsm"]
                # See if distance is better for part 2
                if(round(distance[1],6) < best_distance_part2):
                    best_distance_part2 = round(distance[1],6)
                    best_fsm_part2 = info[i]["fsm"]
                # See if distance is better for the comination of the two parts
                mean_distance = 0.5*distance[0]+0.5*distance[1]
                if(round(mean_distance,6) < best_mean_distance):
                    best_mean_distance = round(mean_distance,6)
                    best_fsm_mean = info[i]["fsm"]
                    distance_part1_mean = distance[0]
                    distance_part2_mean = distance[1]
                info_to_write.append({"iter":i, "distance part1":best_distance_part1, "distance part2":best_distance_part2, "distance mean":best_mean_distance, "fsm part1":best_fsm_part1, "fsm part2":best_fsm_part2, "fsm mean":best_fsm_mean})
                if i == len(info)-1:
                    info_final_plot.append({"exp":exp,"distance part1":distance_part1_mean, "distance part2":distance_part2_mean,"mean distance":best_mean_distance, "fsm":best_fsm_mean})
                    exp+=1
            with open(f"../JSON/{mission}_{date}/best_distance/{mission}_{date}_{j}.json", "w+") as jsonFile:
                json.dump(info_to_write, jsonFile)
    with open(f"../JSON/{mission}_{date}/mean_produced.json", "w+") as jsonFile:
                json.dump(info_final_plot, jsonFile)

    # Every json is processed and put in "best_distance" folder to only keep the best distance so far in the file
    info_final_plot = []
    exp = 1
    for file in os.listdir(f"../JSON/{mission}_{date}/distance"):
        if file.split(".")[-1] == "json":
            info_to_write = []
            f = os.path.join(f"../JSON/{mission}_{date}/distance", file)
            print(f"opening {f}")
            j = file.split(".")[0].split("_")[-1]
            with open(f, "r") as file:
                info = json.load(file)
            best_distance_part1 = round(info[0]["distance part1"],6)
            best_distance_part2 = round(info[0]["distance part2"],6)
            best_norm = math.sqrt(best_distance_part1**2+best_distance_part2**2)
            best_fsm_norm = info[0]["fsm"]
            for i in range(1, len(info)):
                # distance = arr(info[i]["distance"])
                distance = [info[i]["distance part1"],info[i]["distance part2"]]
                # See if norm distance is better for the comination of the two parts
                norm_distance = math.sqrt(distance[0]**2+distance[1]**2)
                if(round(norm_distance,6) < best_norm):
                    best_norm = round(norm_distance,6)
                    best_fsm_norm = info[i]["fsm"]
                    best_distance_part1 = distance[0]
                    best_distance_part2 = distance[1]
                if i == len(info)-1:
                    info_final_plot.append({"exp":exp,"distance part1":best_distance_part1, "distance part2":best_distance_part2,"norm distance":best_norm, "best norm fsm":best_fsm_norm})
                    exp+=1
    with open(f"../JSON/{mission}_{date}/l2_norm.json", "w+") as jsonFile:
                json.dump(info_final_plot, jsonFile)
    
    
    # Draw graphs

    ## Graph with the evolution of the best mean distance so far for every experiment
    dfs = []

    for i in range(nb_designs):
        path = f"../JSON/{mission}_{date}/best_distance/{mission}_{date}_{i+1}.json"
        dictionary = json.load(open(path, 'r'))
        dfs.append(pd.DataFrame(dictionary))

    plt.figure(figsize=(15,6))

    i=1
    ax = dfs[0]['distance mean'].plot(label=f"experience {i}", legend=True)
    for df in dfs[1:]:
        i += 1
        df['distance mean'].plot(ax=ax, label=f"experience {i}", legend=True)
        print(f"t: {df['distance mean'].tolist()}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean distance to expert feature") 
    plt.title("Evolution of mean distance between mu and muE over "+str(nb_designs)+" independent designs")
    plt.savefig(f"../JSON/{mission}_{date}/evolution_best_mean_distance.png")


    ## Graph with the evolution of the part 1 distance for every experiment
    dfs = []

    for i in range(nb_designs):
        path = f"../JSON/{mission}_{date}/distance/{mission}_{date}_{i+1}.json"
        dictionary = json.load(open(path, 'r'))
        dfs.append(pd.DataFrame(dictionary))

    plt.figure(figsize=(15,6))

    i=1
    ax = dfs[0]['distance part1'].plot(label=f"experience {i}", legend=True)
    for df in dfs[1:]:
        i += 1
        df['distance part1'].plot(ax=ax, label=f"experience {i}", legend=True)
        print(f"distance: {df['distance part1'].tolist()}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance to expert feature for part 1") 
    plt.title("Evolution of distance between mu and muE over "+str(nb_designs)+" independent designs for the first part of the sequence")
    plt.savefig(f"../JSON/{mission}_{date}/evolution_distance_part1.png")

    ## Graph with the evolution of the part 1 distance for every experiment
    dfs = []

    for i in range(nb_designs):
        path = f"../JSON/{mission}_{date}/distance/{mission}_{date}_{i+1}.json"
        dictionary = json.load(open(path, 'r'))
        dfs.append(pd.DataFrame(dictionary))

    plt.figure(figsize=(15,6))

    i=1
    ax = dfs[0]['distance part2'].plot(label=f"experience {i}", legend=True)
    for df in dfs[1:]:
        i += 1
        df['distance part2'].plot(ax=ax, label=f"experience {i}", legend=True)
        print(f"distance: {df['distance part2'].tolist()}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance to expert feature for part 2") 
    plt.title("Evolution of distance between mu and muE over "+str(nb_designs)+" independent designs for the second part of the sequence")
    plt.savefig(f"../JSON/{mission}_{date}/evolution_distance_part2.png")


    ## Graph with the evolution of t part1 for every experiment
    dfs = []

    for i in range(nb_designs):
        path = f"../JSON/{mission}_{date}/distance/{mission}_{date}_{i+1}.json"
        dictionary = json.load(open(path, 'r'))
        dfs.append(pd.DataFrame(dictionary))

    plt.figure(figsize=(15,6))

    i=1
    ax = dfs[0]['t part1'].plot(label=f"experience {i}", legend=True)
    for df in dfs[1:]:
        i += 1
        df['t part1'].plot(ax=ax, label=f"experience {i}", legend=True)
        print(f"distance: {df['t part1'].tolist()}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Marginal distance for part 1") 
    plt.title("Evolution of marginal distance over "+str(nb_designs)+" independent designs for the first part of the sequence")
    plt.savefig(f"../JSON/{mission}_{date}/evolution_t_part1.png")


    ## Graph with the evolution of t part1 for every experiment
    dfs = []

    for i in range(nb_designs):
        path = f"../JSON/{mission}_{date}/distance/{mission}_{date}_{i+1}.json"
        dictionary = json.load(open(path, 'r'))
        dfs.append(pd.DataFrame(dictionary))

    plt.figure(figsize=(15,6))

    i=1
    ax = dfs[0]['t part2'].plot(label=f"experience {i}", legend=True)
    for df in dfs[1:]:
        i += 1
        df['t part2'].plot(ax=ax, label=f"experience {i}", legend=True)
        print(f"distance: {df['t part2'].tolist()}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Marginal distance for part 2") 
    plt.title("Evolution of marginal distance over "+str(nb_designs)+" independent designs for the second part of the sequence")
    plt.savefig(f"../JSON/{mission}_{date}/evolution_t_part2.png")


    ## Graph with the results in function of the two distances
    mission_1 = mission.split("-")[1]
    mission_2 = mission.split("-")[2]

    dictionary1 = json.load(open(f"../JSON/{mission}_{date}/last_produced.json", 'r'))
    df1 = pd.DataFrame(dictionary1)

    dictionary2 = json.load(open(f"../JSON/{mission}_{date}/mean_produced.json", 'r'))
    df2 = pd.DataFrame(dictionary2)

    dictionary3 = json.load(open(f"../JSON/{mission}_{date}/l2_norm.json", 'r'))
    df3 = pd.DataFrame(dictionary3)

    plt.figure(figsize=(15,6))

    ax = df1.plot(kind='scatter', x='distance part1',y='distance part2', marker='o', label="Last PFSM", c="c")
    df2.plot(ax=ax, kind='scatter', x='distance part1', y='distance part2', marker='^', label="Mean distance", c="lime")
    df3.plot(ax=ax, kind='scatter', x='distance part1', y='distance part2', marker='*', label="L2 norm", c="m")
    ax.set_xlabel("Distance sub-mission "+mission_1)
    ax.set_ylabel("Distance sub-mission "+mission_2) 
    plt.title("Aggregated results for the "+mission+" mission")
    plt.savefig(f"../JSON/{mission}_{date}/results.png")

    
    # fig, ax2 = plt.subplots(ncols=2)
    # AAC
    colormap=matplotlib.cm.get_cmap('RdYlBu_r')
    w_part1 = []
    w_part2 = []

    for j in range(nb_designs):
        path = f"../JSON/{mission}_{date}/{mission}_{date}_{j+1}.json"
        w = json.load(open(path, 'r'))[-1]["w"]
        print("Les sommes des w")
        print(sum(w[0]))
        print(sum(w[1]))
        w_part1.append(w[0])
        w_part2.append(w[1])
    w = [w_part1,w_part2]

    plt.figure(figsize=(8, 5))

    ax1 = plt.subplot(1, 2, 1)
    a = np.array(w[0])
    res = np.average(a, 0)
    h2 = sns.heatmap(res.reshape(len(a[0]), 1), center=0, ax=ax1, cmap=colormap, yticklabels=5, xticklabels=False)
    cbar2 = h2.collections[0].colorbar
    cbar2.ax.set_yticklabels(cbar2.ax.get_yticklabels(), rotation=90, va='center')
    ax1.invert_yaxis()
    ax1.set_ylabel("Circle (0-19)       Black right (20-39)       Black left (40-59)     Neighbor(60-79)",  fontsize=8) #aac
    ax1.set_xlabel(mission_1)
    ax1.tick_params(axis='y', pad=-1)


    ax2 = plt.subplot(1, 2, 2)
    a = np.array(w[1])
    res = np.average(a, 0)
    h2 = sns.heatmap(res.reshape(len(a[0]), 1), center=0, ax=ax2, cmap=colormap, yticklabels=5, xticklabels=False)
    cbar2 = h2.collections[0].colorbar
    cbar2.ax.set_yticklabels(cbar2.ax.get_yticklabels(), rotation=90, va='center')
    ax2.invert_yaxis()
    ax2.set_ylabel("Circle (0-19)       Black right (20-39)       Black left (40-59)     Neighbor(60-79)",  fontsize=8) #aac
    ax2.set_xlabel(mission_2)
    ax2.tick_params(axis='y', pad=-1)

    plt.tight_layout(pad=2)

    plt.savefig(f"../JSON/{mission}_{date}/heatmap_"+mission+".png")


    # fig, ax2 = plt.subplots(ncols=2)
    # AAC
    colormap=matplotlib.cm.get_cmap('RdYlBu_r')
    w_part1 = []
    w_part2 = []

    for j in range(nb_designs):
        path = f"../JSON/{mission}_{date}/{mission}_{date}_{j+1}.json"
        w = json.load(open(path, 'r'))[-1]["w"]
        print("Les sommes des w")
        print(sum(w[0]))
        print(sum(w[1]))
        w_part1.append(w[0])
        w_part2.append(w[1])
    w = [w_part1,w_part2]

    plt.figure(figsize=(4, 5))

    ax1 = plt.subplot(1, 2, 1)
    a = np.array(w[0])
    res = np.average(a, 0)
    h2 = sns.heatmap(res.reshape(len(a[0]), 1), center=0, ax=ax1, cmap=colormap, yticklabels=5, xticklabels=False)
    cbar2 = h2.collections[0].colorbar
    cbar2.ax.set_yticklabels(cbar2.ax.get_yticklabels(), rotation=90, va='center')
    ax1.invert_yaxis()
    ax1.set_ylabel(" Circle (0-19)          Black right (20-39)    Black left (40-59)     Neighbor(60-79)",  fontsize=8) #aac
    # ax1.set_xlabel(mission_1)
    ax1.tick_params(axis='y', pad=-1)


    ax2 = plt.subplot(1, 2, 2)
    a = np.array(w[1])
    res = np.average(a, 0)
    h2 = sns.heatmap(res.reshape(len(a[0]), 1), center=0, ax=ax2, cmap=colormap, yticklabels=5, xticklabels=False)
    cbar2 = h2.collections[0].colorbar
    cbar2.ax.set_yticklabels(cbar2.ax.get_yticklabels(), rotation=90, va='center')
    ax2.invert_yaxis()
    # ax2.set_ylabel("Circle (0-19)       Black right (20-39)       Black left (40-59)     Neighbor(60-79)",  fontsize=8) #aac
    # ax2.set_xlabel(mission_2)
    ax2.tick_params(axis='y', pad=-1)

    plt.tight_layout(pad=2)

    plt.savefig(f"../JSON/{mission}_{date}/heatmap_"+mission+"final.png")
    
    exit(0)
