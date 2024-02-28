# Demo-Fruit

## Files locations
Keep the content of "/local code" on your local compute.
Load the content of "cluster code" on your cluster.

## Running an experiment for Demo-Cho with the scalarization
1. Run "python3 Weighted_sum/runAutodemo.py [missionName_date] [number of missions in the sequence] [number of independent designs] [number of iterations]" to launch the design process on the cluster.
2. Whait for the cluster to finished to run the design process.
3. Run "python3 scalar/JsonCollector.py [mission name + date] [number of independent designs]" to collect all data of the design process in for of Json files.

## Running an experiment for Demo-Cho without the scalarization
1. Run "python3 Rank_based/runAutoMoDe.py [missionName_date] [number of missions in the sequence] [number of independent designs] [number of iterations]" to launch the design process on the cluster.
2. Whait for the cluster to finished to run the design process.
3. Run "python3 noscalar/JsonCollector.py [mission name + date] [number of independent designs]" to collect all data of the design process in for of Json files.
