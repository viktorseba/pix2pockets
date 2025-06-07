# pix2pockets

This is the repo of the project ```pix2pockets``` developed as an extension of a bachelor thesis at DTU.

## Dataset
The dataset for the project is available on roboflow:
```shell
https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o
```

To download the dataset, a roboflow API key is required. To initialize this, create the file ```roboflow.json``` including the following line:
```shell
{
    "ROBOFLOW_API_KEY": "Your_api_here"
}
```

## Project Demo
The notebook ```pix2pockets_demo.ipynb``` contains a simple demo of the overall pipeline of the project and all implemented functionality.

## Paper Abstract 
Computer vision models have seen increased usage in sports, and reinforcement learning (RL) is famous for beating humans in strategic games such as Chess and Go. In this paper, we are interested in building upon these advances and examining the game of classic 8-ball pool. We introduce pix2pockets, a foundation for an RL-assisted pool coach. Given a single image of a pool table, we first aim to detect the table and the balls and then propose the optimal shot suggestion. For the first task, we build a dataset with 195 diverse images where we manually annotate all balls and table dots, leading to 5748 object segmentation masks. For the second task, we build a standardized RL environment that allows easy development and benchmarking of any RL algorithm. Our object detection model yields an AP50 of 91.2 while our ball location pipeline obtains an error of only 0.4 cm. Furthermore, we compare standard RL algorithms to set a baseline for the shot suggestion task and we show that all of them fail to pocket all balls without making a foul move. We also present a simple baseline that achieves a per-shot success rate of 94.7% and clears a full game in a single turn 30% of the time.

Paper available here:
```shell
https://pix2pockets.compute.dtu.dk
```