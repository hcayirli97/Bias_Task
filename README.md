# Bias Job Interview Task
 
The aim of the project is to determine the road lanes in the images taken from the camera located in front of the car and to determine the lane parameters. The HybridNets model is determined as the model that determines the location of the strips. After obtaining the masks of the lanes by semantic segmentation, the resulting road lane blobs were obtained and the polynomial line fitting to these blobs was determined.

## Getting Started
After the raw frame is obtained from the videos, it is provided as input to the model.

![frame](https://github.com/hcayirli97/Bias_Task/blob/main/imgs/frame.png)

By using the pretrained model trained with the bdd100k dataset of the HybritNets model, only road lanes are determined. The segmentation outputs obtained are shown below.

![seg](https://github.com/hcayirli97/Bias_Task/blob/main/imgs/segmentation.png)

After obtaining the segmentation mask, morphology operations were applied. The aim is to get rid of the meaningless blobs in the resulting mask image and obtain the desired road lanes. A line fit was then applied to the blobs. The resulting lines are shown below.

![line](https://github.com/hcayirli97/Bias_Task/blob/main/imgs/lines.png)

## How to Run

Outputs can be obtained by running the [Bias_Task.ipynb](https://drive.google.com/file/d/1RVYv92drRRV6-6yZdNhfnbadeJO8e63s/view?usp=sharing) code in the repo via google colab. After providing the necessary input arguments to the main.py python file, the parameters of the road lanes are printed. Example command to run is given below.

```sh  
python main.py --source "Videos/NO20221023-112720-000012F.MP4"     
```

After processing a frame, the output of the road strips looks as follows.

```sh  
Frame 1010 
--------------------
line 0 - Coefficients a:0.0001 b:0.3319 c:335.4891
line 1 - Coefficients a:0.0000 b:-0.7696 c:1982.0282
line 2 - Coefficients a:0.0001 b:-0.2156 c:825.0295
line 3 - Coefficients a:0.0000 b:0.0678 c:937.2379
line 4 - Coefficients a:0.0001 b:0.3771 c:191.1986
line 5 - Coefficients a:0.0000 b:-0.1423 c:1018.3421    
```
