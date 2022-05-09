# One-Shot Free-View Neural Talking Head Synthesis
Unofficial pytorch implementation of paper "One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing" from [zhanglonghao](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis)
****
Driving | Result:  
![show](https://github.com/indranil53/Spr22_CS256_GroupA_Motion-Style-Transfer/blob/main/Results/Driving_Source_Pair.gif)

****
Train:  
--------
```
python run.py --config config/vox-256.yaml --device_ids 0,1,2,3,4,5,6,7
```

Demo:  
--------
```
python demo.py --config config/vox-256.yaml --checkpoint path/to/checkpoint --source_image path/to/source --driving_video path/to/driving --relative --adapt_scale --find_best_frame
```
free-view (e.g. yaw=20, pitch=roll=0):
```
python demo.py --config config/vox-256.yaml --checkpoint path/to/checkpoint --source_image path/to/source --driving_video path/to/driving --relative --adapt_scale --find_best_frame --free_view --yaw 20 --pitch 0 --roll 0
```
Note: run ```crop-video.py --inp driving_video.mp4``` first to get the cropping suggestion and crop the raw video.  
