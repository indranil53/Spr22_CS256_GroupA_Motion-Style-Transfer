# One-Shot Free-View Neural Talking Head Synthesis Implementation for Video Conferencing
### To run this locally you need a CUDA-enabled (NVIDIA) video card. Otherwise it will fallback to the central processor and run very slowly.
****
Live Feed Driving | Compressed Model Output :  


****
1. Run `run_windows.bat`. If execution was successful, two windows "cam" and "" will appear. Leave these windows open for the next steps. <!--If there are multiple cameras (including virtual ones) in the system, you may need to select the correct one. Open `scripts/settings_windows.bat` and edit `CAMID` variable. `CAMID` is an index number of camera like 0, 1, 2, ...-->
2. Install [OBS Studio](https://obsproject.com/) for capturing Avatarify output.
3. Install [VirtualCam plugin](https://obsproject.com/forum/resources/obs-virtualcam.539/). Choose `Install and register only 1 virtual camera`.
4. Run OBS Studio.
5.  In the Sources section, press on Add button ("+" sign), select Windows Capture and press OK. In the appeared window, choose "[python.exe]: avatarify" in Window drop-down menu and press OK. Then select Edit -> Transform -> Fit to screen.
6.  In OBS Studio, go to Tools -> VirtualCam. Check AutoStart, set Buffered Frames to 0 and press Start.
7.  Now `OBS-Camera` camera should be available in Zoom (or other videoconferencing software).

## Requirements
1. opencv-python
2. face-alignment
3. numpy
4. torch
5. scikit-image
6. pytorch
7. torchvision 
8. cudatoolkit


etc.