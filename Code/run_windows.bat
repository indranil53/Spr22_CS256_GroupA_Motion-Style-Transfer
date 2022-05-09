@echo off

set CONFIG=./ons/config/vox-256.yaml

set PYTHONPATH=%PYTHONPATH%;%CD%;%CD%;%CD%/ons

call python cam/cam_fomm.py --config %CONFIG% --relative --adapt_scale --no-pad --checkpoint ./ons/model/00000189-checkpoint.pth.tar %*
::call python cam/cam_fomm.py --config %CONFIG% --relative --adapt_scale --no-pad --checkpoint ./ons/model/quantized_model.pth.tar %*
::call python afy/cam_fomm.py --config %CONFIG% --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar %*