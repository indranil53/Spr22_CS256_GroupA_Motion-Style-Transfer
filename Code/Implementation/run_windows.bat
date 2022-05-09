@echo off

set CONFIG=./One-Shot/config/vox-256.yaml

set PYTHONPATH=%PYTHONPATH%;%CD%;%CD%;%CD%/One-Shot

call python cam/cam_fomm.py --config %CONFIG% --relative --adapt_scale --no-pad --checkpoint ./One-Shot/model/00000189-checkpoint.pth.tar %*
