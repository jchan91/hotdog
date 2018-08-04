
@ECHO OFF

activate keras

set scriptDir=%~dp0
pushd %scriptDir%
python train.py
