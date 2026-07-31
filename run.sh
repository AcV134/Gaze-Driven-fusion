#preparation
export PYTHONPATH=`pwd`:$PYTHONPATH
export LD_LIBRARY_PATH=/home/anirudh/miniconda3/envs/disc/lib:$LD_LIBRARY_PATH

#For training, uncomment this command
# python tools/train.py trainer.devices=1 use_gaze=True

#For validation, uncomment this command
# python tools/train.py resume.ckpt_path='/home/models/DISC/adl-project-gaze/pretrain/DISC_KITTI360.ckpt' validate=true trainer.devices=1 

#for inference, uncomment this command
# python tools/test.py ++ckpt_path='/home/models/DISC/adl-project-gaze/pretrain/DISC_KITTI360.ckpt' trainer.devices=1 

python tools/generate_outputs.py

#for visualization, uncomment this command
# QT_QPA_PLATFORM=offscreen python tools/visualize.py