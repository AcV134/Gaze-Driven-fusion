#preparation
export PYTHONPATH=`pwd`:$PYTHONPATH
export LD_LIBRARY_PATH=/home/schischk/miniconda3/envs/disc/lib:$LD_LIBRARY_PATH

#For training, uncomment this command
# python tools/train.py trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false

#For validation, uncomment this command
# python tools/train.py resume.ckpt_path='/home/models/DISC/adl-project-gaze/weight_nicolas/DISC_KITTI360.ckpt' validate=true trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false

#for inference, uncomment this command
python tools/test.py ++ckpt_path='/home/models/DISC/adl-project-gaze/weight_nicolas/DISC_KITTI360.ckpt' trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false