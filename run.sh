#preparation
export PYTHONPATH=`pwd`:$PYTHONPATH
export LD_LIBRARY_PATH=/home/anirudh/miniconda3/envs/disc/lib:$LD_LIBRARY_PATH

# use_gaze=True if you want to use gaze data for training, validation and inference. 
# gaze_input=True if you want to use gaze data as input to the model.
# trainer.limit_val_batches=5 is used to limit the number of validation batches to 5 for faster valid#ation. You can change this value as per your requirement.
# trainer.limit_train_batches=5 is used to limit the number of training batches to 5 for faster training. You can change this value as per your requirement.

#  use '/home/models/DISC/adl-project-gaze/outputs/DISC/DISC-v1-360/Gaze_ckpt/gaze-04-ce0.1868.ckpt' where we use gaze data as input to the model and '/home/models/DISC/adl-project-gaze/outputs/DISC/DISC-v1-360/Gaze_ckpt/e4_miou0.1791.ckpt' where we don't use gaze data as input to the model.


#For training, uncomment this command
python tools/train.py trainer.devices=1 use_gaze=True gaze_input=True

#For validation, uncomment this command
# python tools/train.py resume.ckpt_path='adl-project-gaze/pretrain/DISC_KITTI360.ckpt' validate=true trainer.devices=1 

#for inference, uncomment this command
# python tools/test.py ++ckpt_path='adl-project-gaze/pretrain/DISC_KITTI360.ckpt' trainer.devices=1 

# python tools/generate_outputs.py use_gaze=True ckpt_path='outputs/DISC/DISC-v1-360/best_overfit-39-ce2.0240.ckpt'

#for visualization, uncomment this command
# QT_QPA_PLATFORM=offscreen python tools/visualize.py path='outputs/KITTI360/overfit/predictions'