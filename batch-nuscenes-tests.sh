# TRAIN CAM_FRONT BOSTON TEST CAM_FRONT BOSTON
python3 test-nuscenes.py /media/scratch/camera-miscalib-data/nuScenes/index_grouped.csv car_b1+CAM_FRONT+test -batch_size 6 -buffer_size 12 -model_path /home/alpetrov/alpetrov/plr_miscalib_detection/models/nuscenes-b-front/ -model_name model-34-70455 -v 1 -njobs 20 -csvname "nuscenes-tr_b1_front-ts_b1_front-1pzero.csv"

# TRAIN CAM_FRONT BOSTON TEST CAM_FRONT SINGAPORE
python3 test-nuscenes.py /media/scratch/camera-miscalib-data/nuScenes/index_grouped.csv car_s1+CAM_FRONT+test -batch_size 6 -buffer_size 12 -model_path /home/alpetrov/alpetrov/plr_miscalib_detection/models/nuscenes-b-front/ -model_name model-34-70455 -v 1 -njobs 20 -csvname "nuscenes-tr_b1_front-ts_s1_front-1pzero.csv"

# TRAIN CAM_FRONT BOSTON TEST CAM_FRONT_LEFT BOSTON
python3 test-nuscenes.py /media/scratch/camera-miscalib-data/nuScenes/index_grouped.csv car_b1+CAM_FRONT_LEFT+test -batch_size 6 -buffer_size 12 -model_path /home/alpetrov/alpetrov/plr_miscalib_detection/models/nuscenes-b-front/ -model_name model-34-70455 -v 1 -njobs 20 -csvname "nuscenes-tr_b1_front-ts_b1_frontleft-1pzero.csv"

# TRAIN CAM_FRONT BOSTON TEST CAM_FRONT_RIGHT BOSTON
python3 test-nuscenes.py /media/scratch/camera-miscalib-data/nuScenes/index_grouped.csv car_b1+CAM_FRONT_RIGHT+test -batch_size 6 -buffer_size 12 -model_path /home/alpetrov/alpetrov/plr_miscalib_detection/models/nuscenes-b-front/ -model_name model-34-70455 -v 1 -njobs 20 -csvname "nuscenes-tr_b1_front-ts_b1_frontright-1pzero.csv"
