CUDA_VISIBLE_DEVICES=0
mkdir -p outputs
python train.py --task_name STS-B --output_dir outputs/STS-B-model > outputs/STS-B.log
python train.py --task_name ATEC --output_dir outputs/ATEC-model > outputs/ATEC.log
python train.py --task_name BQ --output_dir outputs/BQ-model > outputs/BQ.log
python train.py --task_name LCQMC --output_dir outputs/LCQMC-model > outputs/LCQMC.log
python train.py --task_name PAWSX --output_dir outputs/PAWSX-model > outputs/PAWSX.log
