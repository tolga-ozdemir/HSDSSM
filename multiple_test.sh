python hsi_test.py -a lamamba -p gauss -r -rp ./checkpoints/lamamba/complex_v1/model_epoch_100_525083.pth --gpu-ids 0 --loss char > ./checkpoints/lamamba/complex_v1/results_lamamba_100_fixed.txt

python hsi_test.py -a lamamba -p gauss -r -rp ./checkpoints/lamamba/complex/model_epoch_81_493600.pth --gpu-ids 0 --loss char > ./checkpoints/lamamba/complex/results_lamamba_81_fixed.txt
python hsi_test.py -a lamamba -p gauss -r -rp ./checkpoints/lamamba/complex/model_epoch_82_495257.pth --gpu-ids 0 --loss char > ./checkpoints/lamamba/complex/results_lamamba_82_fixed.txt
python hsi_test.py -a lamamba -p gauss -r -rp ./checkpoints/lamamba/complex/model_epoch_83_496914.pth --gpu-ids 0 --loss char > ./checkpoints/lamamba/complex/results_lamamba_83_fixed.txt
python hsi_test.py -a lamamba -p gauss -r -rp ./checkpoints/lamamba/complex/model_epoch_84_498571.pth --gpu-ids 0 --loss char > ./checkpoints/lamamba/complex/results_lamamba_84_fixed.txt
python hsi_test.py -a lamamba -p gauss -r -rp ./checkpoints/lamamba/complex/model_epoch_85_500228.pth --gpu-ids 0 --loss char > ./checkpoints/lamamba/complex/results_lamamba_85_fixed.txt
python hsi_test.py -a lamamba -p gauss -r -rp ./checkpoints/lamamba/complex/model_epoch_100_525083.pth --gpu-ids 0 --loss char > ./checkpoints/lamamba/complex/results_lamamba_100_fixed.txt

python hsi_test.py -a ssumamba -p gauss -r -rp ./checkpoints_old/ssumamba_correct_as_paper/complex_v1/model_epoch_100_525083.pth --gpu-ids 0 --loss char > ./checkpoints_old/ssumamba_correct_as_paper/complex_v1/results_ssumamba_100_fixed.txt

