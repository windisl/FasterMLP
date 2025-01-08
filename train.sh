
python train_test.py -g 1 --num_nodes 1 -n 8 -b 2048 -e 300 \
--data_dir ./data/imagenet1k --pin_memory --wandb_project_name fasternet \
--model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') --cfg cfg/fasternet_t0.yaml --wandb_offline


