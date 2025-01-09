
## Image Classification
### 1. Dependency Setup
Create an new conda virtual environment
```
conda create -n fasternet python=3.9.12 -y
conda activate fasternet
```
Clone this repo and install required packages:
```
git clone https://github.com/windisl/FasterMLP
cd FasterMLP/
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

### 3. Pre-trained Models
[FasterMLP-S](https://github.com/windisl/FasterMLP/releases/download/untagged-36c521c81aa994aae9fb/fastermlp-epoch.283-val_acc1.72.9740.pth) 

### 4. Evaluation

We give an example evaluation command for a ImageNet-1K pre-trained FasterMLP-T0 on a single GPU:
```
python train_test.py -c cfg/fasternet_t0.yaml --checkpoint_path model_ckpt/last.ckpt --data_dir /data/imagenet1k --test_phase -g 1 -e 125
```

- For evaluating other model variants, change `-c`, `--checkpoint_path` accordingly. You can get the pre-trained models from the tables above. 
- For multi-GPU evaluation, change `-g` to a larger number or a list, e.g., `8` or `0,1,2,3,4,5,6,7`. Note that the batch size for evaluation 
should be changed accordingly, e.g., change `-e` from `125` to `1000`.

To measure the latency on CPU/ARM and throughput on GPU (if any), run
```
python train_test.py -c cfg/fasternet_t0.yaml \
--checkpoint_path model_ckpt/fastermlp-epoch=283-val_acc1=72.9740.pth \
--data_dir ../../data/imagenet --test_phase -g 1 -e 32  --measure_latency --fuse_conv_bn
```
 
- `-e` controls the batch size of input on GPU while the batch size of input is fixed internally to 1 on CPU/ARM.  


### 5. Training

FasterMLP-S training on ImageNet-1K with a 8-GPU node:
```
python train_test.py -g 1 --num_nodes 1 -n 4 -b 64 -e 64 --data_dir ./path/to/date --pin_memory --wandb_project_name fasternet  --model_ckpt_dir ./model_ckpt/ --cfg cfg/fasternet_t0.yaml --wandb_offline
```
To train other FasterNet variants, `--cfg` need to be changed. You may also want to change the training batch size `-b`.       
  
