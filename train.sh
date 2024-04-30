# export CUDA_VISIBLE_DEVICES=0,1,2,3;
python train.py --name p_trained --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot /Users/johnson/Downloads/fake_detection --batch_size 32 --lr 0.00005 --gpu_ids -1 --loadSize 64 --cropSize 224
