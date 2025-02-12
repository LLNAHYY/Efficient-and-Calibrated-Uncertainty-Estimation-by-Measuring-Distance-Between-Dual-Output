python ./study/qu_main/main_ensemble.py --seeds 0,93,7777
python ./study/qu_main/main_ensemble.py --seeds 0,93,7777 --model_name resnet34

python ./study/qu_main/main_masked_ensemble.py --seeds 0,93,7777
python ./study/qu_main/main_masked_ensemble.py --seeds 0,93,7777 --model_name resnet34

 
python ./study/qu_main/main_mcdropout.py --seeds 0,93,7777
python ./study/qu_main/main_mcdropout.py --seeds 0,93,7777 --model_name resnet34


python ./study/qu_main/main_normal.py --seeds 0,93,7777 --model_name resnet18
python ./study/qu_main/main_normal.py --seeds 0,93,7777 --model_name resnet34


python ./study/qu_main/main_duq.py --seeds 0,93,7777 --l_gradient_penalty 0.01 --model_name resnet18
python ./study/qu_main/main_duq.py --seeds 0,93,7777 --model_name resnet34 --l_gradient_penalty 0.01


python ./study/qu_main/main_zigzag.py --seeds 0,93,7777
python ./study/qu_main/main_zigzag.py --seeds 0,93,7777 --model_name resnet34
