vocals.wav : 60초 길이의 보컬 분리된 음원입니다.

estimated.npy : 해당 음원을 네트워크에 넣어서 출력된 prob. matrix 입니다. 크기는 [time_step(2580), phoneme class(112)]

utils.py : phoneme table을 비롯한 util들이 포함되어있습니다.

ground_truth.txt : 60초 길이에 포함된 실제 가사 및 포님 입니다. 


'''목표'''

estimated.npy 에 포함된 확률값을 바탕으로
ground_truth.txt에 포함된 phoneme seq의 각 phoneme이 
[0,2580] 중 어디서 부터 어디까지를 담당하는지, time index를 찾아내는 알고리즘 고민하기
Keyword : forced alignment, DTW, Lyric alignment
paper : https://arxiv.org/abs/1902.06797

혹시 궁금한 점은 01064956991 또는 406호에서 물어봐주세요!
생각을 도와주셔서 감사합니다 :)
