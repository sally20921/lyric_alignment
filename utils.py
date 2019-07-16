# waveform load
import librosa
wave, sr = librosa.load('wave_path')


# phoneme_table
Hangul = np.asarray([' ', 'ㅂ', 'ㅍ' ,'ㅃ', 'ㄷ', 'ㅌ', 'ㄸ', 'ㄱ', 'ㅋ', 'ㄲ', 'ㅅ', 'ㅆ', 'ㅎ', 'ㅈ', 'ㅊ', 'ㅉ', 'ㅁ', 'ㄴ', 'ㄹ', 'ㅂ', 'ㄷ', 
            'ㄱ', 'ㅁ', 'ㄴ', 'ㅇ', 'ㄹ', 'ㄱㅅ', 'ㄴㅈ', 'ㄴㅎ', 'ㄹㄱ', 'ㄹㅁ', 'ㄹㅂ', 'ㄹㅅ', 'ㄹㅌ', 'ㄹㅍ', 'ㄹㅎ',
            'ㅂㅅ', 'ㅣ', 'ㅔ', 'ㅐ', 'ㅏ', 'ㅡ', 'ㅓ', 'ㅜ', 'ㅗ', 'ㅖ', 'ㅒ', 'ㅑ', 'ㅕ', 'ㅠ', 'ㅛ', 'ㅟ', 'ㅚ', 'ㅙ', 'ㅞ', 'ㅘ', 'ㅝ', 'ㅢ',
            'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
            '0','1','2','3','4','5','6','7','8','9', "'", '.', ',', '!', '?', ':', '’', '-','*','`','&','火', '葬', '%', '/', '有','‘', '-'])

# prob to text seq
def prob_to_text_seq(prob):
    result = ''
    for item in prob:
        result += Hangul[np.argmax(item)]
    return result



import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')


import matplotlib.font_manager as fm

font_location = '/home/home/juheon/NanumSquareEB.ttf'  
                    # ex - 'C:/asiahead4.ttf'
font_name = fm.FontProperties(fname = font_location).get_name()
matplotlib.rc('font', family = font_name)



fontprop = fm.FontProperties(fname=font_location, size=2)


import librosa
import matplotlib.pyplot as plt 
import glob
from scipy.interpolate import interp1d


def phoneme_to_grapheme(phoneme_seq):
    table = np.asarray(['p0', 'ph', 'pp', 't0', 'th', 'tt', 'k0', 'kh', 'kk', 's0', 'ss', 'h0', 'c0', 'ch', 'cc', 'mm', 'nn', 'rr', 'pf', 'tf', 
            'kf', 'mf', 'nf', 'ng', 'll', 'ks', 'nc', 'nh', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh', 
            'ps', 'ii', 'ee', 'qq', 'aa', 'xx', 'vv', 'uu', 'oo', 'ye', 'yq', 'ya', 'yv', 'yu', 'yo', 'wi', 'wo', 'wq', 'we', 'wa', 'wv', 'xi'])
    Hangul = np.asarray(['ㅂ', 'ㅍ' ,'ㅃ', 'ㄷ', 'ㅌ', 'ㄸ', 'ㄱ', 'ㅋ', 'ㄲ', 'ㅅ', 'ㅆ', 'ㅎ', 'ㅈ', 'ㅊ', 'ㅉ', 'ㅁ', 'ㄴ', 'ㄹ', 'ㅂ', 'ㄷ', 
            'ㄱ', 'ㅁ', 'ㄴ', 'ㅇ', 'ㄹ', 'ㄱㅅ', 'ㄴㅈ', 'ㄴㅎ', 'ㄹㄱ', 'ㄹㅁ', 'ㄹㅂ', 'ㄹㅅ', 'ㄹㅌ', 'ㄹㅍ', 'ㄹㅎ',
            'ㅂㅅ', 'ㅣ', 'ㅔ', 'ㅐ', 'ㅏ', 'ㅡ', 'ㅓ', 'ㅜ', 'ㅗ', 'ㅖ', 'ㅒ', 'ㅑ', 'ㅕ', 'ㅠ', 'ㅛ', 'ㅟ', 'ㅚ', 'ㅙ', 'ㅞ', 'ㅘ', 'ㅝ', 'ㅢ',
            'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
            '0','1','2','3','4','5','6','7','8','9', "'", '.', ',', '!', '?', ':', '’', '-','*','`','&','火', '葬', '%', '/', '有','‘'])
    result = ''
    for item in phoneme_seq:
        if item == 0:
            result += ' '
        else :
            result += Hangul[item-1]
    return result


sess = tf.Session() 
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('/home/home/juheon/007_KAKAO_lyric/train_result/MODEL_baseline_sepa/model.ckpt-470000.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/home/juheon/007_KAKAO_lyric/train_result/MODEL_baseline_sepa/'))


# Graph 에서 가져올 변수들 지정
graph = tf.get_default_graph()

wave_input = graph.get_tensor_by_name("Placeholder:0")
output_prob = graph.get_tensor_by_name("conv1d_7/LeakyRelu:0")
output_prob_sm = graph.get_tensor_by_name("Reshape_1:0")
seq_len = graph.get_tensor_by_name("Placeholder_4:0")
logits = graph.get_tensor_by_name("transpose_6:0")
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

import librosa
'/home/HDD1/DATA/KAKAO/song_separated/%EC%B9%B4%EB%8D%94%EA%B0%80%EB%93%A0/0830781652/vocals.wav'

wave,sr = librosa.load('/home/HDD1/DATA/KAKAO/song_separated/카더가든/0830781652/vocals.wav')

Hangul = np.asarray(['ㅂ', 'ㅍ' ,'ㅃ', 'ㄷ', 'ㅌ', 'ㄸ', 'ㄱ', 'ㅋ', 'ㄲ', 'ㅅ', 'ㅆ', 'ㅎ', 'ㅈ', 'ㅊ', 'ㅉ', 'ㅁ', 'ㄴ', 'ㄹ', 'ㅂ', 'ㄷ', 
            'ㄱ', 'ㅁ', 'ㄴ', 'ㅇ', 'ㄹ', 'ㄱㅅ', 'ㄴㅈ', 'ㄴㅎ', 'ㄹㄱ', 'ㄹㅁ', 'ㄹㅂ', 'ㄹㅅ', 'ㄹㅌ', 'ㄹㅍ', 'ㄹㅎ',
            'ㅂㅅ', 'ㅣ', 'ㅔ', 'ㅐ', 'ㅏ', 'ㅡ', 'ㅓ', 'ㅜ', 'ㅗ', 'ㅖ', 'ㅒ', 'ㅑ', 'ㅕ', 'ㅠ', 'ㅛ', 'ㅟ', 'ㅚ', 'ㅙ', 'ㅞ', 'ㅘ', 'ㅝ', 'ㅢ',
            'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
            '0','1','2','3','4','5','6','7','8','9', "'", '.', ',', '!', '?', ':', '’', '-','*','`','&','火', '葬', '%', '/', '有','‘'])


total_result =''
decoding = []
total_output_prob_sm_ = np.zeros((1, 1, 112))
for i in range(12):
    plt.clf()
    input_wave = wave[(i*5)*sr : (i*5+5)*sr]
    stft = np.log(np.abs(librosa.stft(input_wave, n_fft=1024, hop_length=16))+1e-5)
    input_wave = np.expand_dims(input_wave, axis=-1)
    input_wave = np.expand_dims(input_wave, axis=0)
    seq_len_ = [215]
    feed_dict = {wave_input:input_wave,
                seq_len : seq_len_
                }
    run_param = [output_prob, output_prob_sm, decoded[0]]
    output_prob_, output_prob_sm_, d_ = sess.run(run_param, feed_dict)
    print(d_.values)
    decoding = np.concatenate((decoding, d_.values))
    result = ''
    for j in range(215):
        temp = np.argmax(output_prob_sm_[0][j])
        if temp == 111:
            result += '-'
        elif temp == 0:
            result += ' '
        else :
            temp_char = Hangul[temp-1]
            result += temp_char
    total_result += result
    total_output_prob_sm_ = np.concatenate((total_output_prob_sm_, output_prob_sm_), axis=1)
    # plt.imshow(stft)
    # for index, char in enumerate(result):
    #     plt.text(index*sr*5/(215*16), 200, char, fontproperties=fontprop, color='white')
    # plt.savefig('./test_kor_eng_' + (str)(i) + '.png', dpi=1000)
    # plt.clf()
    # print((str)(i*5) + 's ~ ' + (str)(i*5+5) + 's :')
    # print(result)

# 5 초 -> 215 칸 
# 1 칸 -> 5/215 초

def kor_seq_to_vec(temp):
    result = []
    for item in temp:
        if item == '-':
            result.append(np.eye(112)[111])
        else :
            result.append(np.eye(112)[np.where(Hangul==item)[0][0]+1])
    return result


def vec_seq_to_kor(temp):
    result = ''
    for item in temp:
        if np.argmax(item) == 111:
            result += '-'
        elif np.argmax(item) == 0:
            continue
        else :
            result += Hangul[np.argmax(item)-1]
    return result



without_blank = []
without_blank_index = []
for index, item in enumerate(total_output_prob_sm_[0]):
    if index == 0:
        continue
    if np.argmax(item) != 111 and np.argmax(item) != 0:
        without_blank.append(item)
        without_blank_index.append(index)


from g2p import runKoG2P
#estimation = total_output_prob_sm_[0]
estimation = np.asarray(without_blank)
estimation_text = vec_seq_to_kor(estimation)
gt = '베인마음속에난무얼찾아볼수있을까서투른판단에더울적해진채로보낸밤우린선을그어버릴까'
gt_phoneme = runKoG2P(gt, 'rulebook.txt')
gt_list = ''
for item in gt_phoneme.split(' '):
    gt_list += Hangul[np.where(table==item)[0][0]] 


gt_vec = kor_seq_to_vec(gt_list)


import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

distance,path = fastdtw(estimation, gt_vec, dist=euclidean)

final_timing = np.zeros(len(gt_vec),1)
for item in path:
    if final_timing[item[1]] != 0:
        continue
    else :
        final_timing[item[1]] = without_blank_index[item[0]]

np_path = np.asarray(path)
temp = np_path[:,0]
final = ''
for item in temp:
    final += gt_list[item]


input_wave = wave[0*sr : 10*sr]
stft = np.log(np.abs(librosa.stft(input_wave, n_fft=1024, hop_length=16))+1e-5)
plt.imshow(stft)
for index, char in enumerate(final):
    plt.text(index*sr*5/(2150*16), 200, char, fontproperties=fontprop, color='white')
plt.savefig('./HYUK_' + (str)(i) + '.png', dpi=1000)
plt.clf()
print((str)(i*5) + 's ~ ' + (str)(i*5+5) + 's :')
print(final)



total_result = total_result.replace(' ', '-')
result = []
start = 0
start_flag = 0
for index, item in enumerate(total_result):    
    if index == len(total_result)-1:
        break
    else :
        if item == '-' and total_result[index+1] != '-':
            result.append([start, index])
            start_flag = 0
        if item == '-' and total_result[index+1] == '-':
            continue
        if item != '-' and start_flag == 0:
            start = index
            start_flag = 1
        if item != '-' and start_flag == 1:
            continue

for index, item in enumerate(result):
    librosa.output.write_wav('/home/home/juheon/007_KAKAO_lyric/segmented_sample/' + 'SWJA_' + (str)(index) + '.wav' ,wave[(int)(item[0]*5*sr/215):(int)(item[1]*5*sr/215)], sr)


def viterbi_segment(text, P):
    """Find the best segmentation of the string of characters, given the
    UnigramTextModel P."""
    # best[i] = best probability for text[0:i]
    # words[i] = best word ending at position i
    n = len(text)
    words = [''] + list(text)
    best = [1.0] + [0.0] * n
    ## Fill in the vectors best, words via dynamic programming
    for i in range(n+1):
        for j in range(0, i):
            w = text[j:i]
            if P[w] * best[i - len(w)] >= best[i]:
                best[i] = P[w] * best[i - len(w)]
                words[i] = w
    ## Now recover the sequence of best words
    sequence = []; i = len(words)-1
    while i > 0:
        sequence[0:0] = [words[i]]
        i = i - len(words[i])
    ## Return sequence of best words and overall probability
    return sequence, best[-1]



def decoding_to_grapheme(decoding):
    list = ''
    for item in decoding:
        if (int)(item) == 0 :
            char = ' '
        else :
            char = Hangul[(int)(item-1)]
        list = list + char
    return list



table = np.asarray(['p0', 'ph', 'pp', 't0', 'th', 'tt', 'k0', 'kh', 'kk', 's0', 'ss', 'h0', 'c0', 'ch', 'cc', 'mm', 'nn', 'rr', 'pf', 'tf', 
            'kf', 'mf', 'nf', 'ng', 'll', 'ks', 'nc', 'nh', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh', 
            'ps', 'ii', 'ee', 'qq', 'aa', 'xx', 'vv', 'uu', 'oo', 'ye', 'yq', 'ya', 'yv', 'yu', 'yo', 'wi', 'wo', 'wq', 'we', 'wa', 'wv', 'xi'])



temp = total_result[0:900]
gt = '그때난어떤맘이었길래내모든걸주고도웃을수있었나'
gt_phoneme = runKoG2P(gt, 'rulebook.txt')
gt_list = ''
for item in gt_phoneme.split(' '):
    gt_list += Hangul[np.where(table==item)[0][0]]



def kor_seq_to_vec(temp):
    result = []
    for item in temp:
        if item == '-':
            result.append(np.ones(110)*-1)
        elif item == ' ':
            result.append(np.zeros(110))
        else :
            result.append(np.eye(110)[np.where(Hangul==item)[0][0]])
    return result

gt_vec = kor_seq_to_vec(gt_list)
tp_vec = kor_seq_to_vec(temp)


import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

distance,path = fastdtw(gt_vec, tp_vec, dist=euclidean)





input_wave = wave[0*sr:5*sr]
stft = librosa.stft(input_wave, n_fft=1024, hop_length=256)
plt.imshow(np.log(np.abs(stft)+1e-5))
for index, item in enumerate(total_result):
    plt.text(60, .025, total_result[i])


def total_result_separator(total_result):
    