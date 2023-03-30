#Jaewoo's import 
from flask import Flask, render_template, request, send_file, redirect, url_for
import sys
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
import pandas as pd
import csv
import os, glob
import time

#Heejin's import
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
#1. Jaewoo's part : #Download information of youtube video script using Flask 
app = Flask(__name__)

@app.route("/")
def hello():
	return render_template('input.html')
#posted by 'input.html'
@app.route("/post",methods=['POST']) 
def post():
	value = request.form['input'].split("=")
	if(value[0]!="https://www.youtube.com/watch?v"):
		return "그거 아님다"
	else:
		srt = YouTubeTranscriptApi.get_transcript(value[1], languages = ['en'])
	#youtube script text by .csv
	tweet = []
	for i in range(len(srt)): 
		tweet.append(srt[i]['text'])
	df2 = pd.DataFrame(tweet, columns=['tweet'])
	df2.to_csv("{}.csv".format(value[1]), index = False)
	#############################################################################

	#2. Heejin's part: 

	#loading until csv file exists
	path=os.path.dirname(os.path.realpath(__file__))
	count = 1
	while not glob.glob(os.path.join(path,"{}.csv".format(value[1]))):
		print(count)
		count+=1
		time.sleep(1)

		print('totensorflow',time)

	df=pd.read_csv('data/labeled_data.csv')
	print('총 샘플의 수 :',len(df))
	data=df[['hate_speech','tweet']].copy()
	data.rename(columns={'hate_speech':'v1','tweet':'v2'},inplace=True)
	#v1 1,2,3...로 있어서 v1이 1 이상이면 1 값으로 
	data['v1']=np.where(data['v1']>0,1,0)  
	data['v1'].value_counts().plot(kind='bar');

	print(data.groupby('v1').size().reset_index(name='count'))
	#Mix the data
	data = data.sample(frac=1).reset_index(drop=True)

	X_data = data['v2']
	y_data = data['v1']

	df_len=[ len(i) for i in X_data]
	max_len=max(df_len)

	# max_len=max_len*2

	print('메일 본문의 개수: {}'.format(len(X_data)))
	print('레이블의 개수: {}'.format(len(y_data)))

	#전체 단어의 개수를 1,000개로 제한하고 정수 인코딩을 진행합니다.
	vocab_size = 1000
	tokenizer = Tokenizer(num_words = vocab_size)
	tokenizer.fit_on_texts(X_data) # 5169개의 행을 가진 X의 각 행에 토큰화를 수행
	sequences = tokenizer.texts_to_sequences(X_data) # 단어를 숫자값, 인덱스로 변환하여 저장

	# 상위 5개의 샘플을 출력해봅시다.

	print(sequences[:5])

	#훈련 데이터와 테스트 데이터의 분리 비율 결정 (8:2)
	n_of_train = int(len(sequences)*0.8)
	n_of_test = int(len(sequences)-n_of_train)
	print('훈련 데이터의 개수:',n_of_train)
	print('테스트 데이터의 개수:',n_of_test)

	#전체 데이터에서 가장 길이가 긴 메일과, 전체 메일 데이터의 길이 분포 알아보기
	X_data = sequences
	print('data 최대 길이:%d' %max(len(I) for I in X_data))
	print('data 평균 길이:%f' %(sum(map(len,X_data))/len(X_data)))
	plt.hist([len(s) for s in X_data],bins=50)
	plt.xlabel('length of samples')
	plt.ylabel('number of samples')
	plt.show

	#전체 데이터셋의 길이를 max_len으로 맞춘다.
	# max_len=172
	data = pad_sequences(X_data, maxlen = max_len)
	print("훈련 데이터의 크기(shape):",data.shape)

	#maxlen에는 가장 긴 메일의 길이였던 172라는 숫자를 넣음.
	#이는 5169개의 X_data길이를 전부 172로 바꿈.
	#172보다 길이가 짧은 메일 샘플은 전부 숫자 0이 패딩되어 172의 길이를 가짐.

	#이제 X_data는 5169 * 172 의 크기를 가지게 됨.
	#이제 X_train과 X_test를 분리(??)

	X_train = data[:int(len(data)*0.7)] #X_data 데이터 중에서 앞의 4135개의 데이터, 즉 70%만 저장
	y_train = np.array(y_data[:int(len(data)*0.7)])

	X_val = data[int(len(data)*0.7):int(len(data)*0.9)]
	y_val = np.array(y_data[int(len(data)*0.7):int(len(data)*0.9)])

	X_test = data[int(len(data)*0.9):]
	y_test = np.array(y_data[int(len(data)*0.9):])



	from keras.layers import Dense, Embedding, LSTM
	from keras.layers import Flatten, Dropout
	from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
	from keras.models import Sequential
	from keras.callbacks import EarlyStopping, ModelCheckpoint


	cnn_lstm_model = Sequential()

	# cnn_lstm_model.add(Embedding(1000, 128, input_length=172))
	cnn_lstm_model.add(Embedding(1000, 128, input_length=max_len))
	cnn_lstm_model.add(Dropout(0.2))
	cnn_lstm_model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
	cnn_lstm_model.add(MaxPooling1D(pool_size=4))
	cnn_lstm_model.add(LSTM(128))
	cnn_lstm_model.add(Dense(1, activation='sigmoid'))
	cnn_lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

	# es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
	es = EarlyStopping(monitor = 'val_acc', mode = 'min', verbose = 1, patience = 3)
	mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

	# .fit: 학습시키기
	history = cnn_lstm_model.fit(X_train, y_train,
    	        epochs = 1,
    	        batch_size = 64,
        	    validation_data = (X_val, y_val),
            	callbacks = [es,mc])

	print("\n 테스트 정확도: %.4f" %(cnn_lstm_model.evaluate(X_test, y_test)[1]))
	pred = cnn_lstm_model.predict(X_test)


	#4. put new csv file to trained model
	df2=pd.read_csv("{}.csv".format(value[1]))
	df2_copy=df2.copy()
	print('총 샘플의 수 :',len(df2))

	data2=df2[['tweet']].copy()
	data2.rename(columns={'tweet':'v2'},inplace=True)	

	#mix the data
	data2 = data2.sample(frac=1).reset_index(drop=True)
	X_data2 = data2['v2']
	df2_len=[ len(i) for i in X_data2]	
	max_len2=max(df2_len)

	# max_len2=max_len2*2
	print('메일 본문의 개수: {}'.format(len(X_data2)))

	#전체 단어의 개수를 1,000개로 제한하고 정수 인코딩을 진행합니다.
	vocab_size = 1000
	tokenizer = Tokenizer(num_words = vocab_size)
	tokenizer.fit_on_texts(X_data2) # 5169개의 행을 가진 X의 각 행에 토큰화를 수행
	sequences2 = tokenizer.texts_to_sequences(X_data2) # 단어를 숫자값, 인덱스로 변환하여 저장

	# 상위 5개의 샘플을 출력해봅시다.
	print(sequences2[:5])

	#전체 데이터셋의 길이를 max_len으로 맞춘다.
	# max_len= 000


	X_data2 = sequences2
	data2 = pad_sequences(X_data2, maxlen = max_len2)
	print("훈련 데이터의 크기(shape):",data2.shape)

	X_test2 = data2

	pred2 = cnn_lstm_model.predict(X_test2)

	#pred2의 복사본
	pred3 = pred2.copy()

	#5. return the result csv file

	df2_copy['hate speech'] = pred3
	#df2_copy.to_csv('C://Users//User//Desktop//hate_results//result.csv', sep=',', na_rep='NaN')
	df2_copy.to_csv("{}_result.csv".format(value[1]), sep=',', na_rep='NaN')

	#########################################################################################

	#3. Jaewoo's part: Make new subtitle file
	hate_speech = []
	text = ''
	result_text = []
	color = '#ffffff'
	criteria = 0.35
	 # convert time information in youtube_transcript_api to srt format
	def display_time(y):
		for i in range(len(srt)):
			startHour = int(y // 3600)
			startMin = int(y //60 - startHour*60)
			startSec = int(y % 60 //1)
			startMili = int(y % 1 * 1000)
			if startHour < 10 : 
				startHour = str(0) + str(startHour)
			if startMin < 10 : 
				startMin = str(0) + str(startMin)
			if startSec < 10 : 
				startSec = str(0) + str(startSec)
		return str(startHour) + ":" + str(startMin) + ":" + str(startSec) + "," + str(startMili)

	#loading until csv file exists
	path=os.path.dirname(os.path.realpath(__file__))
	count = 1
	while not glob.glob(os.path.join(path,"{}_result.csv".format(value[1]))):
		print(count)
		count+=1
		time.sleep(1)
		print('toflask',time)

	
	# get result file from tensorflow
	f = open("{}_result.csv".format(value[1]), 'r') 
	rdr = csv.reader(f)
	for line in rdr:
		hate_speech.append(line[2])
		result_text.append(line[1])
	f.close()

	# convert to srt format
	with open("{}.srt".format(value[1]), "w", encoding='utf-8') as SRTFormatter: 	
		for i in range(len(srt)) :		
			SRTFormatter.write("{}\n".format(i))
			SRTFormatter.write("{}\n".format(display_time(srt[i]['start']) + " --> " + display_time(srt[i]['start']+srt[i]['duration'])))
			# print(hate_speech[i+1])
			if float(hate_speech[i+1]) >criteria or '[' in result_text[i+1] :
				color = '#fc9d9d'
			else:
				color = '#ccddff'
			SRTFormatter.write("{}".format('<font color={}>'.format(color)))
			SRTFormatter.write("{}".format(srt[i]['text']))
			SRTFormatter.write("{}\n\n".format('</font>'))
	
	for i in range(len(srt)):
		text += str(i) + ' ' + srt[i]['text'] + '\r'
	PATH = "{}.srt".format(value[1])
	print(url_for('post'))
	return send_file(PATH, as_attachment=True)

if __name__ == "__main__":
		app.run(debug=True)