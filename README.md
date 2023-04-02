**0. Participant**

**GDSC GIST**

1. Hee-Jin Seo
2. Gabin Kim
3. Jaewoo Park
4. Suyong Huh



ECWALL-2023-GDSC-Solution-Challenge

**1. Description**

This program was created to raise awareness of discriminatory remarks in YouTube videos. In particular, in the case of young children, if they encounter stimulating information in a state where their values are not properly established, this can have a negative impact. Therefore, as a way to solve this problem, an ECWALL program was created that enables easy visual discrimination of whether a specific sentence contains a discriminatory word.

* * *
**2. Environment**

 Compiler : VSCode (There is a problem if you use 'Spyder')
 
* * *
**3. Prerequisite**
```
pip install flask
pip install youtube_transcript_api
pip install panda
pip install matplotlib
pip install tensorflow
pip install keras
```
matplotlib, tensorflow, keras : Use command prompt(CMD) with administrator privileges

(To start a project, put 'flask run' in your compiler.)

* * *
**4. Files**
- input.html

This is the part that receives user input. It receives the YouTube link and passes it to app.py.
Next, it receives the .srt subtitle file from app.py and provides it to the user.
- app.py

After extracting subtitles (text and time information) from the youtube link received from input.html, analyze the text using tensorflow to check whether there is a discrimination element in each sentence. Afterwards, the srt file with the subtitle color changed according to whether there is a discriminatory remark (the subtitle turns red when discriminatory remarks are included) is sent back to input.html.
- labeled_data.csv

This is a file used when tensorflow is trained, and discriminatory remarks are stored.

* * *
**5. Usage**

Users copy the YouTube link they want to watch, paste it on the ECWALL site, and submit it by clicking the submit button. After receiving the srt subtitle file, it is inserted into YouTube using the Google extension program, and discriminatory remarks are checked while observing the color change of the subtitle.

* * *
**6. Progress and plans**

progress
1. The AI team and the FE team are divided and start developing separately.
(compiler: AI - Spyder, FE - VSCode)
(languate: AI - tensorflow, FE - Flask)
2. After each development was completed, Spyder tried to link it, but failed.
(Trouble running Flask from spyder)
3. Integration success in VSCode

plans
1. problem: 
- The current version has changed the color of the subtitles, but there is a problem that you don't know which part contains discriminatory remarks unless you directly play the part where the subtitles appear. 
- It only works in the local environment yet.

2. Solution: 
- We are considering ways to display it directly on the progress bar, like YouTube's timestamp function.
- Proceed with distribution
