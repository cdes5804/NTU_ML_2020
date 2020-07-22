本次作業因為hw1.sh跟hw1_best.sh都是使用gradient descent的方式，因此兩份script都使用同一個python檔案。

若要獨立執行Python code，執行的方式為
python3 lin_reg.py [test.csv] [result.csv]
若只是要檢驗training而沒有要實際做test或輸出，則[test.csv]與[result.csv]都可以省略


建立object時，可以用的參數如下：

time_len: 代表要取前幾個小時的資料，範圍需為1-9之間，預設為9

source: 要使用哪些feature。請將要用的features放進一個list裡面，且必須至少包含PM2.5。若有加入其他feature，請務必按照train.csv裡每個feature出現的順序放在list內

w: 若想直接使用pre-trained好的w作testing，可以在這裡指定。型態需為numpy.array

no_train: 是否有要進行training。可將no_train設為True，此時將不會去讀取train.csv，否則請將train.csv放在同目錄下，並且呼叫train()的函式來進行training

若要進行training，可以用的參數如下：

iter: 進行gradient descent的次數，預設為5000

validate: 要切出的validation set的大小，隨機切出相同數量的data後，剩下的作為training set進行training。若數值為0則所有data都會被當作training set使用。預設為0

verbose: 是否每次gradient descent後都會輸出目前在training set與validation set上的loss。預設為False，不進行輸出。

seed: 隨機切出validation set使用的seed。若validate=0，則此項可忽略。

若要進行testing，需要的參數為：

test_file: test data的位置

result_file: 答案要輸出的檔案位置
