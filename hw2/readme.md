建立object時可以調整參數no_train: 若為True，則不會去讀取training相關的檔案，用預設的model直接進行testing。若為False，請將X_train與Y_train放在同目錄下，並呼叫train()函式進行training。

各函式說明如下：

normalize(): 對資料進行normalization。若is_train為True則對training data操作，否則對testing data操作。

accuracy(): 獲得當前model的Accuracy。回傳值為tuple，第一個數值為training set的正確率，第二個數值為validation set的正確率。若沒有切validation set則第二個數值為None。若verbose設為True則回傳前會輸出正確率。

cross_entropy(): 獲得當前model的cross entropy。模式與參數與accuracy()相同。

train(): 進行training。iter參數調整回圈次數，verbose調整是否每次更新都輸出當前accuracy與cross entropy。
