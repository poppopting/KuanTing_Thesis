
=======requirements.txt =======
執行所有程式所需import的packages
在command window中打:
pip install -r requirements.txt 

便會安裝所有套件
對於非linux版本(windows)，Cython 可能會出現windows install cython error
"unable to find vcvarsall.bat"，可參考下面網站安裝visual studio 
https://jarvus.dragonbeef.net/note/noteCython.php

======= Ting_MF =======
存放自己寫的package和function，包含四個主要的矩陣分解模型

======= SimulationOrdReg(table1) =======
simulation ordinal regrssion model
包含生成Table1的所有程式

======= simulationOrdRec(table2to3) =======
simulation ordinal recommender system
包含生成Table2, 3的所有程式

======= RealApplication =======
將四個主要的模型套用到movielens-100k, movielens-1M 
包含資料集下載、切分、模型配置、計算梯度值，有生成Table4-9的程式