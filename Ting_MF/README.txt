===========check開頭 gradient結尾的四個.py檔 ==========
計算各訓練完的模型，其對應的梯度值

===========my_metric.py ============
包含svd和ordinal模型的預測函數，以及FCP的函數 

===========my_split.py =============
論文5.1中的提出的切分資料集的方法

===========setup.py ================
將存成.pyx(不能存.py檔，否則不能轉換)的模型檔案轉成可以被import 的package

轉換方式是在command window中打:
python setup.py build_ext --inplace

轉換後會多出現 .c 及 python extension檔 (執行過程出現的warnings 可忽略)

============build ===============
透過執行setup.py會自動產生的資料夾 可忽略


===========剩餘的檔案 ==============
即為四個主要模型的程式(用Cython寫的)
 


