# README

上一版是在python3.7, cuda10.0, tensorflow2.3, pytorch1.7.1来运行的，这次尝试使用python3.10, cuda11.8, tensorflow2.10.1, pytorch2.0.0来作为运行环境

**step 0:** 准备数据。

将.mat格式的高光谱数据文件（如Pavia.mat和Pavia_gt.mat）放入正确的文件夹（指定为：C:\Matlab练习\duogun，也可以是DEFG盘的相同名称文件夹） 

**step 1:** 运行HSI_Search.py，运行结束后所获得的最优神经网络结构将被自动写入genotypes.py的最后面。 **step 2:** 运行HSI_Classificaion.py，将自动读取step1中的最优神经网络结构。 

**step 3:** 查看运行结果。

step 1和step 2运行过程中的控制台输出信息保存在1D-Auto-CNN-Spectral-Classification-Tensorflow2_10_1\result\log.txt中， 分类结果保存于1D-Auto-CNN-Spectral-Classification-Tensorflow2_10_1\result\classification_Pavia.txt中