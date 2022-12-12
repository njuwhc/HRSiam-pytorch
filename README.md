# HRSiam-pytorch

## 代码说明

* `config.py`：其中包含网络在训练中的各种参数，如学习率、epoch、epoch

* `dataset.py`：包含两个类GOT10kDataset和OTBDataset均继承Dataset，用于读取数据集，以便Data loader进行迭代，主要函数为__getitem__。

* `GMM.py`：混合高斯模型

* `loss.py`：定义两个损失函数

* `network.py`：原来的代码，原始模型，在此基础上进行魔改

* `hrsiam.py`：上面👆魔改后的代码，HRSiamRPNNet为网络模型

* `kalman_filter.py`：卡尔曼滤波器

* `utils.py`：常用的算法

* `track.py`：原来的代码，利用network.py中的模型构建的追踪器，用来完成追踪任务

* `hr_tracker.py`：在上面👆的基础上魔改的代码，构建的追踪器

* got10k中的文件不用管

* `hr_train.py`：使用GOT-10k数据集训练HRSiamRPNNet

  修改数据集路径，确保cuda可用应该可以正常运行，会自动保存

  ```python
  if __name__ == '__main__':
      train('/home/wanghaochen/Download/GOT-10k/GOT10k/full_data')
  ```

* `hr_test.py`：用来测试训练模型，暂时不需要使用

* `hr_train_otb.py`：使用OTB2015数据集训练HRSiamRPNNet，该模型为benchmark，不具备训练价值

  修改方式同上

* `demo.py`：使用训练出的模型来完成追踪任务，需要指定训练模型的路径和测试数据的路径

## 数据集

### [OTB2015](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.htmlhttp://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)

可以下几个跑试试看，例如把几个文件夹放在一个大文件下

### [GOT-10k](https://opendatalab.com/GOT-10k/cli)

完整数据集66G，比较大
