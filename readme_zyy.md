整体实现顺序：
1. mamadroid、drebin、apigraph 三种方法 在 三种检测方法的所有实验结果
2. classifier 重新进行训练
3. 探究 va_fed方法
4. 探究BagMoo方法实验结果

一. 写一个DataSet类，将DataSet与Model解耦   [10点前必须完成]
1. x_train、x_test 仅与detection有关，与classifier无关，e.g.,mamaDroid-family、mamaDroid-package、drebin
2. x_adv与detection有关、也与classifier有关，比如mamaDroid-family-rf、mamaDroid-family-svm生成的对抗样本数量不同
3. 从x_test选择等量数据与x_adv进行混合

总结，DataSet取决于Attack(AdvDroidZero)、detection(mamaDroid-family)、classifier(drebin)

二、写一个Model类，能够训练和加载模型  [11点50前必须完成]
1. 厘清数据集的拆分与过滤
    [原有逻辑]
    - 测试集数量3060，其中前1470个样本为恶意样本
    - 对抗样本数据集为1470个样本
    - 根据classifier过滤测试样本剩余2818个
    - 同样对应过滤对抗样本
    [现行逻辑]
    - 测试集数量3072个
    - 从中随机抽出对抗样本200个(满足真实为恶意样本且被classifier真实分类为恶意样本)
    - 攻击成功的样本数量为150+
    - 从测试样本中随机抽取同等数量的良性样本进行检测？ 
      [是否要求classifier分类正确，如果要求，取决于原有分类器还是替代模型的reply]
2. 自定义一个NN模型，加载RF的model，根据RF的model的reply值来训练NN模型
    - RF model的存储位置与加载
    - NN模型的结构定义
    - NN模型的训练[重点]
    - NN模型的保存
3. 模型仅与detection、classifier有关，与attack等均无关
    - 将模型复制到公共文件夹(或考虑重新生成位置保存结果)
    - 加载模型并进行预测(需要安装相关环境)
    - 改写detection.py


三、根据DataSet类与Model类，实现AAD方法的检测结果   [16点前必须完成]
1. 计算AAD的检测结果
   - 寻找GPU, 保存模型   epoch调小，调为10先试下结果
   - 是否绘图等参数命令行化、数据结果按照文件夹分类保存至相应的文件夹下(提前创建相应的文件夹)
   - [改进建议：新建一个Attacker类和一个Detector类，结构会更好]
2. 再生成 drebin-svm、apigraph-svm的对抗样本    同步修改detection-FS、detection-MagNet、detection-LID
   - 先调试成功
   - 修改为等量混合数据
   - 调试结果
3. 绘制实验结果Table表
4. 获取全部的detection-sharpness结果


四、实现FS、MagNet、LID方法的检测结果    [18点前必须完成]
1. 实现detection_fs.py, dataset可复用; model需要重新实现，命名为model/nn_tensorflow.py
2. 

5.6 开始复现 BagAmmo
一、找到生成对抗样本的npy文件





问题记录：
1. 可替代模型拟合程度不够：可替代模型训练后，对样本的准确率比较低，仅有0.887
2. 测试和对抗数据集选择问题：
   - AdvDroidZero 生成对抗样本时也为每个样本生成原始样本
3. sharpness检测结果很差
   - 模型原因：拟合模型训练不够、训练数据集有问题导致模型训练不合适(数据处理出了问题)  [直接加载原有模型]
     - 直接加载根据HIV提供的数据样本，发现在test集上的结果很差。说明不同的mamaDroid处理的数据集完全不能共用
   - 数据原因：对抗样本与测试样本比例严重不均衡   [针对对抗样本重新准备 x_adv_orig]
     - 取等量数据集，发现结果要好很多
   - 攻击方法原因：sharpness无法解决AdvDroidZero方法生成的对抗样本
