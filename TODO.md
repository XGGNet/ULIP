# 病理基因+ULIP
## 将病理基因部分的代码移植到ULIP; 先写单折的情况
- [X] 基因网络SNN
- [X] 基因dataset
  - [X] 尝试通过在shapenet-40上预训练, 学习dataset的输出
  - [X] 移植数据pipeline

- [X] 跑通基因CLIP 
  - 先做最简陋的搞法 (基因网络lr目前还是用点云CLIP的; 单折; 图像端完全不tune; 测试时只在基因端)


## 设计数据划分, 以及evaluation benchmark 
- [X] 其实先单折即可
- [X] 加上AP, AUC指标

## 急需解决的几个issue 07/16
- [] 增加visual端的 PEFT模块 
- [] vanilla text prompt for WHO grading labels provides insufficient semantic information for CLIP 

## debug & check list
1. bactch_size能随意调吗?

2. visual branch的性能变化很奇怪? Why?.. check下loss...
    1. 为啥加了visual adapter的 visual performance 几乎不怎么上涨..? --> 对比损失只加在 img-gene & gene-text, 因此visual branch的分类性能确实不会增长..
       1. check 下CG3D里的对比损失加在哪些分支? --> 微调时也会加在img&text之间.
    2. 为啥不加visual adapter的 visual performance 还会动...? --> 这里有点解释不通, 没动visual branch参数, 为啥输出还会动?
       1. try 把adapter相关的东西remove, 跑一遍只tune gene branch的, 看看visual performance 会不会变? --> 发现只tune gene branch, visual 和 generic branch的logits也都会变...好诡异.. --> * [问题找到了] **因为test_sample 被aug了...**
          1. check 优化参数里有无visual parameters --> 无
          2. try把optimizer全停了, visual 性能会不会动? --> 还是会, 看来因为是test_loader里的数据增强...
          3. check下是不是model里添加的随机project layer的问题?

/########## The above issues have been addressed ##########
  
1. 其实数据量问题没那么严重, 因为还涉及到patch-wise random sampling
2. [X] test data 去除 aug
3. [X] test data 采用slide window testing
   1. [X] verify test_dataset 是否正常?
   2. [X] verify test_dataset_patches 是否正常?
4. [X] 测试 test_data不变后, 把 备注掉的 optimzier恢复
5. [X] 只tune基因branch, 看下visual性能会不会变? -> 不变了, 正常了
6. 为啥 acc 和 AP&AUC的排序完全不同?
   1. check下师姐里论文 ACC / AP / AUC 的大致比例关系 -> 看上去感觉是师姐的MMT的三个指标的比例是对的; 而geneLIP的比例不大对的上..
      1.  check GeneLIP 里
         1. [X] metric code写错了? >>  好像看不出哪有明显错误
         2. [X] 输入metric calculator的pred/label写错了? >>  好像看不出哪有明显错误
      2. 好像 logits_per_omic 预测比较均匀的结果? 这个会一直保持嘛? >> 先skip吧, 整理最新结果看看; 如果还是有问题, 可以先只看acc, acc应该是对的

/####### Several fundamental bugs have been addressed ####
/####### Next, we mainly aim to get the baseline results as well we the VPT & descriptions from FM
1. 跑出单折&patches的结果对比
   1. [X] 师姐MICCAI22
   2. [X] GeneLIP 只tune gene branch 
      1. **Found bug 0719** loss 有点问题, omic和text的一个对齐项写成了omic和image的 --> 这样会导致gene的分类会差..  folder: <gene_GBMLGG_0719 [Bug]>
   3. [X] GeneLIP + tune visual_adapter + loss上加入vis和text的Con_loss
   4. [X] *GeneLIP + tune visual_prompt + loss上加入vis和text的Con_loss* >> 可能需要先跑通CG3D才行.
      1. [] ~~CG3D 装环境~~
      2. [] ~~跑通CG3D~~  -> 看了下, server自带的CUDA12.2版本太高. 目前的torch只对应到11.7.. 不然还是直接移植..
      3. ~~[] 跑通VPT~~
         1. ~~src/models/vit_backbones 常规VIT模型~~
         2. ~~src/models/vit_prompt  VIT + VPT~~
      4. [X] 代码移植 -> 直接把timm里的底层函数改了...或者重新写个model.... 比跑通一个其他的repo快多了..
         1. 为啥显存一直爆.... 只是加了一堆prompt呀... -> 因为SNN巨小, 就是MLP...
         2. prompt is verified to be tuned.
         3. ~~[] 调通并行模式...~~  >> 先不浪费时间; 往下做
         4. [X] 这里是比较粗暴的直接改了网络 只能用prompt; 后面可以改成可以参数选择是否需要Prompt 
            1. [X] 记录Deep prompt 结果
            2. [X] 记录Shallow prompt 结果
         5. [X] 直接fine-tune CLIP visual encoder的结果..
   5. [] Stronger CLIP backbone for pathology domain
      1. [X] Study >> TOP priority: BiomedCLIP;  2nd priority: MI-Zero 
      2. [X] load BiomedCLIP
         1. [X] *比较一下 biomedclip 是否会比 slip 在病理上的zero-shot性能 好?* >> biomedclip 在病理图像上的zero-shot性能 和 SLIP 差不多...
            1. [X] 测一下CLIP的病理ZS性能
            2. [X] check normalization的问题; 现在都是用 pathology dataset的normalization, 试下用CLIP/SLIP自带的. >> 发现还是data自带的norm效果佳
               1. ~~[] 为啥发现改了normalization, 基因的预测也会变...?~~
                  1. ~~verify text_input, omic_input; text_features会不会变~~
            3. [X] verify BiomedCLIP效果劣于SLIP
               1. 确认下 official demo里使用biomedclip做zero-shot classification的范式 >> 两种版本的预测结果是一样的...
               2. logit_scale对 ROC等的影响不大..
      3. [X] *load MI-Zero* >> 搞不动了... 不调了..
   
   6. [] class prompt ==> itemized descriptions
      1. [X] Stdudy FM
      2. [] Coding
         1. 把make_autopromptsv2.py用在基因图像数据上, 并且跑通
      3. 先随便用个LLM输出下description..; 然后测下zero-shot性能
         1. 
   7. [X] AC 和 AP / AUC 指标趋势不一致的问题 >> 暂时找不出任何问题了... 先不浪费时间查这个issue了..
      1. 首先study排除函数的问题, 两边函数进行同样的输入, 看输出是否一致?
      2. Finding: 加了softmax就会下降..
         1. MICCA code 里的roc计算有无问题? >> 无问题..
         2. accuracy计算? >> 无问题..
   8. [X] 留个心眼
      1.  normalization有点不一样 会不会有问题..
      2.  Loss里, 只要不是成对数据就排斥...可以要是类别名是一样的情况怎么办? [已发邮件询问]


## 0803 ##
- {EXP} full data + multi-modal + visual adapter
- {EXP} full data + multi-modal + visual adapter + fix gene
- {EXP} full data + multi-modal + visual adapter + w/o gene

- {EXP} 50% data + multi-modal + visual adapter
- {EXP} 50% data + multi-modal + visual adapter + fix gene
- {EXP} 50$ data + multi-modal + visual adapter + w/o gene
base2new_classbase2new_class
- {EXP} 50% sup data + 50% paired data + multi-modal + visual adapter
  - Note: 前50% train data 保留标注, 后50% 只保留和基因的匹配关系
- {EXP} 50% sup data + 50% paried data + multi-modal + visual adapter + fix gene

## 0815 ##
- [X] {EXP} supervised cont of image and gene
- [] few-shot data pipeline
- 

## 0829 ##
- [] {EXP} 三个loss的weight的比例如何设置才好
- [] {CODE} few-shot data pipeline
- [] {CODE} base_to_new generalization pipeline
- [] {THINK} design multi-modal graph..

## 0903 ##
- [] {EXP} add hete_graph, train in end2end
  - 不使用graph地正常训练完一个epoch后, 在updated feature上构图for every subject, then continue training for another epoch {这种方式in-efficient, 但是掉点的可能性就小很多}
    - 前面 非graph训练 fix不动
    - 一个epoch 结束后进入graph: 1. graph construction from mm data, 2. apply GCN, 3. apply training loss and graph model updating..
    - 对graph_data 首先不用任何data_loader和augmentation...
    - 先只考虑全batch的..

## 0905 ## 
- 移植乐权代码的思路
   - data pipeline 不用管, 直接输入graph即可
   - 直接调用HEAT4网络
- figure

- how to evaluate whether adding graph work?
    - torch200 不带 graph的结果
    - torch200 加graph (graph_cls), 但是推断不带graph的, 看下non-graph网络是否会被学的更好.. >> 发现graph_cls_loss很大..why? >> 如果只使用graph_cls_loss, 如何? 

- 核心问题: graph loss是否朝着预期的方向下降? watch train_graph_cls_loss, test_graph_cls_loss; 可以先把其他网络和loss都freeze, 就对feature 用graph去拟合; 看下能不能正常拟合?  
  - check metrics >> improve is done, so we found that the training loss on graph is not normal...
  - 如果graph fitted to training data都做不到...那还做啥...

  - *直接对image_node_feature 做分类试试, 看下loss能否正常降... -> temp1* >> 好像也不会正常掉... 
  - 直接对node_feature 做 CE_loss也是不行.... 可能是节点和标签分配出问题了... >> infer 写的有错.. 重跑.. >>似乎没问题, OK了..
  - 跳过 image_node assignment, 直接对image_embed做cls_loss >> OK了..

-- 推断用graph

- graph + residual + cls_head
- graph + residual + cls_head + person -> cos

- graph + residual + language_head
- 用到全部节点

- 改graph edge connection


>>>>>>>
-- 只用图像node 试下
-- 泄露标签 (make image node only connected with the GT text node...)
>>>>>>>

0907 找到问题了...纯粹因为 training 建图时, 数据乱了, 但是label没有相应打乱..

GCN是可以的....
但是换成HEAT, 即使只用image_node, 还是崩塌...


-- lequan code + loss


- how to further improve? 核心思想: 先把性能堆上去, 再整合设计....
    - 推断用graph
    - graph_cls也使用language-based
    - improve graph construction (edge across descriptions by world-net)
    - how to dig out the graph relation across different subjects (graphs)?



设计思路
- 尽量主体全部放在graph层面, 非graph部分只用来前期预热
- graph内的交互
- graph之间的对比学习
- test_time graph prompt tuning

## 0908
- 主体思路已经确定
- TODO   
  - [X] 为每个slide生成高质量的文本描述
  - [TBD] supervise only node features or the fused node features from all modalities
  - 



- Grading II/III/IV 这种描述能被CLIP识别出来吗? 感觉大概率得用到外部医学知识吧?