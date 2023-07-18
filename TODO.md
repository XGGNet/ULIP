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
       1. *try 把adapter相关的东西remove, 跑一遍只tune gene branch的, 看看visual performance 会不会变?
3. 为啥 acc 和 AP&AUC的排序完全不同?
   1. metric code写错了?
   2. 输入metric calculator的pred/label写错了?
   





# 眼底基因+ULIP

## 网络

## 数据


# 随记
- Grading II/III/IV 这种描述能被CLIP识别出来吗? 感觉大概率得用到外部医学知识吧?