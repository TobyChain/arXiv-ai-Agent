ArXiv AI Daily Report - Wed, 17 Dec 2025



# ArXiv AI Daily Report

Wed, 17 Dec 2025



## [Understanding and Improving Hyperbolic Deep Reinforcement Learning](http://arxiv.org/abs/2512.14202v1)

Timo Klein, Thomas Lang, Andrii Shkabrii, Alexander Sturm, Kevin Sidak, Lukas Miklautz, Claudia Plant, Yllka Velaj, Sebastian Tschiatschek

[PDF](https://arxiv.org/pdf/2512.14202v1)
[Abstract](http://arxiv.org/abs/2512.14202v1)

### 中文摘要

强化学习（RL）智能体的表现在很大程度上依赖于其基础特征表示的质量。双曲空间的特征表示在此方面非常适用，因为它们自然能捕捉复杂RL环境中常见的层级结构和关系结构。然而，由于强化学习的非平稳性，利用这些空间通常面临优化难题。在本文中，我们分析了影响双曲深度RL智能体训练成败的关键因素。通过对双曲几何中的Poincaré球模型和双曲双曲面模型中核心操作的梯度进行分析，我们发现大范数嵌入会导致基于梯度的训练不稳定，进而引发近端策略优化（PPO）中的信任区违规。在此基础上，我们提出了Hyper++，一种新的双曲PPO智能体，包含三个组成部分：（一）通过分类值损失代替回归实现稳定的评论家训练；（二）采用特征正则化，保证范数有界，同时避免裁剪带来的维数灾难；（三）使用一种更易于优化的双曲网络层表达方式。实验结果显示，在ProcGen环境中，Hyper++实现了稳定的学习，优于以往的双曲智能体，并将实际运行时间缩短约30%。在使用Double DQN的Atari-5任务中，Hyper++显著优于欧几里得空间和其他双曲空间基线方法。我们的代码已在https://github.com/Probabilistic-and-Interactive-ML/hyper-rl开源。

BibTeX

```
@article{2512.14202v1,
  title={Understanding and Improving Hyperbolic Deep Reinforcement Learning},
  author={Timo Klein and Thomas Lang and Andrii Shkabrii and Alexander Sturm and Kevin Sidak and Lukas Miklautz and Claudia Plant and Yllka Velaj and Sebastian Tschiatschek},
  journal={arXiv preprint arXiv:2512.14202v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14202v1}
}
```

## [OPTIMA: Optimal One-shot Pruning for LLMs via Quadratic Programming Reconstruction](http://arxiv.org/abs/2512.13886v1)

Mohammad Mozaffari, Samuel Kushnir, Maryam Mehri Dehnavi, Amir Yazdanbakhsh

[PDF](https://arxiv.org/pdf/2512.13886v1)
[Abstract](http://arxiv.org/abs/2512.13886v1)

### 中文摘要

模型后训练剪枝是一种具有潜力的解决方案，但面临着权衡：简单的启发式方法通过将权重置零速度快，但会降低模型的准确性；而基于原则的联合优化方法能够恢复准确性，但在现代规模下计算成本过高。像SparseGPT这样的单次方法提供了一种在最优性方面的实际折中，通过应用高效的近似启发式权重更新。为缩小这一差距，我们提出了OPTIMA，一种实用的单次后训练剪枝方法，能够在保持准确率的同时具有良好的可扩展性。OPTIMA将掩码选择后的层级权重重建问题转化为彼此独立的、行级别的二次规划问题（QPs），这些QPs共享一个共同的层级海森矩阵。通过求解这些QPs，可以获得在估算的海森矩阵条件下，针对每一行的全局最优重建更新。共享的海森矩阵结构使得该问题非常适合在加速器上进行批量处理。我们实现了一种适合加速器的Qp求解器，该求解器在每层累计一个海森矩阵，支持多小型Qp的并行求解，从而实现无需微调的单次后训练剪枝在大规模环境中的应用。OPTIMA可以与现有的掩码选择器协同工作，在多种大规模语言模型和不同稀疏率场景下持续提升零-shot性能，带来最高达3.97%的绝对准确率提升。在一台NVIDIA H100加速器上，OPTIMA能够在40小时内对一个8B参数的Transformer模型进行端到端的剪枝，峰值内存使用60GB。这些成果共同开启了单次后训练剪枝在准确率与效率之间的新纪元，树立了行业的最优平衡点。

BibTeX

```
@article{2512.13886v1,
  title={OPTIMA: Optimal One-shot Pruning for LLMs via Quadratic Programming Reconstruction},
  author={Mohammad Mozaffari and Samuel Kushnir and Maryam Mehri Dehnavi and Amir Yazdanbakhsh},
  journal={arXiv preprint arXiv:2512.13886v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13886v1}
}
```

## [Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-Editors](http://arxiv.org/abs/2512.13860v1)

Henger Li, Shuangjie You, Flavio Di Palo, Yiyue Qian, Ayush Jain

[PDF](https://arxiv.org/pdf/2512.13860v1)
[Abstract](http://arxiv.org/abs/2512.13860v1)

### 中文摘要

工具调用使大型语言模型（LLMs）能够通过工具调用与外部环境交互，为克服预训练局限性提供了一种切实可行的方法。然而，工具使用的效果很大程度上依赖于相关文档和知识库上下文的质量。这些材料通常是为人工用户编写，常常与LLMs解释信息的方式不一致。在工业环境中，这一问题尤为突出，成百上千的功能重叠的工具带来了可扩展性、多样性和歧义性方面的挑战。我们提出了验证引导的上下文优化（VGCO）框架，该框架利用LLMs作为编辑器，自动优化与工具相关的文档和知识库上下文。VGCO分为两个阶段：第一阶段，评估（Evaluation）收集实际故障案例，识别工具与其上下文之间的不匹配；第二阶段，优化（Optimization）通过结构感知的离线学习和上下文中的层次化编辑，进行结构化改进。我们的LLM编辑器具有三大创新点：第一，采用层次化结构，能够自然融入工具调用流程；第二，具备状态感知、动作特定性和验证引导特性，限制搜索空间，提高目标导向的效率；第三，支持成本高效的子任务专业化，无论是通过大规模编辑模型的提示工程，还是通过微调较小的编辑模型。与强调多轮推理的先前工作不同，VGCO专注于单轮大规模工具调用问题，在准确性、鲁棒性和跨LLMs的泛化能力方面实现了显著提升。

BibTeX

```
@article{2512.13860v1,
  title={Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-Editors},
  author={Henger Li and Shuangjie You and Flavio Di Palo and Yiyue Qian and Ayush Jain},
  journal={arXiv preprint arXiv:2512.13860v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13860v1}
}
```

## [Beyond Procedural Compliance: Human Oversight as a Dimension of Well-being Efficacy in AI Governance](http://arxiv.org/abs/2512.13768v1)

Yao Xie, Walter Cullen

[PDF](https://arxiv.org/pdf/2512.13768v1)
[Abstract](http://arxiv.org/abs/2512.13768v1)

### 中文摘要

主要的人工智能伦理指南和法律法规，包括欧盟人工智能法案，强调有效的人类监督，但未将其定义为一种独立且可发展的能力。本文将人类监督界定为一种福祉能力，置于新兴的“福祉效能”框架中。该概念融合了人工智能素养、伦理判别能力以及对人类需求的觉察，认识到部分需求可能彼此冲突或具有潜在危害。由于人们不可避免地会将自己的欲望、恐惧和利益投射到AI系统中，监督过程需要具备审查和在必要时限制问题性需求的能力。
作者认为，这一能力的可持续且具成本效益的发展，依赖于其在各个教育层面的融合，从职业培训到终身学习。将人类监督作为一种福祉能力的框架，为实现从高层监管目标到持续培养人类主体性与责任感提供了务实途径，这对于确保AI的安全与伦理运作至关重要。本文为未来关于福祉效能的教育实施与在多种情境中进行经验验证的研究奠定了理论基础。

BibTeX

```
@article{2512.13768v1,
  title={Beyond Procedural Compliance: Human Oversight as a Dimension of Well-being Efficacy in AI Governance},
  author={Yao Xie and Walter Cullen},
  journal={arXiv preprint arXiv:2512.13768v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13768v1}
}
```

## [Towards Deep Learning Surrogate for the Forward Problem in Electrocardiology: A Scalable Alternative to Physics-Based Models](http://arxiv.org/abs/2512.13765v1)

Shaheim Ogbomo-Harmitt, Cesare Magnetti, Chiara Spota, Jakub Grzelak, Oleg Aslanidi

[PDF](https://arxiv.org/pdf/2512.13765v1)
[Abstract](http://arxiv.org/abs/2512.13765v1)

### 中文摘要

在心电学中，正向问题旨在由心脏电活动计算体表电位，传统上采用基于物理的模型，如双随机联系方程（双区模型）或单区模型，进行求解。尽管这些方法具有较高的精度，但计算成本较高，限制了其在实时和大规模临床应用中的使用。本文提出了一种深度学习（DL）框架，作为正向求解器的高效代理。该模型采用时序依赖、基于注意力的序列到序列结构，能够从心脏电压传播图预测心电图（ECG）信号。为了同时保持时间域和频率域的准确性，模型引入了一种融合了Huber损失和频谱熵项的混合损失函数。在包含健康、纤维化和缝隙连接重塑等多种模拟条件的二维心肌模型上进行验证后，模型展现出极高的预测精度（平均$R^2 = 0.99 \pm 0.01$）。消融实验确认了卷积编码器、时间感知注意力机制以及频谱熵损失的贡献。这些结果表明，深度学习作为一种具有良好扩展性且成本效益高的替代方案，有望在临床和数字孪生等应用中发挥重要作用。

BibTeX

```
@article{2512.13765v1,
  title={Towards Deep Learning Surrogate for the Forward Problem in Electrocardiology: A Scalable Alternative to Physics-Based Models},
  author={Shaheim Ogbomo-Harmitt and Cesare Magnetti and Chiara Spota and Jakub Grzelak and Oleg Aslanidi},
  journal={arXiv preprint arXiv:2512.13765v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13765v1}
}
```

## [STAR: STacked AutoRegressive Scheme for Unified Multimodal Learning](http://arxiv.org/abs/2512.13752v1)

Jie Qin, Jiancheng Huang, Limeng Qiao, Lin Ma

[PDF](https://arxiv.org/pdf/2512.13752v1)
[Abstract](http://arxiv.org/abs/2512.13752v1)

### 中文摘要

多模态大型语言模型（MLLMs）在推动通用人工智能的研究中发挥着关键作用。然而，由于优化冲突和性能权衡，实现多模态理解与生成的统一目标仍面临诸多挑战。为有效提升生成能力的同时保持现有的理解水平，我们提出了STAR：一种用于任务渐进式统一多模态学习的堆叠自回归（STacked AutoRegressive）方案。该方法将多模态学习划分为理解、生成和编辑多个阶段。通过固定基础自回归（AR）模型的参数，并逐步堆叠同构的AR模块，避免了任务之间的相互干扰，同时扩展了模型的能力。同时，我们引入了高容量的矢量量子化（VQ）技术，以增强图像表达的细粒度，并采用隐式推理机制以提升复杂条件下的生成质量。实验证明，STAR在GenEval（0.91）、DPG-Bench（87.44）和ImgEdit（4.34）等任务上均达到了最先进的性能，验证了其在实现统一多模态学习方面的有效性。

BibTeX

```
@article{2512.13752v1,
  title={STAR: STacked AutoRegressive Scheme for Unified Multimodal Learning},
  author={Jie Qin and Jiancheng Huang and Limeng Qiao and Lin Ma},
  journal={arXiv preprint arXiv:2512.13752v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13752v1}
}
```

## [Comparative Evaluation of Embedding Representations for Financial News Sentiment Analysis](http://arxiv.org/abs/2512.13749v1)

Joyjit Roy, Samaresh Kumar Singh

[PDF](https://arxiv.org/pdf/2512.13749v1)
[Abstract](http://arxiv.org/abs/2512.13749v1)

### 中文摘要

金融情感分析有助于增强市场洞察力，但在资源有限环境下，基于标准自然语言处理的方法在处理小型数据集时面临重大挑战。本研究对在资源受限条件下用于金融新闻情感分类的嵌入方法进行了对比评估。通过手工标注的新闻头条，评估了Word2Vec、GloVe和句子变换器等表征方法结合梯度提升模型的性能。实验结果显示，模型在验证集上的表现显著优于测试集，存在较大的性能差距，甚至比简单基线模型表现更差，尽管验证指标依然较好。分析表明，预训练嵌入在数据不足达到某一临界阈值后，收益逐渐递减；同时，小规模的验证集容易导致模型在选择过程中的过拟合。论文以每周情感汇总和市场监测工作流程中的情感概要为实际应用场景，进行了示范。研究结果提供了实证证据，表明单纯依赖嵌入质量无法解决情感分类中的根本性数据稀缺问题。对于资源有限的实践者而言，当标注样本稀缺时，需考虑诸如少样本学习、数据增强或基于词典的混合方法等替代策略，以弥补数据不足带来的挑战。

BibTeX

```
@article{2512.13749v1,
  title={Comparative Evaluation of Embedding Representations for Financial News Sentiment Analysis},
  author={Joyjit Roy and Samaresh Kumar Singh},
  journal={arXiv preprint arXiv:2512.13749v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13749v1}
}
```

## [Why Text Prevails: Vision May Undermine Multimodal Medical Decision Making](http://arxiv.org/abs/2512.13747v1)

Siyuan Dai, Lunxiao Li, Kun Zhao, Eardi Lila, Paul K. Crane, Heng Huang, Dongkuan Xu, Haoteng Tang, Liang Zhan

[PDF](https://arxiv.org/pdf/2512.13747v1)
[Abstract](http://arxiv.org/abs/2512.13747v1)

### 中文摘要

随着大规模语言模型（LLMs）的快速发展，先进的多模态大模型（MLLMs）在视觉-语言任务中展现出了令人印象深刻的零样本能力。然而，在生物医学领域，即使是最先进的多模态大模型也难以胜任基本的医疗决策（MDM）任务。我们采用两个具有挑战性的数据集对这一限制进行探讨：一是三阶段阿尔茨海默病（AD）分类（正常、轻度认知障碍、痴呆症），其中类别差异在视觉上非常细微；二是MIMIC-CXR胸部影像分类，涵盖14种非互斥条件。实证研究表明，仅依赖文本的推理表现始终优于纯视觉或视觉-文本结合的方式，而多模态输入的表现往往不如仅使用文本的效果。为了解决这一问题，我们探索了三种策略：一是在上下文中使用带有推理标签的示例进行学习；二是通过影像描述生成后进行文本推理；三是对视觉特征提取模型进行少量样本微调，结合分类监督。这些结果揭示了当前多模态大模型缺乏稳固的视觉理解能力，并指明了提升医疗多模态决策能力的潜在方向。

BibTeX

```
@article{2512.13747v1,
  title={Why Text Prevails: Vision May Undermine Multimodal Medical Decision Making},
  author={Siyuan Dai and Lunxiao Li and Kun Zhao and Eardi Lila and Paul K. Crane and Heng Huang and Dongkuan Xu and Haoteng Tang and Liang Zhan},
  journal={arXiv preprint arXiv:2512.13747v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13747v1}
}
```

## [The Laminar Flow Hypothesis: Detecting Jailbreaks via Semantic Turbulence in Large Language Models](http://arxiv.org/abs/2512.13741v1)

Md. Hasib Ur Rahman

[PDF](https://arxiv.org/pdf/2512.13741v1)
[Abstract](http://arxiv.org/abs/2512.13741v1)

### 中文摘要

随着大型语言模型（LLMs）的普及，确保其免受对抗“越狱”攻击的挑战也日益加剧。现有的防御策略多依赖于计算成本高昂的外部分类器或脆弱的词汇过滤器，忽视了模型推理过程的内在动态。在本研究中，我们提出了层流假说（Laminar Flow Hypothesis），该假说认为，良性输入会引发LLM在高维潜在空间中平滑、渐进的变化，而对抗性提示则会引发混沌、高方差的轨迹——即所谓的“语义湍流”（Semantic Turbulence），这是由安全对齐与指令执行目标之间的内在冲突所导致的。这一现象通过一种新颖的零-shot指标得以形式化，即层级余弦速度的方差。在多个不同的小型语言模型中进行的实验显示出该指标具有显著的诊断能力。经RLHF校准的Qwen2-1.5B模型在受到攻击时，湍流值呈现出75.4%的显著增加（p值小于0.001），验证了内部冲突的假说。相反，Gemma-2B模型的湍流降低了22.0%，表现出一种低熵的“反射式”拒绝机制。这些发现表明，语义湍流不仅可以作为一种轻量级、实时的越狱检测工具，而且可用作一种非侵入性的方法，用以分类黑箱模型的底层安全架构。

BibTeX

```
@article{2512.13741v1,
  title={The Laminar Flow Hypothesis: Detecting Jailbreaks via Semantic Turbulence in Large Language Models},
  author={Md. Hasib Ur Rahman},
  journal={arXiv preprint arXiv:2512.13741v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13741v1}
}
```

## [TF-MCL: Time-frequency Fusion and Multi-domain Cross-Loss for Self-supervised Depression Detection](http://arxiv.org/abs/2512.13736v1)

Li-Xuan Zhao, Chen-Yang Xu, Wen-Qiang Li, Bo Wang, Rong-Xing Wei, Qing-Hao Menga

[PDF](https://arxiv.org/pdf/2512.13736v1)
[Abstract](http://arxiv.org/abs/2512.13736v1)

### 中文摘要

近年来，基于脑电图（EEG）信号的重度抑郁症（MDD）检测的监督方法显著增加，但标注MDD样本仍然具有挑战性。作为一种自监督学习方法，对比学习能够弥补监督方法在MDD检测中对标签的过度依赖的不足。然而，现有的对比学习方法尚未针对EEG信号的时频分布进行专门设计，其在获取低语义数据表征方面的能力仍然不足以满足MDD检测的需求。为了解决这一问题，本文提出了一种融合时频特征和多域交叉损失的（TF-MCL）模型，用于MDD检测。该模型通过融合映射头（FMH）生成时频混合表征，有效地将时频域信息映射到融合域，从而增强模型的时频信息整合能力。此外，通过优化多域交叉损失函数，重建了时频域与融合域中表征的分布，提升了模型获得融合表征的能力。我们在公开的MODMA和PRED+CT数据集上对模型进行评估，结果显示模型在准确率方面取得了显著提升，分别比现有的最先进方法（SOTA）高出5.87%和9.96%。

BibTeX

```
@article{2512.13736v1,
  title={TF-MCL: Time-frequency Fusion and Multi-domain Cross-Loss for Self-supervised Depression Detection},
  author={Li-Xuan Zhao and Chen-Yang Xu and Wen-Qiang Li and Bo Wang and Rong-Xing Wei and Qing-Hao Menga},
  journal={arXiv preprint arXiv:2512.13736v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13736v1}
}
```

## [DARTs: A Dual-Path Robust Framework for Anomaly Detection in High-Dimensional Multivariate Time Series](http://arxiv.org/abs/2512.13735v1)

Xuechun Liu, Heli Sun, Xuecheng Wu, Ruichen Cao, Yunyun Shi, Dingkang Yang, Haoran Li

[PDF](https://arxiv.org/pdf/2512.13735v1)
[Abstract](http://arxiv.org/abs/2512.13735v1)

### 中文摘要

多变量时间序列异常检测（MTSAD）旨在准确识别和定位大型工业控制系统中复杂的异常模式。现有方法在低维场景下对不同模式的识别效果显著，但在高维带噪声时间序列中学习表征时，往往难以稳健捕捉长距离的时空依赖关系。为了解决这些限制，本文提出了一种鲁棒的长短时双路径框架——DARTs，结合窗口感知的时空软融合机制，主要由三个互补组件组成。具体而言，在短期路径中，我们引入多视角稀疏图学习器和扩散多关系图单元，协作动态捕捉高噪声时间序列中的层级判别性短期时空模式；而在长远路径中，我们设计了多尺度时空图构建器，用于在高维表示空间中建模显著的长远动态。最后，提出一种窗口感知的时空软融合机制，有效过滤残余噪声的同时，无缝整合异常模式。大量的定性与定量实验结果在主流数据集上验证了本方法的优越性与鲁棒性。同时，我们还进行了系列消融实验，探讨了各个关键设计因素对性能的影响。我们的代码和模型将于近期开源。

BibTeX

```
@article{2512.13735v1,
  title={DARTs: A Dual-Path Robust Framework for Anomaly Detection in High-Dimensional Multivariate Time Series},
  author={Xuechun Liu and Heli Sun and Xuecheng Wu and Ruichen Cao and Yunyun Shi and Dingkang Yang and Haoran Li},
  journal={arXiv preprint arXiv:2512.13735v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13735v1}
}
```

## [Low-Rank Compression of Language Models via Differentiable Rank Selection](http://arxiv.org/abs/2512.13733v1)

Sidhant Sundrani, Francesco Tudisco, Pasquale Minervini

[PDF](https://arxiv.org/pdf/2512.13733v1)
[Abstract](http://arxiv.org/abs/2512.13733v1)

### 中文摘要

基于低秩分解的压缩大型语言模型的方法取得了显著进展，特别是引入了激活和损失感知的奇异值分解（SVD），这提升了分解秩与下游任务性能之间的权衡。然而，仍然存在一个持续的挑战——如何选择每一层的最优秩，以同时优化压缩率和下游任务的准确性。目前的方法要么依赖启发式策略，由于搜索空间有限，可能导致次优结果，要么是基于梯度的方法，虽然可行，但在未进行后续微调的情况下，其性能不及启发式方法。为了解决这些问题，我们提出了“学习低秩压缩（LLRC）”的方法，这是一种基于梯度的策略，能够在无需微调的情况下，直接学习用于选择奇异值的掩码权重。通过使用校准数据集，我们仅训练掩码的权重，使其逐步减少所选的奇异值数量，同时最小化中间激活与原始模型之间的差异。我们的方案在多个压缩比、面向常识推理和开放域问答任务上，优于同类无需后续微调的排名选择方法。例如，在Llama-2-13B模型压缩率为20%的情况下，LLRC在MMLU、BoolQ和OpenbookQA任务中的表现，分别比竞争对手的敏感性截断秩搜索（STRS）提升了12%、3.5%和4.4%。与其他压缩技术相比，我们的方法在不同数据集和压缩率下，始终优于无需微调的SVD-LLM和LLM-Pruner变体。此外，我们的无微调方案在性能上也与微调版的LLM-Pruner相当。

BibTeX

```
@article{2512.13733v1,
  title={Low-Rank Compression of Language Models via Differentiable Rank Selection},
  author={Sidhant Sundrani and Francesco Tudisco and Pasquale Minervini},
  journal={arXiv preprint arXiv:2512.13733v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13733v1}
}
```

## [PIS: A Generalized Physical Inversion Solver for Arbitrary Sparse Observations via Set-Conditioned Diffusion](http://arxiv.org/abs/2512.13732v1)

Weijie Yang, Xun Zhang

[PDF](https://arxiv.org/pdf/2512.13732v1)
[Abstract](http://arxiv.org/abs/2512.13732v1)

### 中文摘要

从有限的间接观测中估算由偏微分方程（PDE）约束的物理参数本质上是一个病态逆问题，尤其在观测数据稀疏、不规则且受限于实际传感器布置时尤为突出。这一挑战在流体力学、地震反演和结构健康监测等领域普遍存在。现有的深度学习和算子学习模型在这些条件下容易失效：固定网格假设不成立，重建效果迅速恶化，反演变得不可靠，鲁棒性不足且无法进行不确定性量化（UQ）。
我们提出了物理反演求解器（PIS），一种基于集合条件的扩散框架，能够从真正任意的观测集进行反演。PIS采用基于集合变换器（Set Transformer）的编码器，能够处理任意数量和几何形状的观测数据，并引入余弦退火稀疏度课程，以实现卓越的鲁棒性。附带的信息论分析揭示了在极端稀疏条件下反演的极限，阐明了观测熵在不同物理系统中的变化方式。
在三类具有挑战性的偏微分方程反问题——达西流、波场反演（亥姆霍兹方程）以及结构健康监测（胡克定律）——中对PIS进行评估。在所有任务和稀疏度水平下（包括观测率仅为$0.29\%$的极端情况），现有的算子学习基线模型均无法有效重建有意义的物理场，常出现发散或完全崩溃。相比之下，PIS保持了稳定性和高精度，反演误差降低了12.28%到88.73%，并能可靠地产生经校准的后验样本。所生成的样本能够真实反映数据稀缺性和内在的物理不确定性。
这些结果表明，PIS是一种强大、通用且对稀疏极端情况具有超强韧性的物理反演解决方案，特别适用于任意且严重采样不足的观测条件。

BibTeX

```
@article{2512.13732v1,
  title={PIS: A Generalized Physical Inversion Solver for Arbitrary Sparse Observations via Set-Conditioned Diffusion},
  author={Weijie Yang and Xun Zhang},
  journal={arXiv preprint arXiv:2512.13732v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13732v1}
}
```

## [Made-in China, Thinking in America:U.S. Values Persist in Chinese LLMs](http://arxiv.org/abs/2512.13723v1)

David Haslett, Linus Ta-Lun Huang, Leila Khalatbari, Janet Hui-wen Hsiao, Antoni B. Chan

[PDF](https://arxiv.org/pdf/2512.13723v1)
[Abstract](http://arxiv.org/abs/2512.13723v1)

### 中文摘要

随着大型语言模型在获取信息和辅助决策方面的日益普及，它们正逐渐成为美国和中国等全球行为体之间软实力竞争的工具。目前，语言模型似乎更倾向于体现西方国家的价值观，但这一伦理偏见的证据主要来自美国公司开发的模型。最新一代的尖端模型中也包含一些由中国开发的模型，因此我们开展了首次大规模研究，探讨由中美两国制造的模型在价值观上与中国和美国人民的契合程度。我们让十个中国模型和十个美国模型回答《道德基础问卷 2.0》和《世界价值观调查》，并将其回答结果与成千上万的中国和美国民众的反应进行了比较。结果显示，所有模型的回答都更倾向于美国人民的观点，而非中国人民的观点。即使用中文提示或赋予模型中国人设，这种偏向美国价值观的倾向也仅得到略微缓解。这些发现对未来格外重要，因为大型语言模型将生成大量人们所消费的内容，并在地缘政治中影响规范性价值观。

BibTeX

```
@article{2512.13723v1,
  title={Made-in China, Thinking in America:U.S. Values Persist in Chinese LLMs},
  author={David Haslett and Linus Ta-Lun Huang and Leila Khalatbari and Janet Hui-wen Hsiao and Antoni B. Chan},
  journal={arXiv preprint arXiv:2512.13723v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13723v1}
}
```

## [Scaling and Transferability of Annealing Strategies in Large Language Model Training](http://arxiv.org/abs/2512.13705v1)

Siqi Wang, Zhengyu Chen, Teng Xiao, Zheqi Lv, Jinluan Yang, Xunliang Cai, Jingang Wang, Xiaomeng Li

[PDF](https://arxiv.org/pdf/2512.13705v1)
[Abstract](http://arxiv.org/abs/2512.13705v1)

### 中文摘要

学习率调度在大规模语言模型的训练中至关重要，但如何在不同模型架构下实现最优的退火策略仍具挑战性。本研究探讨了大规模语言模型训练中退火动态的转移性，并提出了一种经过改进的通用预测框架，用于在Warmup-Steady-Decay（WSD）调度器下优化退火策略。该框架集成了训练步骤、最大学习率和退火行为，提升了学习率调度的优化效率。我们的工作为在无需进行繁琐超参数搜索的情况下选择最优退火策略提供了实用指导，证明较小模型可以作为优化大模型训练动态的可靠代理。通过大量基于密集模型和专家混合模型（MoE）的实验验证，发现最优的退火比例呈现出一致的规律，并且可以在不同的训练配置之间实现迁移。

BibTeX

```
@article{2512.13705v1,
  title={Scaling and Transferability of Annealing Strategies in Large Language Model Training},
  author={Siqi Wang and Zhengyu Chen and Teng Xiao and Zheqi Lv and Jinluan Yang and Xunliang Cai and Jingang Wang and Xiaomeng Li},
  journal={arXiv preprint arXiv:2512.13705v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13705v1}
}
```

## [Writing in Symbiosis: Mapping Human Creative Agency in the AI Era](http://arxiv.org/abs/2512.13697v1)

Vivan Doshi, Mengyuan Li

[PDF](https://arxiv.org/pdf/2512.13697v1)
[Abstract](http://arxiv.org/abs/2512.13697v1)

### 中文摘要

大型语言模型（LLMs）的广泛应用引发了关于当我们与具有说服力和创造力的机器建立日益共生关系时，作为人的意义的关键问题。本文探讨了创造性写作中人类与人工智能共同演化的模式，研究人类创作能力与主观能动性如何在机器能力的发展中进行调整。我们挑战了主流的风格同质化观点，通过分析纵向写作数据中的多样化模式。在横跨LLM出现前后的大规模语料库中，观察到一种“双轨演进”的模式：围绕AI相关主题的趋同，同时伴随着结构化的风格差异。我们的分析揭示了三种新兴的适应模式：一是作者表现出向AI风格靠拢的趋势；二是表现出与AI风格减少相似性的趋势；三是维持风格稳定但涉及AI相关主题。此“创意原型图”阐明了作者身份如何与AI共同演变，为关于人机协作、检测难题以及创造性多样性保护等议题提供了新的视角。

BibTeX

```
@article{2512.13697v1,
  title={Writing in Symbiosis: Mapping Human Creative Agency in the AI Era},
  author={Vivan Doshi and Mengyuan Li},
  journal={arXiv preprint arXiv:2512.13697v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13697v1}
}
```

## [Dual Attention Guided Defense Against Malicious Edits](http://arxiv.org/abs/2512.14333v1)

Jie Zhang, Shuai Dong, Shiguang Shan, Xilin Chen

[PDF](https://arxiv.org/pdf/2512.14333v1)
[Abstract](http://arxiv.org/abs/2512.14333v1)

### 中文摘要

近年来，文本到图像扩散模型的进展极大推动了基于文本提示的图像编辑技术，但同时也带来了严重的伦理挑战，尤其是在潜在被用于制造误导性或有害内容的风险方面。当前的防护方法试图通过嵌入难以察觉的扰动来缓解此类风险，但在应对恶意篡改时效果有限。为此，本文提出了一种双重注意力引导噪声扰动（DANP）免疫方法，，该方法在模型的推理过程中加入微不可察觉的扰动，以干扰其语义理解和生成机制。DANP在多个时间步内操作，调控交叉注意力图与噪声预测过程，利用动态阈值生成掩码，从而识别文本相关与无关区域。在此基础上，减少相关区域的注意力输出，同时增强无关区域的注意力，从而误导模型将编辑效果偏向错误区域，同时保持目标区域的完整性。此外，该方法还最大化注入噪声与模型预测噪声之间的差异，以进一步扰乱生成过程。通过同时攻击注意力机制和噪声预测机制，DANP表现出强大的抗恶意编辑能力，大量实验结果验证了其在性能上的优越性，达到了先进水平。

BibTeX

```
@article{2512.14333v1,
  title={Dual Attention Guided Defense Against Malicious Edits},
  author={Jie Zhang and Shuai Dong and Shiguang Shan and Xilin Chen},
  journal={arXiv preprint arXiv:2512.14333v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14333v1}
}
```

## [Estimating problem difficulty without ground truth using Large Language Model comparisons](http://arxiv.org/abs/2512.14220v1)

Marthe Ballon, Andres Algaba, Brecht Verbeken, Vincent Ginis

[PDF](https://arxiv.org/pdf/2512.14220v1)
[Abstract](http://arxiv.org/abs/2512.14220v1)

### 中文摘要

近年来，针对大型语言模型（LLMs）微调的研究取得了显著进展，极大地提升了它们在既定基准测试中的性能，强调了对更具挑战性的合成数据的需求。在这一数据生成流程中，一个关键环节是估算问题难度的方法。目前的方法，例如人工调节或基于性能的评分，难以推广到分布外问题，即目前人类和LLMs都无法解决的问题，因为它们缺乏可扩展性，耗时且依赖于正确答案的存在。因此，我们提出了一种新的问题难度估算方法——LLM 比较，该方法解决了上述局限性。具体而言，LLM 通过成对进行难度比较，然后根据结果计算Bradley-Terry评分。为了验证该方法的有效性，我们首先提出了一个概念框架，将现有方法置于“构造”、“规模”和“依赖”三个正交维度上，分析了不同量度在分布外问题评分中的位置。LLM 比较在所有理想区域内自然体现：它是连续且动态的、模型无关的，且不依赖于 ground truth 信息。第二个验证方面，我们展示了LLM 比较在与人类标注高度一致方面表现优异（皮尔逊相关系数$ r \geq 0.80 $，样本数 $ n=1876 $）。第三，我们证明了LLM 比较对幻觉（hallucinations）具有较强的鲁棒性，注入10%的噪声时，皮尔逊相关下降少于6%。我们的研究是实现替代耗时人工标注和合成数据生成的重大一步，并将在课程设计、模型评估以及人工智能辅助的研究思路生成中发挥重要作用。

BibTeX

```
@article{2512.14220v1,
  title={Estimating problem difficulty without ground truth using Large Language Model comparisons},
  author={Marthe Ballon and Andres Algaba and Brecht Verbeken and Vincent Ginis},
  journal={arXiv preprint arXiv:2512.14220v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14220v1}
}
```

## [ProtoFlow: Interpretable and Robust Surgical Workflow Modeling with Learned Dynamic Scene Graph Prototypes](http://arxiv.org/abs/2512.14092v1)

Felix Holm, Ghazal Ghazaei, Nassir Navab

[PDF](https://arxiv.org/pdf/2512.14092v1)
[Abstract](http://arxiv.org/abs/2512.14092v1)

### 中文摘要

目的：精确的手术识别对于推动人工智能辅助手术具有关键作用，但当前的研究受到高昂的标注成本、数据稀缺以及缺乏可解释模型的限制。虽然场景图提供了手术事件的结构化抽象，其潜力尚未充分挖掘。本工作提出了ProtoFlow，一种创新框架，能学习动态场景图原型，以一种可解释且鲁棒的方式建模复杂的手术流程。
方法：ProtoFlow采用基于图神经网络（GNN）的编码器-解码器结构，将自监督预训练用于丰富的表示学习，并融合基于原型的微调阶段，发现并优化能够捕捉手术交互中的重复、临床意义重大模式的核心原型，从而为流程分析提供一个可解释的基础。
结果：我们在细粒度的CAT-SG数据集上对所提方法进行了评估。ProtoFlow不仅在整体准确率上优于常规GNN基线，还在少样本、少量数据场景中表现出极强的鲁棒性，即使仅用一段手术视频进行训练，也能保持优异性能。定性分析显示，学到的原型成功识别了不同的手术子技术，并能提供清晰、可解释的洞察，有助于理解流程偏差和罕见的并发症。
结论：将鲁棒的表示学习与固有的可解释性相结合，ProtoFlow在开发更透明、可信且数据高效的人工智能系统方面迈出了重要步伐，有助于其在手术培训、实时决策支持和流程优化中的临床应用潜力加速实现。

BibTeX

```
@article{2512.14092v1,
  title={ProtoFlow: Interpretable and Robust Surgical Workflow Modeling with Learned Dynamic Scene Graph Prototypes},
  author={Felix Holm and Ghazal Ghazaei and Nassir Navab},
  journal={arXiv preprint arXiv:2512.14092v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14092v1}
}
```

## [Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed](http://arxiv.org/abs/2512.14067v1)

Yonggan Fu, Lexington Whalen, Zhifan Ye, Xin Dong, Shizhe Diao, Jingyu Liu, Chengyue Wu, Hao Zhang, Enze Xie, Song Han, Maksim Khadkevich, Jan Kautz, Yingyan Celine Lin, Pavlo Molchanov

[PDF](https://arxiv.org/pdf/2512.14067v1)
[Abstract](http://arxiv.org/abs/2512.14067v1)

### 中文摘要

扩散语言模型（dLMs）已成为一种具有前景的范式，它实现了并行的非自回归生成，但在从零开始训练时，其学习效率仍落后于自回归（AR）语言模型。为此，我们研究了AR到dLM的转换方法，旨在将预训练的AR模型转化为高效的dLM，兼顾速度和任务准确性。我们通过识别现有AR到dLM方法在注意力模式和目标上的局限性，提出了更为有效的转换原则与方法。
具体而言，首先，我们系统比较了不同的注意力模式，发现保持预训练AR模型的权重分布对于有效转换至关重要。因此，提出了一种采用块式注意力模式的连续预训练方案，在保持跨块因果性的同时，实现每个块内的双向建模。研究表明，这种方法比完全双向建模更能保持预训练AR模型的权重分布，同时具有KV缓存的已知优势，在准确率和效率方面实现了双赢。
其次，为了缓解训练过程中掩码标记分布（均匀分布与高度左到右分布）与测试阶段的差异，我们提出了一种位置相关的标记掩码策略，在训练时对后续标记赋予更高的掩码概率，以更好地模拟测试时的行为。
利用这一框架，我们进行了大量关于dLM注意力模式、训练动态及其他设计选择的深入研究，提供了可操作的见解，推动了可扩展的AR到dLM转换技术。这些研究成果催生了Efficient-DLM系列模型，其性能优于当前最先进的AR模型和dLMs。例如，我们的Efficient-DLM 8B在准确率方面比Dream 7B高出5.4%/2.7%，在吞吐量方面分别提高4.5倍和2.7倍。

BibTeX

```
@article{2512.14067v1,
  title={Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed},
  author={Yonggan Fu and Lexington Whalen and Zhifan Ye and Xin Dong and Shizhe Diao and Jingyu Liu and Chengyue Wu and Hao Zhang and Enze Xie and Song Han and Maksim Khadkevich and Jan Kautz and Yingyan Celine Lin and Pavlo Molchanov},
  journal={arXiv preprint arXiv:2512.14067v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14067v1}
}
```

## [FacEDiT: Unified Talking Face Editing and Generation via Facial Motion Infilling](http://arxiv.org/abs/2512.14056v1)

Kim Sung-Bin, Joohyun Chang, David Harwath, Tae-Hyun Oh

[PDF](https://arxiv.org/pdf/2512.14056v1)
[Abstract](http://arxiv.org/abs/2512.14056v1)

### 中文摘要

我们通常将面部对话编辑和面部生成视为两个独立的问题。在本研究中，我们提出将两者统归为一个统一的框架，即基于语音条件的面部动作填充任务。我们将面部动作填充作为一种自监督预任务，同时也是动态对话面部合成的统一表达方式。为了实现此想法，我们提出了FacEDiT，一种通过流匹配训练的语音条件扩散变换模型。受到掩码自动编码器的启发，FacEDiT学会根据周围的动作和语音信息来合成被掩码的面部动作。这一框架支持局部生成与编辑操作，如替换、插入和删除，同时确保与未编辑区域的平滑过渡。此外，模型引入偏置注意力机制和时间平滑约束，以增强边界的连续性和口型同步效果。为应对缺乏标准化面部编辑评测基准的问题，我们推出了FacEDiTBench，这是首个面部对话编辑数据集，涵盖多样的编辑类型与长度，并配备了新的评估指标。大量实验结果验证了将面部对话编辑与生成作为语音条件动作填充的子任务是有效的；FacEDiT能够生成精准、语音同步的面部编辑，保持强烈的身份特征和流畅的视觉效果，同时在面部对话生成任务中表现出良好的泛化能力。

BibTeX

```
@article{2512.14056v1,
  title={FacEDiT: Unified Talking Face Editing and Generation via Facial Motion Infilling},
  author={Kim Sung-Bin and Joohyun Chang and David Harwath and Tae-Hyun Oh},
  journal={arXiv preprint arXiv:2512.14056v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14056v1}
}
```

## [Memo2496: Expert-Annotated Dataset and Dual-View Adaptive Framework for Music Emotion Recognition](http://arxiv.org/abs/2512.13998v1)

Qilin Li, C. L. Philip Chen, TongZhang

[PDF](https://arxiv.org/pdf/2512.13998v1)
[Abstract](http://arxiv.org/abs/2512.13998v1)

### 中文摘要

音乐情感识别（MER）研究面临高质量带标注数据集有限以及跨曲目特征漂移难题的挑战。本工作提出两项主要贡献以应对此类问题。第一，Memo2496是一个大规模的数据集，包含2496首纯音乐曲目，并配备由30名认证音乐专家进行标注的连续的价值得分和激动得分。标注质量通过与极端情感典范的校准以及采用欧几里得距离衡量的0.25一致性阈值进行保证。此外，本文还提出双视角自适应音乐情感识别模型（DAMER）。该模型整合了三大协同模块：双流注意力融合（DSAF）通过跨注意机制实现梅尔频谱图与耳蜗图之间的逐词交互；逐步置信标签（PCL）利用基于课程的温度调度和Jensen-Shannon发散的一致性量化，生成可靠的伪标签；风格锚定记忆学习（SAML）则通过保持对比记忆队列，有效缓解跨曲目特征漂移。大量在Memo2496、1000songs和PMEmo数据集上的实验结果表明，DAMER达到先进水平，在激动得分维度的准确率分别提升3.43%、2.25%以及0.17%。消融实验与可视化分析验证了各模块的贡献。相关数据集与源代码已公开。

BibTeX

```
@article{2512.13998v1,
  title={Memo2496: Expert-Annotated Dataset and Dual-View Adaptive Framework for Music Emotion Recognition},
  author={Qilin Li and C. L. Philip Chen and TongZhang},
  journal={arXiv preprint arXiv:2512.13998v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13998v1}
}
```

## [A Multicenter Benchmark of Multiple Instance Learning Models for Lymphoma Subtyping from HE-stained Whole Slide Images](http://arxiv.org/abs/2512.14640v1)

Rao Muhammad Umer, Daniel Sens, Jonathan Noll, Christian Matek, Lukas Wolfseher, Rainer Spang, Ralf Huss, Johannes Raffler, Sarah Reinke, Wolfram Klapper, Katja Steiger, Kristina Schwamborn, Carsten Marr

[PDF](https://arxiv.org/pdf/2512.14640v1)
[Abstract](http://arxiv.org/abs/2512.14640v1)

### 中文摘要

及时且准确的淋巴瘤诊断对于指导癌症治疗至关重要。标准诊断流程通常结合苏木精-伊红（HE）染色的全切片图像与免疫组化、流式细胞术及分子遗传检测，以确定淋巴瘤亚型，但此过程依赖昂贵的设备、熟练的专业人员，且易造成治疗延误。深度学习方法有望通过提取常规HE染色切片中的诊断信息来辅助病理学家，但目前关于多中心数据的淋巴瘤亚型分类的全面基准尚缺乏。为此，我们首次构建了涵盖四种常见淋巴瘤亚型及健康对照组织的多中心淋巴瘤基准数据集。我们系统评估了五种公开可用的病理基础模型（H-optimus-1、H0-mini、Virchow2、UNI2、Titan），结合基于注意力机制的多实例学习聚合方法（AB-MIL）和基于变换器的多实例学习方法（TransMIL），在包括10倍、20倍和40倍在内的三种不同倍镜下的性能表现。在分布内测试集中，所有模型在多分类任务中均实现了超过80%的平衡精度，基础模型表现相似，聚合方法之间的结果也基本一致。倍率分析显示，40倍放大倍数已足够，不同于更高倍率或跨倍率汇聚没有明显性能提升。然而，在分布外测试集中，模型性能显著下降，约为60%，凸显了较强的泛化能力挑战。为了推动该领域发展，需要涵盖更多稀有亚型的更大规模多中心研究。本研究还提供了一个自动化的基准评估流程，以促进未来相关工作的发展。

BibTeX

```
@article{2512.14640v1,
  title={A Multicenter Benchmark of Multiple Instance Learning Models for Lymphoma Subtyping from HE-stained Whole Slide Images},
  author={Rao Muhammad Umer and Daniel Sens and Jonathan Noll and Christian Matek and Lukas Wolfseher and Rainer Spang and Ralf Huss and Johannes Raffler and Sarah Reinke and Wolfram Klapper and Katja Steiger and Kristina Schwamborn and Carsten Marr},
  journal={arXiv preprint arXiv:2512.14640v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14640v1}
}
```

## [SASQ: Static Activation Scaling for Quantization-Aware Training in Large Language Models](http://arxiv.org/abs/2512.14481v1)

Shizhuo Mao, Song Chen, Yi Kang

[PDF](https://arxiv.org/pdf/2512.14481v1)
[Abstract](http://arxiv.org/abs/2512.14481v1)

### 中文摘要

大型语言模型（LLMs）在自然语言任务中表现出色，但由于模型规模不断扩大，超出了GPU内存的提升，部署面临诸多挑战。模型量化通过降低权重和激活值的精度，有效缓解了这一问题，但现有方案存在根本性的权衡：动态量化会带来较高的计算开销，并在边缘设备部署时遇到困难，而静态量化则会牺牲部分精度。现有的感知量化训练（QAT）方法还进一步增加了训练成本。为此，我们提出了SASQ——一个专为激活量化因子设计的轻量级QAT框架。SASQ仅优化量化因子（且不改变预训练的权重），实现高精度的静态推理，同时保持部署效率。该方法通过自适应截断部分异常值，减轻了量化的难度，同时保留了激活的分布特征。实验显示，SASQ不仅超越了现有的最先进量化方案，还优于对应的FP16模型。在LLaMA2-7B模型上，SASQ在WikiText2数据集上的困惑度比QuaRot低5.2%，比FP16模型低4.7%。

BibTeX

```
@article{2512.14481v1,
  title={SASQ: Static Activation Scaling for Quantization-Aware Training in Large Language Models},
  author={Shizhuo Mao and Song Chen and Yi Kang},
  journal={arXiv preprint arXiv:2512.14481v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14481v1}
}
```

## [DISCODE: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning](http://arxiv.org/abs/2512.14420v1)

Nakamasa Inoue, Kanoko Goto, Masanari Oi, Martyna Gruszka, Mahiro Ukai, Takumi Hirose, Yusuke Sekikawa

[PDF](https://arxiv.org/pdf/2512.14420v1)
[Abstract](http://arxiv.org/abs/2512.14420v1)

### 中文摘要

大型视觉-语言模型（LVLMs）在多模态任务中展现出令人印象深刻的性能。然而，利用LVLMs进行稳健的图像描述评估仍然具有挑战性，尤其是在存在领域偏移的场景中。为解决这一问题，我们提出了分布感知评分解码器（DISCODE），这是一种无需微调的创新方法，能够生成与人类评判更加贴合、在多领域中表现更为稳健的评估分数。DISCODE的核心思想在于其测试时自适应的评估策略，引入了自适应测试时（ATT）损失，利用高斯先验分布来提升评估分数估计的鲁棒性。我们通过分析推导出一种高效的解析解，在测试时对该损失进行最小化。此外，我们还提出了多领域描述评估（MCEval）基准，这是一个涵盖六个不同领域的新型图像描述评估基准，旨在检验评估指标的稳健性。在实验中，我们证明了DISCODE作为无参考评估指标，在MCEval及四个具有代表性的现有基准中均达到了最新的性能水平。

BibTeX

```
@article{2512.14420v1,
  title={DISCODE: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning},
  author={Nakamasa Inoue and Kanoko Goto and Masanari Oi and Martyna Gruszka and Mahiro Ukai and Takumi Hirose and Yusuke Sekikawa},
  journal={arXiv preprint arXiv:2512.14420v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14420v1}
}
```

## [Spoken DialogSum: An Emotion-Rich Conversational Dataset for Spoken Dialogue Summarization](http://arxiv.org/abs/2512.14687v1)

Yen-Ju Lu, Kunxiao Gao, Mingrui Liang, Helin Wang, Thomas Thebaud, Laureano Moro-Velazquez, Najim Dehak, Jesus Villalba

[PDF](https://arxiv.org/pdf/2512.14687v1)
[Abstract](http://arxiv.org/abs/2512.14687v1)

### 中文摘要

近期的音频语言模型已经能够跟踪长时间对话。然而，关于情感感知或口语对话总结的研究受限于缺乏将语音、摘要与情感和非语言线索关联起来的数据。我们引入了Spoken DialogSum，这是首个将原始对话音频与事实摘要、富含情感的摘要以及说话者年龄、性别和情感等逐句标签进行对齐的语料库。该数据集的构建分为两个阶段：首先，利用大规模语言模型（LLM）重写DialogSum脚本，加入Switchboard风格的填充语和响应线，然后对每句话进行情感、音调和语速等标签标注；第二，使用富有表现力的文本到语音（TTS）引擎，根据带标签的脚本合成语音，并与非语言线索标签对齐。Spoken DialogSum包含13,460个情感多样的对话，每个对话配有事实性摘要和情感聚焦的摘要。该数据集可在https://fatfat-emosum.github.io/EmoDialog-Sum-Audio-Samples/线上获取。实验结果表明，与级联的自动语音识别（ASR）-LLM系统相比，使用音频-LLM的方法使情感摘要的ROUGE-L提升了28%，验证了端到端语音建模的价值。

BibTeX

```
@article{2512.14687v1,
  title={Spoken DialogSum: An Emotion-Rich Conversational Dataset for Spoken Dialogue Summarization},
  author={Yen-Ju Lu and Kunxiao Gao and Mingrui Liang and Helin Wang and Thomas Thebaud and Laureano Moro-Velazquez and Najim Dehak and Jesus Villalba},
  journal={arXiv preprint arXiv:2512.14687v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14687v1}
}
```

## [VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](http://arxiv.org/abs/2512.14677v1)

Sicheng Xu, Guojun Chen, Jiaolong Yang, Yizhong Zhang, Yu Deng, Steve Lin, Baining Guo

[PDF](https://arxiv.org/pdf/2512.14677v1)
[Abstract](http://arxiv.org/abs/2512.14677v1)

### 中文摘要

我们提出了VASA-3D，一种基于音频驱动的单次拍摄三维头部虚拟形象生成器。本研究解决了两个主要挑战：捕捉真实人脸中细微的表情细节，以及从单一肖像图像中重建复杂的三维头部虚拟形象。为了准确模拟表情细节，VASA-3D借鉴了VASA-1中的运动潜在空间，该方法在二维口型动画中表现出极高的真实感和生动性。本研究的关键在于将运动潜在空间转化为三维表示，通过设计一个以运动潜在为条件的三维头部模型实现这一目标。将该模型个性化适配到单一图像的过程，则通过一种优化框架完成，该框架利用由输入图像合成的参考头部的多个视频帧进行训练。该优化过程采用多种对伪影和有限姿态覆盖具有鲁棒性的损失函数。实验证明，VASA-3D能够生成逼真的三维口型动画，超越先前方法，同时支持以最高75帧每秒实时生成512×512像素的自由视角视频，推动更具沉浸感的生动三维虚拟形象的在线生成与互动。

BibTeX

```
@article{2512.14677v1,
  title={VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image},
  author={Sicheng Xu and Guojun Chen and Jiaolong Yang and Yizhong Zhang and Yu Deng and Steve Lin and Baining Guo},
  journal={arXiv preprint arXiv:2512.14677v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14677v1}
}
```

## [EDGC: Entropy-driven Dynamic Gradient Compression for Efficient LLM Training](http://arxiv.org/abs/2511.10333v1)

Qingao Yi, Jiaang Duan, Hanwen Hu, Qin Hua, Haiyan Zhao, Shiyou Qian, Dingyu Yang, Jian Cao, Jinghua Tang, Yinghao Yu, Chenzhi Liao, Kangjin Wang, Liping Zhang

[PDF](https://arxiv.org/pdf/2511.10333v1)
[Abstract](http://arxiv.org/abs/2511.10333v1)

### 中文摘要

训练大规模语言模型（LLMs）在计算资源和内存容量方面面临巨大挑战。虽然分布式训练技术有助于缓解这些问题，但仍存在较大的通信开销。现有方法主要依赖于静态梯度压缩以提高通信效率，但忽视了训练过程中梯度的动态变化，导致性能下降。如何在不牺牲模型性能的前提下，通过压缩实现训练加速，仍是一大难题。为此，本文提出了一种基于信息熵的动态梯度压缩框架，称为EDGC。其核心思想是根据梯度熵的演变趋势动态调整压缩比，兼顾压缩效率与误差控制。EDGC由三部分组成：第一，采用下采样方法高效估算梯度熵，降低计算开销；第二，建立压缩比与梯度熵之间的理论模型，为压缩决策提供依据；第三，利用基于窗口的调整机制，在不同训练阶段动态调节压缩比，以提升通信效率并保证模型性能。我们在搭建的32卡NVIDIA V100集群和64卡NVIDIA H100集群上，分别用于训练GPT-2.5B和GPT-2.1B，实验结果表明，EDGC在保持模型精度的同时，显著降低了通信延迟和训练时间，减少最多46.45%的通信时延和16.13%的训练时间。

BibTeX

```
@article{2511.10333v1,
  title={EDGC: Entropy-driven Dynamic Gradient Compression for Efficient LLM Training},
  author={Qingao Yi and Jiaang Duan and Hanwen Hu and Qin Hua and Haiyan Zhao and Shiyou Qian and Dingyu Yang and Jian Cao and Jinghua Tang and Yinghao Yu and Chenzhi Liao and Kangjin Wang and Liping Zhang},
  journal={arXiv preprint arXiv:2511.10333v1},
  year={2025},
  url={http://arxiv.org/abs/2511.10333v1}
}
```

## [EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery](http://arxiv.org/abs/2512.13857v1)

Kamer Ali Yuksel

[PDF](https://arxiv.org/pdf/2512.13857v1)
[Abstract](http://arxiv.org/abs/2512.13857v1)

### 中文摘要

大型语言模型（LLMs）在程序演化和多智能体系统中的应用日益广泛，但现有大多数方法依赖于基于覆盖的突变操作，仅维护一个候选方案。这类方法会丢弃有价值的变体，容易受到破坏性编辑的影响，并且探索的搜索空间脆弱，容易导致结构性失配。本文提出了EvoLattice（进化格），一种将全部候选程序或智能体行为表示为单一有向无环图（DAG）的框架。在该框架中，每个节点存储多个持久备用方案，图中每一条有效路径定义一个不同的可执行候选，从而在不复制结构的情况下实现了大规模组合搜索空间。EvoLattice支持对每个备用方案进行细粒度的评价，通过对所有出现该备用方案的路径进行评分，生成反映局部设计选择对全局性能影响的统计信息。这些统计提供了密集的数据驱动反馈信号，用于引导LLMs进行突变、重组和裁剪，同时保留成功的组件。其结构正确性由一种确定性自我修复机制保障，无需依赖LLM，可独立强制保持无环性和依赖关系的一致性。EvoLattice自然拓展到智能体演化，将备用方案解释为提示片段或子智能体行为。在程序合成（代理和优化器元学习）任务中，EvoLattice展现出比现有LLM引导方法更稳定的演化过程、更强的表达能力以及更优的改进轨迹。这一动力学特征类似于质量-多样性优化，隐含地由EvoLattice的多备选表示所产生，而非通过外部明确存档实现。

BibTeX

```
@article{2512.13857v1,
  title={EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery},
  author={Kamer Ali Yuksel},
  journal={arXiv preprint arXiv:2512.13857v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13857v1}
}
```

## [Semantic Grounding Index: Geometric Bounds on Context Engagement in RAG Systems](http://arxiv.org/abs/2512.13771v1)

Javier Marín

[PDF](https://arxiv.org/pdf/2512.13771v1)
[Abstract](http://arxiv.org/abs/2512.13771v1)

### 中文摘要

当基于检索增强生成（RAG）的系统出现幻觉现象时，这在嵌入空间中留下了何种几何痕迹？我们提出了“语义基础指数（SGI）”，其定义为在单位超球面 \(\mathbb{S}^{d-1}\) 上，响应到问题的角距离与响应到上下文的角距离之比。我们的核心发现是“语义惰性”：幻觉生成的响应通常在角度上接近问题，而非偏离到检索到的上下文。在HaluEval（样本数为5,000）数据集上，我们在五种嵌入模型中观察到显著的效果大小（Cohen's \(d\) 从0.92到1.28不等），平均模型间相关性（\(r\)）为0.85。关键的是，通过球面三角不等式，我们推导出SGI的判别能力应随问题与上下文之间的角度差 \(θ(q,c)\) 增大而增强——这是一个理论预测，并已通过实验证明：效果大小从在低角度差（\(d=0.61\)）时的表现逐渐增长到高角度差（\(d=1.27\)）时，且AUC值由0.72提升至0.83。子组分析显示，SGI在长响应（\(d=2.05\)）和短问题（\(d=1.22\)）中表现优异，同时在不同上下文长度中依然表现稳健。校准分析结果显示，平均绝对误差（ECE）为0.10，表明SGI评分不仅可用于排序，还可作为概率估计。在TruthfulQA数据集上的负面测试结果（AUC=0.478）表明，角几何指标主要衡量话题相关性，而非事实准确性。SGI为在生产环境中部署的RAG系统中识别需要验证的响应，提供了一个计算高效、理论基础充分的分析工具。

BibTeX

```
@article{2512.13771v1,
  title={Semantic Grounding Index: Geometric Bounds on Context Engagement in RAG Systems},
  author={Javier Marín},
  journal={arXiv preprint arXiv:2512.13771v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13771v1}
}
```

## [Mathematics and Coding are Universal AI Benchmarks](http://arxiv.org/abs/2512.13764v1)

Przemyslaw Chojecki

[PDF](https://arxiv.org/pdf/2512.13764v1)
[Abstract](http://arxiv.org/abs/2512.13764v1)

### 中文摘要

我们研究了数学与编码在人工智能代理心理测量电池模空间中的特殊作用。在此前工作提出的AAI框架和GVU动力学基础上，我们定义了数学纤维，并证明结合形式证明核心（如Lean、Coq）时，GVU在此纤维上的流动通过类观测验证实现了谱稳定的自我提升机制。我们的主要技术结果是一个密度定理：在代理输出的均匀紧性和Lipschitz连续的AAI泛函条件下，由数学定理证明和编码任务生成的电池子空间在评估度量下在电池模空间中是稠密的。从这个意义上讲，编码具有普适性，而单纯的数学则不具备；其优势表现为谱稳定性而非表达能力。这被我们解读为数学与编码提供了“普遍坐标”以进行评估的证据，而形式数学则是高级AI代理实现递归自我提升的自然启动领域。

BibTeX

```
@article{2512.13764v1,
  title={Mathematics and Coding are Universal AI Benchmarks},
  author={Przemyslaw Chojecki},
  journal={arXiv preprint arXiv:2512.13764v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13764v1}
}
```

## [DL$^3$M: A Vision-to-Language Framework for Expert-Level Medical Reasoning through Deep Learning and Large Language Models](http://arxiv.org/abs/2512.13742v1)

Md. Najib Hasan, Imran Ahmad, Sourav Basak Shuvo, Md. Mahadi Hasan Ankon, Sunanda Das, Nazmul Siddique, Hui Wang

[PDF](https://arxiv.org/pdf/2512.13742v1)
[Abstract](http://arxiv.org/abs/2512.13742v1)

### 中文摘要

医学影像分类器在检测胃肠疾病方面表现优异，但它们通常无法解释其决策过程。虽然大型语言模型（LLMs）能够生成临床文本，但在视觉推理方面存在困难，容易产生不稳定或不准确的解释，导致模型的图像理解与临床医生预期的推理方式之间存在鸿沟。为此，我们提出了一种将影像分类与结构化临床推理相结合的框架。其中，设计了一种新型混合模型MobileCoAtNet，专用于内镜影像，在八个胃部相关类别上实现了高识别精度。模型输出的结果被用于指导多种LLMs的推理过程。为了评估这些推理的可靠性，我们构建了两个经过专家验证的基准，包括疾病原因、症状、治疗、生活方式及随访护理等方面，共涵盖医学多维信息。我们对32个LLMs进行了评测，将它们在这些黄金标准上的表现进行比较。结果显示，强大的分类性能有助于提升LLMs解释的质量，但没有一款模型达到人类的稳定水平。即使是表现最好的LLMs，其推理结果也会因提示的变化而发生改变。我们的研究表明，将深度学习与LLMs结合可以生成有价值的临床叙述，但目前的LLMs尚不足以支持高风险医疗决策。本框架有助于明确这些模型的局限性，为构建更安全的推理系统提供了可能。本文所使用的全部源码和数据集已在https://github.com/souravbasakshuvo/DL3M开源。

BibTeX

```
@article{2512.13742v1,
  title={DL$^3$M: A Vision-to-Language Framework for Expert-Level Medical Reasoning through Deep Learning and Large Language Models},
  author={Md. Najib Hasan and Imran Ahmad and Sourav Basak Shuvo and Md. Mahadi Hasan Ankon and Sunanda Das and Nazmul Siddique and Hui Wang},
  journal={arXiv preprint arXiv:2512.13742v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13742v1}
}
```

## [Exploring the Modular Integration of "AI + Architecture" Pedagogy in Undergraduate Design Education: A Case Study of Architectural Design III/IV Courses at Zhejiang University](http://arxiv.org/abs/2512.13730v1)

Wang Jiaqi, Lan Yi, Chen Xiang

[PDF](https://arxiv.org/pdf/2512.13730v1)
[Abstract](http://arxiv.org/abs/2512.13730v1)

### 中文摘要

本研究通过在浙江大学2024-25学年三年级本科生设计工作室的教学实验，探讨了人工智能在建筑教育中的融合方式。采用双模块框架（20小时的AI培训+嵌入伦理讨论），课程引入了深度学习模型、大型语言模型（LLMs）、AIGC、LoRA 和 ComfyUI，同时保持了原有的课程结构，并由专门的技术指导人员提供支持。研究结果表明，分阶段的指导、技术与伦理的平衡以及机构的支持，均具有显著的有效性。该模型不仅提升了学生的数字技能和战略认知能力，还有效地应对了AI伦理问题，为将技术与批判性学习相结合的设计教育提供了一种可复制的实践路径。

BibTeX

```
@article{2512.13730v1,
  title={Exploring the Modular Integration of "AI + Architecture" Pedagogy in Undergraduate Design Education: A Case Study of Architectural Design III/IV Courses at Zhejiang University},
  author={Wang Jiaqi and Lan Yi and Chen Xiang},
  journal={arXiv preprint arXiv:2512.13730v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13730v1}
}
```

## [CurvaDion: Curvature-Adaptive Distributed Orthonormalization](http://arxiv.org/abs/2512.13728v1)

Bhavesh Kumar, Roger Jin, Jeffrey Quesnelle

[PDF](https://arxiv.org/pdf/2512.13728v1)
[Abstract](http://arxiv.org/abs/2512.13728v1)

### 中文摘要

随着语言模型规模扩大至万亿参数级别，跨多台GPU的分布式训练变得尤为重要，但在高带宽、低延迟网络中的梯度同步仍然是一个关键瓶颈。尽管像Dion这样的新方法通过低秩更新在每个训练步骤中减少通信量，但它们无论优化景观如何，仍在每步都进行同步。我们观察到，训练过程中同步的需求会有显著变化：在平坦区域，工作节点自然计算出相似的梯度，频繁同步变得多余；而在高曲率区域，为防止训练发散则需要协同。为此，我们提出了CurvaDion方法，该方法利用相对最大动量变化（RMMC）来检测需要同步的高曲率区域。RMMC利用在优化过程中已计算的动量信息，作为方向曲率的计算近似，只需在每层增加$\mathcal{O}(d)$的计算操作。我们在理论上建立了RMMC与损失函数曲率之间的联系，并证明CurvaDion在实现与基线模型相当的收敛速度的同时，能够减少99%的通信量。

BibTeX

```
@article{2512.13728v1,
  title={CurvaDion: Curvature-Adaptive Distributed Orthonormalization},
  author={Bhavesh Kumar and Roger Jin and Jeffrey Quesnelle},
  journal={arXiv preprint arXiv:2512.13728v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13728v1}
}
```

## [Time-Constrained Recommendations: Reinforcement Learning Strategies for E-Commerce](http://arxiv.org/abs/2512.13726v1)

Sayak Chakrabarty, Souradip Pal

[PDF](https://arxiv.org/pdf/2512.13726v1)
[Abstract](http://arxiv.org/abs/2512.13726v1)

### 中文摘要

与传统推荐任务不同，有限的用户时间预算引入了一项关键的资源约束，要求推荐系统在物品相关性和评估成本之间进行权衡。例如，在移动购物界面中，用户通过滚动与推荐内容进行交互，每次滚动都会触发一列被称为“阵列”的商品。用户在评估商品特征后决定是否点击，此过程会产生评估成本——即花费的时间。高度相关的商品尽管价值较高，但其较高的评估成本可能超出用户的时间预算，从而影响用户的参与度。在本文的立场性论文中，我们的目标是评估强化学习算法，这些算法能够同时学习用户偏好和时间预算的模式，从而在资源受限的情况下，生成具有更高参与潜力的推荐。我们的实验利用阿里巴巴的个性化重排序数据集，探索了在电子商务场景中利用强化学习进行阵列优化的方法。我们的贡献包括：（i）将时间约束的阵列推荐统一建模为具有预算感知效用的马尔可夫决策过程（MDP）；（ii）提出一套模拟框架，用于研究策略在重排序数据上的表现；以及（③）提供实证依据，表明在严格时间预算下，基于策略的控制（无论是策略内还是策略外学习）都能优于传统的上下文多臂老虎机方法，提升推荐性能。

BibTeX

```
@article{2512.13726v1,
  title={Time-Constrained Recommendations: Reinforcement Learning Strategies for E-Commerce},
  author={Sayak Chakrabarty and Souradip Pal},
  journal={arXiv preprint arXiv:2512.13726v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13726v1}
}
```

## [Compressed Causal Reasoning: Quantization and GraphRAG Effects on Interventional and Counterfactual Accuracy](http://arxiv.org/abs/2512.13725v1)

Steve Nwaiwu, Nipat Jongsawat, Anucha Tungkasthan

[PDF](https://arxiv.org/pdf/2512.13725v1)
[Abstract](http://arxiv.org/abs/2512.13725v1)

### 中文摘要

在高风险场景中，具有关联推理、干预推理和反事实推理的因果推理在大型语言模型（LLMs）中的应用至关重要。随着模型部署逐步向边缘计算和资源受限环境转移，INT8 和 NF4 等量化模型逐渐成为行业标准。然而，尺度减少对形式因果推理的影响尚缺乏系统性研究。据我们所知，这是首次对 Pearls 因果阶梯的三个层面进行量化效应的全面评估。我们利用包含3000个样本的分层型 CLadder 基准测试，发现 Llama 3 8B 在量化后各阶层的推理准确率基本保持稳定，NF4 展示出不到1%的整体性能下降。在第二阶层的干预查询中，模型对精度下降最为敏感；而在第三阶层的反事实推理中，表现相对稳定，但在 Collider 偏差和后门调整等不同查询类型上仍存在异质性弱点。在 CRASS 基准测试中，不同精度之间几乎表现一致，表明现有的常识反事实数据集缺乏捕捉量化引起的推理漂移所需的结构敏感性。进一步使用真实因果图进行图结构增强生成（GRA）实验，NF4的干预推理准确率提升了1.7个百分点，部分抵消了模型压缩带来的性能损失。综合结果显示，因果推理对四比特量化表现出出乎意料的鲁棒性，图结构增强技术可以选择性地强化干预推理能力，而当前的反事实基准数据集未能充分反映更深层次的因果脆弱性。本研究初步绘制了经过压缩的因果推理的实证地图，并为在资源有限环境中部署高效且结构支持的因果AI系统提供了实际指导。

BibTeX

```
@article{2512.13725v1,
  title={Compressed Causal Reasoning: Quantization and GraphRAG Effects on Interventional and Counterfactual Accuracy},
  author={Steve Nwaiwu and Nipat Jongsawat and Anucha Tungkasthan},
  journal={arXiv preprint arXiv:2512.13725v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13725v1}
}
```

## [AI-Powered Annotation Pipelines for Stabilizing Large Language Models: A Human-AI Synergy Approach](http://arxiv.org/abs/2512.13714v1)

Gangesh Pathak, Prasanna Kumar

[PDF](https://arxiv.org/pdf/2512.13714v1)
[Abstract](http://arxiv.org/abs/2512.13714v1)

### 中文摘要

大型语言模型（LLM）在高度受规管的行业中面临失稳问题，表现为推理不一致、幻觉现象以及性能波动，尤其在工作流程中尤为明显。这些可靠性问题限制了LLM在需要精确事实和行为一致性的领域中的安全应用（Aiyappa 等，2023）。目前的稳定性提升方法，如基于人类反馈的强化学习（RLHF）和监督式微调，虽然带来了可量化的改善，但成本高昂，依赖于大量人工标注，难以实现可持续的规模化（Dong 等，2023；Retzlaff 等，2024）。本文提出了一种基于人工智能的标注流程，系统地识别、标注并修正LLM输出中的不稳定模式。我们的人机协作方法结合了自动弱监督模型和基于置信度的标注技术，并辅以人工验证，以确保反馈信息的可靠性和道德正当性（Cabitza 等，2023；Jiang 等，2023）。在我们的框架中引入了语义一致性、事实准确性和逻辑连贯性等不稳定性特定的标注类别，从而通过反馈环不断校准模型，提升其稳健性（Honovich 等，2021；Nan 等，2021）。

BibTeX

```
@article{2512.13714v1,
  title={AI-Powered Annotation Pipelines for Stabilizing Large Language Models: A Human-AI Synergy Approach},
  author={Gangesh Pathak and Prasanna Kumar},
  journal={arXiv preprint arXiv:2512.13714v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13714v1}
}
```

## [Enhancing Transparency and Traceability in Healthcare AI: The AI Product Passport](http://arxiv.org/abs/2512.13702v1)

A. Anil Sinaci, Senan Postaci, Dogukan Cavdaroglu, Machteld J. Boonstra, Okan Mercan, Kerem Yilmaz, Gokce B. Laleci Erturkmen, Folkert W. Asselbergs, Karim Lekadir

[PDF](https://arxiv.org/pdf/2512.13702v1)
[Abstract](http://arxiv.org/abs/2512.13702v1)

### 中文摘要

目标：开发人工智能产品护照（AI Product Passport），这是一个基于标准的框架，旨在通过生命周期管理的文档记录提升医疗AI的透明度、可追溯性和合规性。方法与材料：AI产品护照的开发是在AI4HF项目中进行的，重点关注心力衰竭AI工具。我们分析了欧盟AI法案、FDA指南等监管框架及现有标准，设计了一个关系型数据模型，涵盖AI生命周期各阶段的元数据：研究定义、数据集准备、模型生成与评估、部署与监控，以及护照生成。将MLOps/ModelOps的概念融入其中，以增强操作的相关性。通过与AI4HF联盟成员的合作及在里斯本举办的包含21位多元利益相关者的工作坊收集反馈，并通过Mentimeter投票加以评估。采用Python库实现了一个开源平台，用于自动追踪源数据和模型的溯源信息。结果：以现有标准和方法为基础，设计了具有明确生命周期管理和角色基础访问控制的AI产品护照。其实现为基于网页的平台，配备支持可审计文档的关系型数据模型，能够生成机器可读和人体可读的报告，且可根据不同利益相关者需求进行定制。该护照符合FUTURE-AI原则（公平性、普遍性、可追溯性、可用性、鲁棒性、可解释性），确保公平、可追溯性和实用性。导出的护照详细记载模型的用途、数据追溯信息、性能表现及部署环境。GitHub托管的后端和前端代码库提升了平台的可访问性。讨论与结论：人工智能产品护照解决了医疗AI中存在的透明度不足问题，满足监管和伦理要求。其开源特性及与标准的对接，促进了信任建立和适应性增强。未来的优化方向包括引入FAIR数据原则和FHIR接口，以提升互操作性，推动负责任的AI部署。

BibTeX

```
@article{2512.13702v1,
  title={Enhancing Transparency and Traceability in Healthcare AI: The AI Product Passport},
  author={A. Anil Sinaci and Senan Postaci and Dogukan Cavdaroglu and Machteld J. Boonstra and Okan Mercan and Kerem Yilmaz and Gokce B. Laleci Erturkmen and Folkert W. Asselbergs and Karim Lekadir},
  journal={arXiv preprint arXiv:2512.13702v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13702v1}
}
```

## [Arithmetic-Intensity-Aware Quantization](http://arxiv.org/abs/2512.14090v1)

Taig Singh, Shreshth Rajan, Nikhil Iyer

[PDF](https://arxiv.org/pdf/2512.14090v1)
[Abstract](http://arxiv.org/abs/2512.14090v1)

### 中文摘要

随着现代神经网络日益受到内存限制，推理吞吐量已不再主要受计算能力制约，而是受到DRAM带宽的限制。本文提出了基于算术强度感知的量化（AIQ）方法，这是一种混合精度量化框架，能够为每一层选择不同的比特宽度，以最大化算术强度（AI）同时最小化精度损失。AIQ是一种训练后量化技术，通过搜索不同的分层量化方案，最小化AI与精度之间的加权损失。在ResNet-20/CIFAR-10设置中，AIQ比FP32基线提高了大约50%的算术强度，同时保持测试准确率在约1个百分点以内，且优于全局一致性量化方案。在受到内存限制的MobileNetV2架构中，AIQ配置的吞吐率比FP32基线提升了1.66倍，同时保持测试准确率在1个百分点以内。研究还发现，AIQ自然倾向于对较大层采用更激进的量化方式。

BibTeX

```
@article{2512.14090v1,
  title={Arithmetic-Intensity-Aware Quantization},
  author={Taig Singh and Shreshth Rajan and Nikhil Iyer},
  journal={arXiv preprint arXiv:2512.14090v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14090v1}
}
```

## [SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations](http://arxiv.org/abs/2512.14080v1)

Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao

[PDF](https://arxiv.org/pdf/2512.14080v1)
[Abstract](http://arxiv.org/abs/2512.14080v1)

### 中文摘要

混合专家（MoE）模型已成为在不显著增加计算成本的情况下扩展语言模型的事实标准架构。近期的MoE模型展现出向高专家粒度（较小的专家中间维度）和更高稀疏性（激活专家数量保持不变但总专家数增加）的明显趋势，这有助于提升模型每单位浮点运算的性能表现。然而，细粒度的MoE因更高的输入/输出（IO）成本导致激活内存占用增加、硬件效率降低，而更稀疏的MoE则在Grouped GEMM核中的填充操作带来计算浪费的问题。为此，本文提出一种节省内存的算法，用于高效地进行MoE的前向和反向传播计算，最大限度地减少反向传播时的激活缓存需求。同时，我们设计了一种GPU核，能够在计算过程中叠加IO操作，提高所有MoE架构的效率。此外，我们提出一种新颖的“令牌舍入”方法，最小化由于Grouped GEMM核中的填充操作带来的计算浪费。实验结果表明，我们的SonicMoE在Hopper GPU上比ScatterMoE的BF16 MoE核减少了45%的激活内存，并实现了1.86倍的计算吞吐量提升。在一个细粒度的7B参数MoE模型训练中（采用FSDP-2和lm-engine代码库，在96个H100卡上跑），SonicMoE在64个H100上达到了213亿令牌/天的训练速度，接近于ScatterMoE的225亿令牌/天表现。在高稀疏性设置下，我们的基于图块的令牌舍入算法在核执行时间上比传统的Top-$K$路由方案额外提升1.16倍速度，同时保持了相似的下游任务性能。我们开源了所有核心代码，以加快MoE模型的训练效率。

BibTeX

```
@article{2512.14080v1,
  title={SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations},
  author={Wentao Guo and Mayank Mishra and Xinle Cheng and Ion Stoica and Tri Dao},
  journal={arXiv preprint arXiv:2512.14080v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14080v1}
}
```

## [Real-time prediction of workplane illuminance distribution for daylight-linked controls using non-intrusive multimodal deep learning](http://arxiv.org/abs/2512.14058v1)

Zulin Zhuang, Yu Bian

[PDF](https://arxiv.org/pdf/2512.14058v1)
[Abstract](http://arxiv.org/abs/2512.14058v1)

### 中文摘要

日光联动控制（DLC）在建筑节能方面具有巨大潜力，尤其是在充足的自然采光条件下，以及能够实时准确预测室内工作面照度的情况下。现有大部分关于室内日光预测的研究均基于静态场景进行开发和验证。本文提出了一种多模态深度学习框架，利用具有时空特征的非侵入式图像，实时预测室内工作面照度分布。该方法仅从侧光窗区域提取图像特征，而非室内像素，从而确保其在动态占用的室内空间中依然适用。研究在中国广州的一间试验房间进行实地测试，收集了17,344个样本用于模型训练与验证。结果显示，该模型在同分布测试集上达到了R²值大于0.98、均方根误差（RMSE）小于0.14；在未见过的日子测试集上，R²值超过0.82、RMSE小于0.17，验证了模型的高准确性和良好的时序泛化能力。

BibTeX

```
@article{2512.14058v1,
  title={Real-time prediction of workplane illuminance distribution for daylight-linked controls using non-intrusive multimodal deep learning},
  author={Zulin Zhuang and Yu Bian},
  journal={arXiv preprint arXiv:2512.14058v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14058v1}
}
```

## [Evaluating Frontier LLMs on PhD-Level Mathematical Reasoning: A Benchmark on a Textbook in Theoretical Computer Science about Randomized Algorithms](http://arxiv.org/abs/2512.13978v1)

Yang Cao, Yubin Chen, Xuyang Guo, Zhao Song, Song Yue, Jiahao Zhang, Jiale Zhao

[PDF](https://arxiv.org/pdf/2512.13978v1)
[Abstract](http://arxiv.org/abs/2512.13978v1)

### 中文摘要

近年来，大型语言模型（LLMs）的快速发展在自动化数学推理与科学发现方面取得了重要突破。 Georgiev、Gómez-Serrano、Tao 和 Wagner [GGSTW+25] 展示了人工智能系统能够探索新的构造方法并改进现有界限，体现出LLMs在推动数学发现方面日益增长的潜力。同样，Bubeck等人 [BCE+25] 也证明了GPT-5在科学工作流程中的积极作用，从提出假设到生成证明和分析都能发挥重要作用。尽管取得了这些进展，但对这些模型在经典的、研究生水平的数学理论上的严格评估依然必要，以全面理解它们的基础推理能力。在本文中，我们设计了一个涵盖四个前沿模型——GPT-5-Thinking、Gemini-3-Pro、Claude-Sonnet-4.5-Thinking 和 Grok-4——的综合基准测试，基于Motwani和Raghavan的经典教材《随机算法》 [MR95]。
我们要求每个模型针对书中的多个引理和练习，生成正式的LaTeX证明。结果显示，顶尖模型（Gemini 和 Claude）达到了较高的准确率（约66%），展现出对概率方法和形式逻辑的较为牢固的掌握，而其他模型在一致性方面明显落后（约40%）。本文还对生成的证明进行了定性分析，重点比较了其简洁性、幻觉率（hallucination）以及逻辑结构的差异。这些结果表明，虽然前沿模型已达到适合研究生教学和正式数学推导的熟练水平，但在严谨数学推导的可靠性方面，仍存在显著差异。相关代码及全部LLM生成的回答已开源，并可在 https://github.com/magiclinux/math\_benchmark\_probability 公开获取。

BibTeX

```
@article{2512.13978v1,
  title={Evaluating Frontier LLMs on PhD-Level Mathematical Reasoning: A Benchmark on a Textbook in Theoretical Computer Science about Randomized Algorithms},
  author={Yang Cao and Yubin Chen and Xuyang Guo and Zhao Song and Song Yue and Jiahao Zhang and Jiale Zhao},
  journal={arXiv preprint arXiv:2512.13978v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13978v1}
}
```

## [Context Branching for LLM Conversations: A Version Control Approach to Exploratory Programming](http://arxiv.org/abs/2512.13914v1)

Bhargav Chickmagalur Nanjundappa, Spandan Maaheshwari

[PDF](https://arxiv.org/pdf/2512.13914v1)
[Abstract](http://arxiv.org/abs/2512.13914v1)

### 中文摘要

大型语言模型（LLMs）已成为软件工程工作流程的关键组成部分，但其在多轮对话中的效果显著下降。最新研究表明，当指令通过多轮传递时，模型的性能平均下降39%，原因在于模型易于提前假设并且难以进行修正（Laban 等，2025）。这种性能退化在探索性编程任务中特尤为严重，开发者需要在不作出单一决定的情况下，探索多种解决路径。当前的解决方案迫使用户陷入一种虚假的两难：要么在受污染的对话上下文中继续，导致LLM越来越困惑；要么重新开始，丧失已积累的上下文信息。
本文提出了ContextBranch，一种基于版本控制语义的对话管理系统。它提供了四个核心操作——快照（checkpoint）、分支（branch）、切换（switch）和注入（inject）——帮助用户捕获对话状态、在隔离的环境中探索替代方案，并有选择性地合并不同的见解。我们在包括30个故意引入污染的软件工程场景的受控实验中评估了该系统。结果显示，通过分支形成的对话在回答质量上明显优于线性对话，特别是在聚焦度和上下文感知方面实现了显著提升。这些优势在涉及概念跨度较大的复杂场景中尤为突出。分支策略降低了58.1%的上下文长度（从31条减至13条消息），有效剔除了无关的探索内容。我们的工作确立了对话分支作为AI辅助探索工作中的基础操作，证明了隔离探索有助于防止上下文污染，提升多路径探索的效率和效果。

BibTeX

```
@article{2512.13914v1,
  title={Context Branching for LLM Conversations: A Version Control Approach to Exploratory Programming},
  author={Bhargav Chickmagalur Nanjundappa and Spandan Maaheshwari},
  journal={arXiv preprint arXiv:2512.13914v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13914v1}
}
```

## [MuseCPBench: an Empirical Study of Music Editing Methods through Music Context Preservation](http://arxiv.org/abs/2512.14629v1)

Yash Vishe, Eric Xue, Xunyi Jiang, Zachary Novack, Junda Wu, Julian McAuley, Xin Xu

[PDF](https://arxiv.org/pdf/2512.14629v1)
[Abstract](http://arxiv.org/abs/2512.14629v1)

### 中文摘要

音乐编辑在现代音乐制作中扮演着至关重要的角色，广泛应用于电影、广播和游戏开发等领域。近年来，音乐生成模型的突破推动了多种编辑任务的实现，如音色迁移、乐器替换和风格转变。然而，现有许多研究未能充分评估其在保持音乐属性方面的能力，即我们所定义的“音乐上下文保持”(Music Context Preservation, MCP)。虽有部分工作涉及MCP，但其评估方案和指标不一致，导致对比结果缺乏可靠性和公平性。为弥补这一空白，我们提出首个MCP评估基准——MuseCPBench，涵盖四大类音乐特质，支持对五个具有代表性的音乐编辑方法进行全面比较。通过对音乐特质、方法和模型的系统分析，我们发现当前音乐编辑技术在属性保持方面存在持续的差距，并提供了富有洞见的解释。我们希望这些发现能够为开发更高效、可靠且具备强大MCP能力的音乐编辑方案提供实际指导。

BibTeX

```
@article{2512.14629v1,
  title={MuseCPBench: an Empirical Study of Music Editing Methods through Music Context Preservation},
  author={Yash Vishe and Eric Xue and Xunyi Jiang and Zachary Novack and Junda Wu and Julian McAuley and Xin Xu},
  journal={arXiv preprint arXiv:2512.14629v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14629v1}
}
```

## [Model-Based Reinforcement Learning in Discrete-Action Non-Markovian Reward Decision Processes](http://arxiv.org/abs/2512.14617v1)

Alessandro Trapasso, Luca Iocchi, Fabio Patrizi

[PDF](https://arxiv.org/pdf/2512.14617v1)
[Abstract](http://arxiv.org/abs/2512.14617v1)

### 中文摘要

许多实际的决策问题涉及的任务，其成功依赖于整个系统的历史状态，而非仅仅达到某些期望属性的状态。马克式强化学习（RL）方法并不适用于此类任务，而基于非马克式奖励决策过程（NMRDPs）的RL则使智能体能够应对时间依赖性任务。这一策略长期以来被认为在（近）最优性以及样本效率方面缺乏形式保证。我们提出了QR-MAX，这是一种用于离散NMRDP的新型模型基础算法，通过奖励机（reward machines）实现了马克式转移学习与非马克式奖励处理的分离。据我们所知，这是首个利用这种因子化结构，在离散动作空间中实现多项式样本复杂度的PAC收敛至ε-最优策略的模型基础RL算法。随后，我们将QR-MAX拓展到连续状态空间，提出了基于SimHash的Bucket-QR-MAX离散化器，它保持了相同的因子化结构，在无需手工划格或函数逼近的情况下，实现了快速而稳定的学习。我们在复杂性逐步增加的环境中与现代先进的模型基础RL方法进行了实证对比，结果显示我们的方法在样本效率方面有显著提升，同时在找到最优策略方面表现出更强的鲁棒性。

BibTeX

```
@article{2512.14617v1,
  title={Model-Based Reinforcement Learning in Discrete-Action Non-Markovian Reward Decision Processes},
  author={Alessandro Trapasso and Luca Iocchi and Fabio Patrizi},
  journal={arXiv preprint arXiv:2512.14617v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14617v1}
}
```

## [Residual GRU+MHSA: A Lightweight Hybrid Recurrent Attention Model for Cardiovascular Disease Detection](http://arxiv.org/abs/2512.14563v1)

Tejaswani Dash, Gautam Datla, Anudeep Vurity, Tazeem Ahmad, Mohd Adnan, Saima Rafi, Saisha Patro, Saina Patro

[PDF](https://arxiv.org/pdf/2512.14563v1)
[Abstract](http://arxiv.org/abs/2512.14563v1)

### 中文摘要

心血管疾病（CVD）仍然是全球主要的死亡原因，凸显了开发可靠高效的预测工具以支持早期干预的必要性。传统诊断方法依赖于手工特征工程和临床专家经验，而机器学习方法在提高模型可复现性的同时，往往难以在嘈杂且异质的临床数据中实现良好的泛化。在本研究中，我们提出了一种结合残差门控循环单元（Residual GRU）与多头自注意力机制（Multi-Head Self-Attention）的紧凑型深度学习架构，专为表格化临床记录设计。该模型集成了残差双向门控循环单元用于特征列的序列建模，包含通道重加权模块，以及利用可学习的分类标记进行全局上下文捕获的多头自注意力池化。我们在UCI心脏病数据集上通过5折分层交叉验证对模型进行评估，并将其与经典方法如逻辑回归、随机森林、支持向量机，以及现代深度学习基线模型如DeepMLP、卷积网络、循环网络和Transformer进行了对比。实验结果显示，该模型达到了0.861的准确率、0.860的宏平均F1值、0.908的ROC-AUC和0.904的PR-AUC，均优于所有对比基线。消融研究验证了残差递归、通道门控和注意力池化在模型性能中的贡献。t-SNE可视化结果表明，所学习的嵌入表示在疾病与非疾病类别之间的分离比原始特征更为清晰。这些结果表明，轻量级的结合循环与注意力机制的混合架构在提升预测准确率的同时，兼顾计算效率，为资源有限的医疗环境中的临床风险预测提供了有力的解决方案。

BibTeX

```
@article{2512.14563v1,
  title={Residual GRU+MHSA: A Lightweight Hybrid Recurrent Attention Model for Cardiovascular Disease Detection},
  author={Tejaswani Dash and Gautam Datla and Anudeep Vurity and Tazeem Ahmad and Mohd Adnan and Saima Rafi and Saisha Patro and Saina Patro},
  journal={arXiv preprint arXiv:2512.14563v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14563v1}
}
```

## [Context-Picker: Dynamic context selection using multi-stage reinforcement learning](http://arxiv.org/abs/2512.14465v1)

Siyuan Zhu, Chengdong Xu, Kaiqiang Ke, Chao Yu

[PDF](https://arxiv.org/pdf/2512.14465v1)
[Abstract](http://arxiv.org/abs/2512.14465v1)

### 中文摘要

在长篇上下文问答（LCQA）任务中，确定给定查询的最优上下文量是一个重要且具有挑战性的问题。提供的文本过少可能遗漏关键的证据信息，而过多则可能引入噪声，降低回答质量。传统方法如固定Top-$K$检索和单阶段重排序，在选择合适的段落数时面临抉择，尤其对于偏向事实性的问题，这类问题通常只需少量特定证据。为了解决这一难题，本文提出了“Context-Picker”——一种具有推理意识的框架，创新性地将上下文选择的范式由基于相似度的排序转向最小充分子集的筛选。该方法将上下文选择视为一个决策过程，通过受人类启发的两阶段强化学习策略进行优化：首先是以覆盖推理链为目标的“召回导向”阶段，其次是通过积极剪枝多余信息、提取紧凑证据集的“精确导向”阶段。为应对奖励稀疏问题，本文还提出了一种离线证据蒸馏流程，利用“留一法（LOO）”挖掘“最小充分集”，提供密集且与任务高度相关的监督信号。在五个长上下文和多跳问答基准测试中，实验结果显示“Context-Picker”显著优于强大的检索增强生成（RAG）基线模型，在保持或缩短上下文长度的同时实现了更高的答案准确率。消融实验表明， coarse-to-fine优化策略、冗余感知的奖励塑造以及基于推理的引导格式，均对性能提升起到了关键作用。

BibTeX

```
@article{2512.14465v1,
  title={Context-Picker: Dynamic context selection using multi-stage reinforcement learning},
  author={Siyuan Zhu and Chengdong Xu and Kaiqiang Ke and Chao Yu},
  journal={arXiv preprint arXiv:2512.14465v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14465v1}
}
```

## [RePo: Language Models with Context Re-Positioning](http://arxiv.org/abs/2512.14391v1)

Huayang Li, Tianyu Zhao, Richard Sproat

[PDF](https://arxiv.org/pdf/2512.14391v1)
[Abstract](http://arxiv.org/abs/2512.14391v1)

### 中文摘要

上下文学习是现代大型语言模型（LLMs）的基础；然而，现有架构通常通过分配线性或固定的位置索引，施加刚性且固定的上下文结构。基于认知负荷理论（CLT），我们认为这种缺乏信息的结构会增加外在认知负荷，消耗本应用于深度推理与注意力分配的有限工作记忆容量。为此，我们提出了RePo，一种通过上下文重新定位来减轻外在认知负荷的创新机制。与传统方法不同，RePo利用一个可微分模块$f\_φ$，为标记分配具有表达上下文依赖关系的位置信息，而不是依赖预定义的整数范围。通过在OLMo-2 1B骨架模型上持续预训练，我们证明RePo在处理噪声上下文、结构化数据及长上下文长度的任务中，显著提升了性能，同时在普通短上下文任务上保持了具有竞争力的表现。详细分析显示，RePo能够成功将更高的注意力分配给远距离但相关的信息，以密集且非线性的空间为位置赋值，并捕捉输入上下文的内在结构。我们的代码已开源，地址为 https://github.com/SakanaAI/repo。

BibTeX

```
@article{2512.14391v1,
  title={RePo: Language Models with Context Re-Positioning},
  author={Huayang Li and Tianyu Zhao and Richard Sproat},
  journal={arXiv preprint arXiv:2512.14391v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14391v1}
}
```

## [Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](http://arxiv.org/abs/2512.14361v1)

Nicholas Tagliapietra, Katharina Ensinger, Christoph Zimmer, Osman Mian

[PDF](https://arxiv.org/pdf/2512.14361v1)
[Abstract](http://arxiv.org/abs/2512.14361v1)

### 中文摘要

在实际系统中，随着时间的推移，其动力学遵循潜在的因果关系而连续演变，但这些动力学通常是未知的。现有的学习方法要么通过离散化时间——在处理不规则采样数据时表现欠佳，要么忽略系统的潜在因果关系。我们提出了一种新颖的因果发现方法——CaDyT，旨在解决这两大挑战。与采用离散时间动态贝叶斯网络建模的最先进因果发现方法不同，我们的方法基于差分因果模型，使得对连续系统的建模假设更为宽松。CaDyT利用精确的高斯过程推断来建模连续时间动力学，更贴合系统的底层动态过程。我们还提出了一种实用的实现方式，通过贪婪搜索结合算法马尔可夫条件和最小描述长度原则，识别因果结构。实验结果表明，CaDyT在规则采样和不规则采样的数据上均优于现有的先进方法，能够发现更接近真实底层动力学的因果网络。

BibTeX

```
@article{2512.14361v1,
  title={Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis},
  author={Nicholas Tagliapietra and Katharina Ensinger and Christoph Zimmer and Osman Mian},
  journal={arXiv preprint arXiv:2512.14361v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14361v1}
}
```

## [TiCard: Deployable EXPLAIN-only Residual Learning for Cardinality Estimation](http://arxiv.org/abs/2512.14358v1)

Qizhi Wang

[PDF](https://arxiv.org/pdf/2512.14358v1)
[Abstract](http://arxiv.org/abs/2512.14358v1)

### 中文摘要

基数估算是基于代价的查询优化中的关键瓶颈，然而实现有效的改进一直具有挑战性：经典的估算器往往忽略了关联性，而基于学习的估算器则通常需要针对特定工作负载的训练流程以及侵入式的集成到优化器中。本文提出了TiCard，一种低侵入性、基于校正的框架，旨在增强（而非取代）数据库的原生估算器。TiCard利用仅含 EXPLAIN 信息的特征学习乘法残差校正，并仅使用 EXPLAIN ANALYZE 提供离线标签。我们研究了两种实际应用方案：（1）使用梯度提升回归器实现子毫秒级推理；（2）采用TabPFN，一种基于上下文的表格基础模型，通过刷新小规模参考集实现无需梯度再训练的自适应调整。在TiDB上进行的TPCH和Join Order Benchmark测试中，在较低追踪比例（总共263次执行，其中157次用于训练）的设置下，TiCard显著提升了操作符级别的尾部准确性：P90 Q误差由原生的312.85降至13.69（TiCard-GBR），P99由37,974.37降至3,416.50（TiCard-TabPFN），同时仅针对连接的策略也保障了接近完美的中位性能。我们将TiCard定位为AI4DB的基础模块，强调其易于部署的特点：明确的设计范围、保守的集成策略，以及从离线校正到优化器内实时应用的集成路线。

BibTeX

```
@article{2512.14358v1,
  title={TiCard: Deployable EXPLAIN-only Residual Learning for Cardinality Estimation},
  author={Qizhi Wang},
  journal={arXiv preprint arXiv:2512.14358v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14358v1}
}
```

## [Enhancing Interpretability for Vision Models via Shapley Value Optimization](http://arxiv.org/abs/2512.14354v1)

Kanglong Fan, Yunqiao Yang, Chen Ma

[PDF](https://arxiv.org/pdf/2512.14354v1)
[Abstract](http://arxiv.org/abs/2512.14354v1)

### 中文摘要

深度神经网络在多个领域展示了卓越的性能，但其决策过程依然复杂难懂。尽管已有众多解释方法旨在揭示深度神经网络的“黑盒”特性，但它们存在明显的局限性：事后解释方法通常难以忠实反映模型的实际行为，而自解释神经网络则由于其专门的架构设计，在性能和兼容性方面有所牺牲。为了解决这些问题，本文提出了一种新颖的自解释框架，将Shapley值估计作为训练过程中的辅助任务。该方法实现了两大关键突破：一是将模型预测结果公平地分配到图像块中，确保其解释自然符合模型的决策逻辑；二是在结构变动较小的情况下增强模型的可解释性，同时保持其性能与兼容性。大量在多个基准数据集上的实验证明，该方法达到了当前最优的可解释性水平。

BibTeX

```
@article{2512.14354v1,
  title={Enhancing Interpretability for Vision Models via Shapley Value Optimization},
  author={Kanglong Fan and Yunqiao Yang and Chen Ma},
  journal={arXiv preprint arXiv:2512.14354v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14354v1}
}
```

## [Criminal Liability in AI-Enabled Autonomous Vehicles: A Comparative Study](http://arxiv.org/abs/2512.14330v1)

Sahibpreet Singh, Manjit Singh

[PDF](https://arxiv.org/pdf/2512.14330v1)
[Abstract](http://arxiv.org/abs/2512.14330v1)

### 中文摘要

人工智能通过自动驾驶车辆（AVs）引领交通领域的革命，但也带来了涉及违法行为的复杂刑事责任问题。本研究采用比较法分析的方法，分析了美国、德国、英国、中国和印度等国家的主要法律法规、实际责任索赔案例及学术文献，选择这些司法管辖区的原因在于它们在技术发展水平和监管方式上的显著差异。研究重点包括人类错误的归属、人工智能的道德责任主体以及在自动驾驶事故中主要责任人的识别。研究发现，全球监管环境呈现出碎片化态势：印度和美国依托分散的州级法律网络，英国则颁布了具有开创性的《2018年自动化与电动汽车法案》；德国实行严格的安全标准，并根据车辆的运行模式区别责任归属；而中国也在努力建立更为严格的责任体系。研究结论指出，推动全球范围内的法律标准协调，对于促进技术创新、保障最低风险以及明确责任归属具有重要意义。

BibTeX

```
@article{2512.14330v1,
  title={Criminal Liability in AI-Enabled Autonomous Vehicles: A Comparative Study},
  author={Sahibpreet Singh and Manjit Singh},
  journal={arXiv preprint arXiv:2512.14330v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14330v1}
}
```

## [Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity](http://arxiv.org/abs/2512.14320v1)

Shuai Dong, Jie Zhang, Guoying Zhao, Shiguang Shan, Xilin Chen

[PDF](https://arxiv.org/pdf/2512.14320v1)
[Abstract](http://arxiv.org/abs/2512.14320v1)

### 中文摘要

基于扩散模型的文本引导图像编辑虽具有强大能力，但也引发了关于误用的重大担忧，促使研究人员尝试通过无法察觉的扰动对图像进行免疫，以防止未经授权的编辑。现有的免疫成功评估指标通常依赖于比较受保护图像生成的输出与未受保护原始图像的参考输出之间的视觉差异。然而，这种方法根本忽视了图像免疫的核心要求——无论输出是否符合某一特定目标，免疫的关键在于破坏与攻击者意图的语义对齐。我们认为，免疫的成功应当定义为编辑后输出在语义上与提示不一致或在感知质量上存在明显退化，这两者都能有效阻挠恶意行为。为实现这一原则，本文提出了协同中间特征操控（Synergistic Intermediate Feature Manipulation, SIFM）方法，该方法通过双重协同目标有策略地扰动扩散过程中的中间特征：一方面最大化特征与原始编辑轨迹的偏离程度，破坏与预期编辑的语义一致性；另一方面最小化特征范数，诱导感知质量的明显退化。此外，我们还提出了免疫成功率（Immunization Success Rate, ISR）这一全新的评估指标，首次能够严格量化免疫效果的真实程度。ISR衡量免疫使得编辑在语义上无法满足提示或出现明显的感知退化的比例，且该评估借助多模态大模型（MLLMs）实现。大量实验证明，所提出的SIFM在保护视觉内容免受恶意基于扩散的操控方面达到了目前的最佳表现。

BibTeX

```
@article{2512.14320v1,
  title={Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity},
  author={Shuai Dong and Jie Zhang and Guoying Zhao and Shiguang Shan and Xilin Chen},
  journal={arXiv preprint arXiv:2512.14320v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14320v1}
}
```

## [From YOLO to VLMs: Advancing Zero-Shot and Few-Shot Detection of Wastewater Treatment Plants Using Satellite Imagery in MENA Region](http://arxiv.org/abs/2512.14312v1)

Akila Premarathna, Kanishka Hewageegana, Garcia Andarcia Mariangel

[PDF](https://arxiv.org/pdf/2512.14312v1)
[Abstract](http://arxiv.org/abs/2512.14312v1)

### 中文摘要

在中东和北非（MENA）地区，对污水处理厂（WWTPs）的需求日益增加，这对于实现可持续水资源管理具有重要意义。利用卫星图像精准识别WWTPs，有助于环境监测。传统方法如YOLOv8分割模型依赖大量人工标注，既耗时又繁琐。而研究表明，视觉-语言模型（VLMs）通过内在推理能力和注释功能，成为实现同等或更优效果的高效替代方案。本研究提出了一套针对WWTP识别的VLMs比较的结构化方法，分为零样本（zero-shot）和少样本（few-shot）两个阶段。YOLOv8在一份由政府提供的包含83,566张高分辨率卫星图像的数据集上进行了训练，图像来自埃及、沙特阿拉伯和阿联酋，正样本（WWTP）占约85%，负样本（非WWTP）占15%。评估的VLMs包括LLaMA 3.2 Vision、Qwen 2.5 VL、DeepSeek-VL2、Gemma 3、Gemini以及Pixtral 12B（Mistral），用于识别WWTP的组成部分，如圆形/矩形池、曝气池，并通过专家提示区分干扰因素，输出JSON格式的置信度和描述信息。数据集包含1207个经过验证的WWTP位置（阿联酋198个、沙特354个、埃及655个）及等量非WWTP地点，图像为600米×600米的Geo-TIFF格式（缩放比例18、EPSG：4326）。零样本评估结果显示，多个VLMs在WWTP识别中表现优于YOLOv8的真正例率，其中Gemma-3表现最佳。研究结果验证了VLMs，尤其是零样本方法，在无需注释的情况下，能够有效替代YOLOv8进行WWTP的高效分类，推动遥感技术的规模化应用。

BibTeX

```
@article{2512.14312v1,
  title={From YOLO to VLMs: Advancing Zero-Shot and Few-Shot Detection of Wastewater Treatment Plants Using Satellite Imagery in MENA Region},
  author={Akila Premarathna and Kanishka Hewageegana and Garcia Andarcia Mariangel},
  journal={arXiv preprint arXiv:2512.14312v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14312v1}
}
```

## [Leveraging LLMs for Collaborative Ontology Engineering in Parkinson Disease Monitoring and Alerting](http://arxiv.org/abs/2512.14288v1)

Georgios Bouchouras, Dimitrios Doumanas, Andreas Soularidis, Konstantinos Kotis, George A. Vouros

[PDF](https://arxiv.org/pdf/2512.14288v1)
[Abstract](http://arxiv.org/abs/2512.14288v1)

### 中文摘要

本文探讨了通过四种关键方法将大型语言模型（LLMs）应用于帕金森病（PD）监测与预警本体的构建：一次性（OS）提示技术、思维链（CoT）提示、X-HCOME以及SimX-HCOME+。主要目的是评估仅由LLMs是否可以自主创建完整的本体，以及在人机合作的条件下是否能够实现这一目标。研究还评估了LLMs在自动化本体开发中的效果及其通过人机合作获得的提升。
最初采用OS和CoT提示技术进行本体生成，验证了LLMs自主构建用于PD监测与预警的本体的能力。然而，生成的结果缺乏完整性，需大量人工修订以提升其全面性和准确性。
一种结合人工专业知识与LLM能力的混合本体工程方法——X-HCOME，显著改善了本体的完整性。这一方法生成的本体在质量上与专家构建的相似度很高。
此外，采用另一种强调持续人工监督与反复优化的混合方法——SimX-HCOME+，进一步突出持续人工参与的重要性。该方法有助于生成更完整、更准确的本体。
总体而言，本文强调了人机协作在推动本体工程中的潜力，特别是在复杂领域如PD中的应用。研究结果为未来的方向提供了启示，包括开发专门针对本体构建的GPT模型等，为该领域的深度探索提供了有价值的参考。

BibTeX

```
@article{2512.14288v1,
  title={Leveraging LLMs for Collaborative Ontology Engineering in Parkinson Disease Monitoring and Alerting},
  author={Georgios Bouchouras and Dimitrios Doumanas and Andreas Soularidis and Konstantinos Kotis and George A. Vouros},
  journal={arXiv preprint arXiv:2512.14288v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14288v1}
}
```

## [From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition](http://arxiv.org/abs/2512.14244v1)

Yiqing Zhou, Yu Lei, Shuzheng Si, Qingyan Sun, Wei Wang, Yifei Wu, Hao Wen, Gang Chen, Fanchao Qi, Maosong Sun

[PDF](https://arxiv.org/pdf/2512.14244v1)
[Abstract](http://arxiv.org/abs/2512.14244v1)

### 中文摘要

管理大量上下文依然是大型语言模型（LLMs）面临的关键瓶颈，尤其在长文档问答和自主代理等应用中，长输入带来高昂的计算成本且易引入噪声。现有的压缩技术往往通过离散的Token删除破坏局部连贯性，或依赖隐式潜编码，但这些方法受制于位置偏差且难以与封闭源API兼容。为了解决这些问题，我们提出了基于教育单元（EDU）的上下文压缩器，一种新颖的显式压缩框架，旨在同时保留全局结构和细节信息。该方法将上下文压缩重塑为“结构-然后选择”的流程。首先，我们的LingoEDU将线性文本转化为以EDU为节点的结构关系树，并严格以源索引为锚点，避免产生幻觉。其次，利用一个轻量级排序模块，从结构中筛选出与查询相关的子树进行线性化。为了全面评估结构理解能力，我们发布了StructBench，一个手工标注的包含248篇多样化文档的结构数据集。实验证明，该方法在结构预测准确率上达到了先进水平，显著优于前沿LLMs，同时降低了计算成本。此外，我们的结构感知压缩在长上下文任务和复杂的深度搜索场景中，显著提升了下游任务的性能。

BibTeX

```
@article{2512.14244v1,
  title={From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition},
  author={Yiqing Zhou and Yu Lei and Shuzheng Si and Qingyan Sun and Wei Wang and Yifei Wu and Hao Wen and Gang Chen and Fanchao Qi and Maosong Sun},
  journal={arXiv preprint arXiv:2512.14244v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14244v1}
}
```

## [Spherical Leech Quantization for Visual Tokenization and Generation](http://arxiv.org/abs/2512.14697v1)

Yue Zhao, Hanwen Jiang, Zhenlin Xu, Chutong Yang, Ehsan Adeli, Philipp Krähenbühl

[PDF](https://arxiv.org/pdf/2512.14697v1)
[Abstract](http://arxiv.org/abs/2512.14697v1)

### 中文摘要

非参数量化由于其参数效率和对大规模码本的良好扩展性，受到广泛关注。本文从晶格编码的视角出发，提出了一种统一的非参数量化方法的表达框架。晶格码的几何特性解释了在使用某些现有的无查找量化方案（如 BSQ）训练自编码器时引入辅助损失项的必要性。作为进一步的探索，我们考察了几种潜在候选，包括随机晶格、广义斐波那契晶格以及密堆球体包装晶格。其中，基于 Leech 晶格的量化方法——被命名为球面 Leech 量化（$Λ\_{24}$-SQ）——由于其高对称性和在超球面上的均匀分布，实现了简化的训练流程以及改善的重建与压缩权衡。在图像分词和压缩任务中，该量化方法在所有指标上都优于最先进的 BSQ 方法，不仅提高了重建质量，还略微减少了所需比特数。这一优化效果也延伸至最先进的自回归图像生成框架。

BibTeX

```
@article{2512.14697v1,
  title={Spherical Leech Quantization for Visual Tokenization and Generation},
  author={Yue Zhao and Hanwen Jiang and Zhenlin Xu and Chutong Yang and Ehsan Adeli and Philipp Krähenbühl},
  journal={arXiv preprint arXiv:2512.14697v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14697v1}
}
```

## [FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](http://arxiv.org/abs/2512.14601v1)

Zhaolun Li, Jichang Li, Yinqi Cai, Junye Chen, Xiaonan Luo, Guanbin Li, Rushi Lan

[PDF](https://arxiv.org/pdf/2512.14601v1)
[Abstract](http://arxiv.org/abs/2512.14601v1)

### 中文摘要

本文提出了FakeRadar，一种新颖的深度伪造视频检测框架，旨在应对实际场景中跨域泛化的挑战。现有的检测方法通常依赖于特定的操控线索，虽然在已知伪造类型上表现良好，但在面对新兴的操控技术时则展现出严重的局限性。这种泛化能力的不足源于它们难以有效适应未见过的操控模式。为此，我们利用大规模预训练模型（例如CLIP）主动探测特征空间，明确突出真实视频、已知伪造以及未见操控之间的分布差异。具体而言，FakeRadar引入了伪造异常检测（Forgery Outlier Probing），通过动态子簇建模和簇条件的异常样本生成，在估算的子簇边界附近合成异常样本，模拟超出已知操控类型的新的伪造特征。此外，我们设计了异常引导的三阶段训练（Outlier-Guided Tri-Training），利用提出的异常驱动对比学习和异常条件下的交叉熵损失，优化检测器以区分真实、伪造和异常样本。实验证明，FakeRadar在多个深度伪造视频检测基准数据集上均优于现有方法，特别是在跨域评估中，有效应对了多样化的新兴操控技术。

BibTeX

```
@article{2512.14601v1,
  title={FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos},
  author={Zhaolun Li and Jichang Li and Yinqi Cai and Junye Chen and Xiaonan Luo and Guanbin Li and Rushi Lan},
  journal={arXiv preprint arXiv:2512.14601v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14601v1}
}
```

## [Dual Language Models: Balancing Training Efficiency and Overfitting Resilience](http://arxiv.org/abs/2512.14549v1)

David Samuel, Lucas Georges Gabriel Charpentier

[PDF](https://arxiv.org/pdf/2512.14549v1)
[Abstract](http://arxiv.org/abs/2512.14549v1)

### 中文摘要

本文在未对模型架构进行任何修改的情况下，将自回归和掩码扩散训练目标相结合，从而构建出比单一目标模型更具灵活性的语言模型。自回归建模曾是一种广泛应用的方法，部分原因是其训练效率较高；但这也使其容易受到过拟合的影响。相比之下，掩码扩散模型的训练效率较低，但具有更强的抗过拟合能力。在本研究中，我们展示了双目标训练能够兼具两者的优点。为确定两种目标的最佳比例，我们在不同数据重复程度下训练并评估了50个语言模型。结果显示，在所有评估设置中，将两者目标结合使用都是最优选择，而且，无论是针对自回归还是掩码扩散下游任务的性能，最优比例都具有较高的相似性。

BibTeX

```
@article{2512.14549v1,
  title={Dual Language Models: Balancing Training Efficiency and Overfitting Resilience},
  author={David Samuel and Lucas Georges Gabriel Charpentier},
  journal={arXiv preprint arXiv:2512.14549v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14549v1}
}
```

## [Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling](http://arxiv.org/abs/2512.14474v1)

Annu Rana, Gaurav Kumar

[PDF](https://arxiv.org/pdf/2512.14474v1)
[Abstract](http://arxiv.org/abs/2512.14474v1)

### 中文摘要

大型语言模型（LLMs）在处理复杂的多步规划任务时，常常面临Constraint违反率高和解决方案不一致的问题。现有的方法如链式思考（Chain-of-Thought）和ReAct依赖于隐式的状态跟踪，缺乏明确的问题表示。受经典人工智能规划的启发，我们提出了一种“模型优先推理”（Model-First Reasoning，MFR）的方法，这是一种包括两个阶段的范式：首先，LLM构建问题的显式模型，定义实体、状态变量、行动和约束；随后，基于该模型生成解决方案方案。在涉及医疗调度、路线规划、资源分配、逻辑谜题和程序合成等多个规划领域中，MFR相较于链式思考和ReAct，能够有效减少Constraint违反，提升解决方案的质量。消融实验表明，显式建模阶段对于这些改进至关重要。我们的结果表明，许多LLM规划失败源于表示能力的不足，而非推理能力的限制，强调了显式建模是实现鲁棒且具有可解释性的人工智能代理的关键组成部分。所有的提示、评估流程和任务数据集均已公开，以促进重复验证。

BibTeX

```
@article{2512.14474v1,
  title={Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling},
  author={Annu Rana and Gaurav Kumar},
  journal={arXiv preprint arXiv:2512.14474v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14474v1}
}
```

## [PortAgent: LLM-driven Vehicle Dispatching Agent for Port Terminals](http://arxiv.org/abs/2512.14417v1)

Jia Hu, Junqi Li, Weimeng Lin, Peng Jia, Yuxiong Ji, Jintao Lai

[PDF](https://arxiv.org/pdf/2512.14417v1)
[Abstract](http://arxiv.org/abs/2512.14417v1)

### 中文摘要

车辆调度系统（VDS）在自动集装箱码头（ACT）的运营效率中扮演着关键角色。然而，由于其在不同码头之间的迁移性较低，限制了其广泛商业化应用。这一迁移性挑战源自三个方面的限制：对港口运营专家的高度依赖、对特定码头数据的高需求以及繁琐的手动部署流程。借助大型语言模型（LLMs）的出现，本文提出了PortAgent，一种基于LLM的车辆调度代理，能全自动化VDS迁移流程。该系统具有三大特点：（1）无需港口运营专家；（2）数据需求低；（3）部署速度快。具体而言，通过虚拟专家团队（VET）实现对专家依赖的消除。VET与包括知识检索器、建模师、编码员和调试员在内的四个虚拟专家协作，模拟专家团队完成VDS迁移工作流程。这些虚拟专家采用少样本学习方法，专注于码头VDS领域知识，从少量VDS样本中学习相关知识。通过一种基于检索增强生成（RAG）机制的样本检索，这些示例得以获取，减轻了对特定码头数据的高度需求。此外，建立了一套由这些虚拟专家共同完成的自动化VDS设计流程，避免了额外的人工干预。在此流程中，设计引入了受LLM反思（Reflexion）框架启发的自我修正循环，以确保流程的高效与准确。

BibTeX

```
@article{2512.14417v1,
  title={PortAgent: LLM-driven Vehicle Dispatching Agent for Port Terminals},
  author={Jia Hu and Junqi Li and Weimeng Lin and Peng Jia and Yuxiong Ji and Jintao Lai},
  journal={arXiv preprint arXiv:2512.14417v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14417v1}
}
```

## [Step-Tagging: Toward controlling the generation of Language Reasoning Models through step monitoring](http://arxiv.org/abs/2512.14332v1)

Yannis Belkhiter, Seshu Tirupathi, Giulio Zizzo, John D. Kelleher

[PDF](https://arxiv.org/pdf/2512.14332v1)
[Abstract](http://arxiv.org/abs/2512.14332v1)

### 中文摘要

近年来，语言推理模型（LRMs）在训练和推理技术的推动下取得了显著进展，能够进行更长、更准确的推理。然而，越来越多的研究表明，LRMs仍然存在效率较低的问题，常常过度生成验证和反思步骤。为了解决这一挑战，我们提出了Step-Tagging框架，这是一种轻量级的句子分类器，能够实现对LRM生成的推理步骤类型的实时标注。为了监控推理行为，我们引入了ReasonType——一种新颖的推理步骤分类体系。基于该框架，我们展示了对特定推理步骤计数的在线监测能够有效产生可解释的早期停止标准，从而优化LRM的推理过程。在标准基准数据集（如MATH500、GSM8K、AIME）以及非数学任务（如GPQA和MMLU-Pro）上评估了Step-Tagging框架。结果显示，在保持与标准生成相当的准确率的同时，Token数减少了20%到50%，在计算密集型任务中的提升尤为明显。该研究提供了一种新的方式，使对LRM生成过程的控制更加可行，也为研究LRM行为提供了新的工具。

BibTeX

```
@article{2512.14332v1,
  title={Step-Tagging: Toward controlling the generation of Language Reasoning Models through step monitoring},
  author={Yannis Belkhiter and Seshu Tirupathi and Giulio Zizzo and John D. Kelleher},
  journal={arXiv preprint arXiv:2512.14332v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14332v1}
}
```

## [SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions](http://arxiv.org/abs/2512.14277v1)

Panayiotis Smeros, Vincent Emonet, Ruijie Wang, Ana-Claudia Sima, Tarcisio Mendes de Farias

[PDF](https://arxiv.org/pdf/2512.14277v1)
[Abstract](http://arxiv.org/abs/2512.14277v1)

### 中文摘要

大型语言模型的出现促使了新型方法的不断涌现，这些方法有望更好地解决从自然语言生成结构化查询（如SPARQL查询）的挑战。然而，这些新方法大多只关注单一来源的响应准确性，忽略了诸如分布式数据存储的联合查询能力、查询生成的运行时间和成本等其他评估指标。因此，它们往往难以直接投入生产环境，或者在对精度要求较高的（潜在的联邦）知识图谱部署中表现不佳。为了解决这些问题，本文在之前工作的基础上，提出并系统评估了SPARQL-LLM——一种开源、与三元组存储无关的方法。该方法由轻量级元数据驱动，能够根据自然语言文本生成SPARQL查询。首先，我们介绍了其架构，包含用于元数据索引、提示构建以及查询生成和执行的专用组件。随后，我们基于涵盖多语种问题的最先进挑战，以及三个生物信息学领域中最重要的知识图谱提出的问题集，对其进行了评估。结果显示，SPARQL-LLM在该挑战中的F1分数提升了24%，具备对英语、西班牙语等高资源语言的适应能力，并能生成复杂且支持联邦查询的生物信息学请求。此外，我们还发现，其运行速度比参与挑战的其他系统快最多36倍，且每个问题的成本最高为0.01美元，适用于实时、低成本的文本到SPARQL的应用场景。一个在实际去中心化知识图谱上部署的应用示例可以访问https://www.expasy.org/chat。

BibTeX

```
@article{2512.14277v1,
  title={SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions},
  author={Panayiotis Smeros and Vincent Emonet and Ruijie Wang and Ana-Claudia Sima and Tarcisio Mendes de Farias},
  journal={arXiv preprint arXiv:2512.14277v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14277v1}
}
```

## [PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design](http://arxiv.org/abs/2512.14233v1)

Ruozhao Yang, Mingfei Cheng, Gelei Deng, Tianwei Zhang, Junjie Wang, Xiaofei Xie

[PDF](https://arxiv.org/pdf/2512.14233v1)
[Abstract](http://arxiv.org/abs/2512.14233v1)

### 中文摘要

渗透测试对于评估和加强系统安全以应对现实威胁至关重要，但传统的流程仍然高度依赖人工、技能密集且难以扩展。虽然近年来大型语言模型（LLMs）的进展为自动化提供了有前景的机会，但现有应用多依赖于简化的提示设计，缺乏任务分解和领域适应，导致模型行为缺乏可靠性，且难以全面展现模型在渗透测试各阶段的能力。为弥补这一缺口，我们提出了PentestEval，这是首个针对六个分解渗透测试阶段（信息收集、漏洞汇总与筛选、攻击决策、漏洞利用生成与优化）的全面评测基准。PentestEval结合了专家标注的真实标注数据，并通过全自动的评测流程，覆盖了12个真实漏洞场景中的346个任务，涵盖所有测试阶段。对九种广泛使用的LLMs进行的阶段级评估显示，模型整体表现较弱，在渗透测试工作流程的各个阶段都存在明显局限性。端到端流程的成功率仅为31%，而诸如PentestGPT、PentestAgent和VulnBot等基于LLM的系统也展现出类似的不足，自主代理几乎完全失败。这些发现表明，自动化渗透测试需要更加强大的结构化推理能力，而模块化设计有助于提升各个阶段的性能和整体效果。PentestEval为未来细粒度、阶段级评估的研究提供了基础基准，有助于推动基于LLM的自动化向更可靠的方向发展。

BibTeX

```
@article{2512.14233v1,
  title={PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design},
  author={Ruozhao Yang and Mingfei Cheng and Gelei Deng and Tianwei Zhang and Junjie Wang and Xiaofei Xie},
  journal={arXiv preprint arXiv:2512.14233v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14233v1}
}
```

## [End-to-End Learning-based Video Streaming Enhancement Pipeline: A Generative AI Approach](http://arxiv.org/abs/2512.14185v1)

Emanuele Artioli, Farzad Tashtarian, Christian Timmerer

[PDF](https://arxiv.org/pdf/2512.14185v1)
[Abstract](http://arxiv.org/abs/2512.14185v1)

### 中文摘要

视频流的主要挑战在于在保证高视频质量的同时实现流畅播放。虽然传统编码器在这种权衡中表现良好，但由于无法利用上下文信息，它们必须对整个视频数据进行编码并传输给客户端。本文提出了一种名为ELVIS（基于端到端学习的视频流增强管道）的端到端架构，该架构结合了服务器端的编码优化与客户端的生成式修补技术，以消除并重建冗余视频数据。其模块化设计使ELVIS能够集成不同的编码器、修补模型及质量评估指标，从而具备良好的扩展性以适应未来的创新。实验结果显示，现有技术在基线指标基础上最多提高了11个VMAF点，但在实现实时应用方面仍面临计算资源的挑战。ELVIS标志着将生成式人工智能引入视频流处理管道的基础性迈进，从而在不增加带宽需求的情况下实现更高质量的观看体验。

BibTeX

```
@article{2512.14185v1,
  title={End-to-End Learning-based Video Streaming Enhancement Pipeline: A Generative AI Approach},
  author={Emanuele Artioli and Farzad Tashtarian and Christian Timmerer},
  journal={arXiv preprint arXiv:2512.14185v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14185v1}
}
```

## [A Comparative Analysis of Retrieval-Augmented Generation Techniques for Bengali Standard-to-Dialect Machine Translation Using LLMs](http://arxiv.org/abs/2512.14179v1)

K. M. Jubair Sami, Dipto Sumit, Ariyan Hossain, Farig Sadeque

[PDF](https://arxiv.org/pdf/2512.14179v1)
[Abstract](http://arxiv.org/abs/2512.14179v1)

### 中文摘要

将标准语言翻译为其地区方言是自然语言处理中的一项重要挑战，主要源于数据稀缺和语言变异，这在孟加拉语中尤为突出。本文提出并比较了两种创新的检索增强生成（RAG）流程，用于标准孟加拉语向方言的翻译。第一种是基于语音转录的流程，利用大量方言语音转录文本中的句子上下文；第二种是更高效的标准化句对流程，使用结构化的本地区方言与标准孟加拉语句子对。我们在六种孟加拉语方言和多种大型语言模型（LLM）上，通过BLEU、ChrF、WER和BERTScore等指标对两种流程进行了评估。结果表明，句对流程在性能上持续优于基于转录的流程，例如在吉大港方言中将词错误率（WER）从76%降低到55%。更重要的是，该检索增强的方法使得较小的模型（如Llama-3.1-8B）能够超越许多更大规模的模型（如GPT-OSS-120B），展示了合理设计的检索策略比模型规模更为关键。本研究为低资源方言翻译提供了一种无需微调的高效解决方案，为保护语言多样性提供了具有实用价值的方案。

BibTeX

```
@article{2512.14179v1,
  title={A Comparative Analysis of Retrieval-Augmented Generation Techniques for Bengali Standard-to-Dialect Machine Translation Using LLMs},
  author={K. M. Jubair Sami and Dipto Sumit and Ariyan Hossain and Farig Sadeque},
  journal={arXiv preprint arXiv:2512.14179v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14179v1}
}
```

## [TorchTraceAP: A New Benchmark Dataset for Detecting Performance Anti-Patterns in Computer Vision Models](http://arxiv.org/abs/2512.14141v1)

Hanning Chen, Keyu Man, Kevin Zhu, Chenguang Zhu, Haonan Li, Tongbo Luo, Xizhou Feng, Wei Sun, Sreen Tallam, Mohsen Imani, Partha Kanuparthy

[PDF](https://arxiv.org/pdf/2512.14141v1)
[Abstract](http://arxiv.org/abs/2512.14141v1)

### 中文摘要

在机器学习（ML）模型中识别并解决性能反模式对于提高训练和推理的效率至关重要，但这通常需要系统基础架构、ML模型及内核开发等领域的深厚专业知识。尽管大型科技公司依靠专门的ML基础设施工程师来分析Torch的追踪信息和性能基准，但这种资源密集型的工作流程对一般计算机视觉研究人员而言基本难以触及。在众多挑战中，定位长时间执行追踪中的问题片段依然是耗时最长的任务之一，目前的ML模型（包括大型语言模型LLMs）在自动化完成此任务方面仍存在困难。在本研究中，我们首次提出了专门用于评估和提升ML模型在追踪中检测反模式能力的基准数据集。该数据集包含来自多个硬件平台的、涵盖分类、检测、分割和生成等多类计算机视觉模型的600余个PyTorch追踪。本研究还提出了一种创新的迭代方法：首先由轻量级ML模型检测出具有反模式的追踪片段，然后由大型语言模型（LLM）进行细粒度分类和有针对性的反馈。实验证明，我们的方法在检测反模式区域方面显著优于无监督聚类和基于规则的统计技术，同时也有效弥补了LLM在长文本推理中的有限上下文长度和推理效率不足的问题。

BibTeX

```
@article{2512.14141v1,
  title={TorchTraceAP: A New Benchmark Dataset for Detecting Performance Anti-Patterns in Computer Vision Models},
  author={Hanning Chen and Keyu Man and Kevin Zhu and Chenguang Zhu and Haonan Li and Tongbo Luo and Xizhou Feng and Wei Sun and Sreen Tallam and Mohsen Imani and Partha Kanuparthy},
  journal={arXiv preprint arXiv:2512.14141v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14141v1}
}
```

## [SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance](http://arxiv.org/abs/2512.14121v1)

Wenbo Tian, Ruting Lin, Hongxian Zheng, Yaodong Yang, Geng Wu, Zihao Zhang, Zhang Zhang

[PDF](https://arxiv.org/pdf/2512.14121v1)
[Abstract](http://arxiv.org/abs/2512.14121v1)

### 中文摘要

现有的智能体育分析系统主要集中于“评分与可视化”，往往缺乏自动性能诊断与可解释的训练指导。近年来，大型语言模型（LLMs）以及运动分析技术的进展，为解决上述问题带来了新的机遇。本文提出了SportsGPT，一种基于大型语言模型的可解释体育动作评估与训练指导框架，建立了从运动时序数据到专业训练建议的闭环流程。首先，针对一组高质量目标模型，提出了MotionDTW，这是一种两阶段的时序对齐算法，旨在从骨架运动序列中准确提取关键帧。随后，设计了基于知识的可解释体育动作评估模型（KISMAM），通过与目标模型对比关键帧，输出一组具备可解释性的评估指标（如不足的伸展度）。最后，提出了基于Qwen3的RAG（ Retrieval-Augmented Generation）训练指导模型——SportsRAG，它利用一个6B词元的知识库，通过检索特定领域的问答对，促使LLM生成专业的训练建议。实验结果表明，MotionDTW在时间误差和交并比（IoU）方面显著优于传统方法。消融实验验证了KISMAM和SportsRAG的有效性，证实SportsGPT在诊断准确率和专业性方面优于一般的LLMs。

BibTeX

```
@article{2512.14121v1,
  title={SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance},
  author={Wenbo Tian and Ruting Lin and Hongxian Zheng and Yaodong Yang and Geng Wu and Zihao Zhang and Zhang Zhang},
  journal={arXiv preprint arXiv:2512.14121v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14121v1}
}
```

## [HydroGEM: A Self Supervised Zero Shot Hybrid TCN Transformer Foundation Model for Continental Scale Streamflow Quality Control](http://arxiv.org/abs/2512.14106v1)

Ijaz Ul Haq, Byung Suk Lee, Julia N. Perdrial, David Baude

[PDF](https://arxiv.org/pdf/2512.14106v1)
[Abstract](http://arxiv.org/abs/2512.14106v1)

### 中文摘要

实时流量监测网络每年产生数百万条观测数据，然而在数千个偏远监测站点中保持数据质量仍然是一项耗费人力的任务。我们提出了HydroGEM（用于监测的水文广泛泛化编码器），一种面向大陆规模流量质量控制的基础模型。HydroGEM采用两阶段训练策略：首先在来自3,724个USGS站点的600多万条序列上进行自监督预训练，以学习水文表示；随后通过合成异常进行微调，用于检测与重建。该模型采用混合的时序卷积网络-Transformer架构（参数量为1420万），能够捕捉局部时间模式和长距离依赖关系，同时利用层次化归一化处理流量变化的六个数量级。在由799个站点和18种专家验证的异常类型组成的测试集中，HydroGEM在异常检测中的F1值达0.792，异常重建误差降低达68.7%，比现有方法提升36.3%。零-shot迁移至加拿大环境与气候变化部的100个站点后，F1值为0.586，超越所有基线模型，展现出跨国泛化能力。该模型在不同修正幅度下都能保持稳定的检测性能，并与季节性操作模式保持一致。HydroGEM设计用于人机协同工作流程——其输出是质控建议而非自主修正，需由专家进行审核。

BibTeX

```
@article{2512.14106v1,
  title={HydroGEM: A Self Supervised Zero Shot Hybrid TCN Transformer Foundation Model for Continental Scale Streamflow Quality Control},
  author={Ijaz Ul Haq and Byung Suk Lee and Julia N. Perdrial and David Baude},
  journal={arXiv preprint arXiv:2512.14106v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14106v1}
}
```

## [Grammar Search for Multi-Agent Systems](http://arxiv.org/abs/2512.14079v1)

Mayank Singh, Vikas Yadav, Shiva Krishna Reddy Malay, Shravan Nayak, Sai Rajeswar, Sathwik Tejaswi Madhusudhan, Eduardo Blanco

[PDF](https://arxiv.org/pdf/2512.14079v1)
[Abstract](http://arxiv.org/abs/2512.14079v1)

### 中文摘要

自动化搜索多智能体系统（Multi-Agent Systems）近年来已成为人工智能领域中的一个关键研究重点。此前的多种方法多依赖于基于大规模语言模型（LLM）的自由形式搜索，涵盖代码空间。在本研究中，我们提出了一种更为结构化的框架，通过一组固定的简单且可组合的组件来探索相同的空间。我们展示了，尽管在候选生成阶段缺少LLM的生成灵活性，我们的方法在两个领域（数学和问答）中的五个基准测试中有四个超越了现有的方法。此外，我们的方法还具有其他优势，包括更具成本效率的搜索过程以及生成具有简洁逻辑、模块化且易于解释的多智能体系统。

BibTeX

```
@article{2512.14079v1,
  title={Grammar Search for Multi-Agent Systems},
  author={Mayank Singh and Vikas Yadav and Shiva Krishna Reddy Malay and Shravan Nayak and Sai Rajeswar and Sathwik Tejaswi Madhusudhan and Eduardo Blanco},
  journal={arXiv preprint arXiv:2512.14079v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14079v1}
}
```

## [RADAR: Accelerating Large Language Model Inference With RL-Based Dynamic Draft Trees](http://arxiv.org/abs/2512.14069v1)

Junjie Ma, Jinlong Li

[PDF](https://arxiv.org/pdf/2512.14069v1)
[Abstract](http://arxiv.org/abs/2512.14069v1)

### 中文摘要

使用现代大型语言模型（LLMs）进行推理既昂贵又缓慢，而推测采样（speculative sampling）已成为解决这一问题的有效方案。然而，推测采样中用于生成候选词的模型调用次数作为预设超参数，缺乏灵活性。为了更有效地生成和利用候选词，我们提出了 RADAR，一种结合强化学习（RL）动态草稿树的创新推测采样方法。RADAR 将草稿树的生成过程建模为马尔可夫决策过程（MDP），并采用离线强化学习训练预测模型，从而实现对模型调用次数的实时决策，减少重复计算，进一步加速推理过程。在三种大型语言模型和四项任务上的评估显示，RADAR在自动回归解码基础上实现了3.17倍到4.82倍的加速。相关代码可在 https://github.com/minaduki-sora/RADAR 获取。

BibTeX

```
@article{2512.14069v1,
  title={RADAR: Accelerating Large Language Model Inference With RL-Based Dynamic Draft Trees},
  author={Junjie Ma and Jinlong Li},
  journal={arXiv preprint arXiv:2512.14069v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14069v1}
}
```

## [OpenDataArena: A Fair and Open Arena for Benchmarking Post-Training Dataset Value](http://arxiv.org/abs/2512.14051v1)

Mengzhang Cai, Xin Gao, Yu Li, Honglin Lin, Zheng Liu, Zhuoshi Pan, Qizhi Pei, Xiaoran Shang, Mengyuan Sun, Zinan Tang, Xiaoyang Wang, Zhanping Zhong, Yun Zhu, Dahua Lin, Conghui He, Lijun Wu

[PDF](https://arxiv.org/pdf/2512.14051v1)
[Abstract](http://arxiv.org/abs/2512.14051v1)

### 中文摘要

大型语言模型（LLMs）快速发展的前提是训练后数据集的质量与多样性。然而，仍存在一个关键的矛盾：尽管模型在各种基准测试中被严格评估，但支撑这些模型的数据却如同黑盒一般——其组成模糊不清、来源不明，缺乏系统性的评估。这种不透明性阻碍了实验的可复现性，也使得数据特性与模型行为之间的因果关系难以明确。为了解决这一问题，我们提出了OpenDataArena（ODA）——一个旨在全面评估训练后数据内在价值的开放平台。ODA建立了一个由四个核心支柱构成的生态系统：一是统一的训练与评估流程，确保在不同模型（如Llama、Qwen）和领域中的公正、开放的比较；二是多维度评分框架，从多个角度对数据质量进行细致描述；三是交互式数据谱系浏览器，用于可视化数据集的血缘关系及其组成来源；四是完全开源的训练、评估及评分工具包，旨在推动数据科学研究。我们在ODA平台上进行了大量实验，涵盖了来自多个领域的120余个训练数据集，基于22项基准测试，经过600多次训练运行与4千万条数据处理，获得了丰富的洞见。分析结果揭示了数据复杂度与任务性能之间的内在权衡，通过谱系追踪发现了流行基准中的冗余现象，并映射了不同数据集的血缘关系。我们将所有结果、工具和配置向公众开放，目标是 democratiz e 高质量数据评估的途径。与其仅仅维护一个排行榜，ODA更期望推动数据驱动人工智能从盲目试错向科学化、系统化转变，为数据混合规律和基础模型的策略组建奠定基础。

BibTeX

```
@article{2512.14051v1,
  title={OpenDataArena: A Fair and Open Arena for Benchmarking Post-Training Dataset Value},
  author={Mengzhang Cai and Xin Gao and Yu Li and Honglin Lin and Zheng Liu and Zhuoshi Pan and Qizhi Pei and Xiaoran Shang and Mengyuan Sun and Zinan Tang and Xiaoyang Wang and Zhanping Zhong and Yun Zhu and Dahua Lin and Conghui He and Lijun Wu},
  journal={arXiv preprint arXiv:2512.14051v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14051v1}
}
```

## [OmniDrive-R1: Reinforcement-driven Interleaved Multi-modal Chain-of-Thought for Trustworthy Vision-Language Autonomous Driving](http://arxiv.org/abs/2512.14044v1)

Zhenguo Zhang, Haohan Zhen, Yishen Wang, Le Xu, Tianchen Deng, Xuefeng Chen, Qu Chen, Bo Zhang, Wuxiong Huang

[PDF](https://arxiv.org/pdf/2512.14044v1)
[Abstract](http://arxiv.org/abs/2512.14044v1)

### 中文摘要

在自动驾驶等安全关键领域部署视觉-语言模型（VLM）面临的主要挑战是其可靠性不足，尤其是对象幻想等失误。这些问题源于模型对未加实体基础的文本链式推理（Chain-of-Thought, CoT）的依赖。尽管现有的多模态CoT方法试图进行缓解，但它们存在两大根本不足：（1）感知与推理阶段的分离，阻碍端到端的联合优化；（2）对昂贵且密集的本体标注的依赖。因此，我们提出了OmniDrive-R1，一种面向自动驾驶的端到端视觉-语言模型框架，采用交替多模态链式推理（interleaved Multi-modal Chain-of-Thought, iMCoT）机制，融合感知与推理。我们的核心创新在于引入一种由强化学习驱动的视觉定位能力，使模型能够自主调节注意力，"聚焦"于关键区域进行细粒度分析。这一能力通过纯粹的两阶段强化学习训练流程和Clip-GRPO算法实现。值得强调的是，Clip-GRPO引入了一种无需标注的、基于过程的定位奖励，该奖励不仅免除了对密集标签的需求，还通过在视觉焦点与文本推理之间强制实时跨模态一致性，避免了调用外部工具的不稳定性。在DriveLMM-o1数据集上的大量实验证明了我们模型的显著提升。与基线Qwen2.5VL-7B相比，OmniDrive-R1将整体推理得分由51.77%提升至80.35%，最终答案准确率由37.81%提升至73.62%。

BibTeX

```
@article{2512.14044v1,
  title={OmniDrive-R1: Reinforcement-driven Interleaved Multi-modal Chain-of-Thought for Trustworthy Vision-Language Autonomous Driving},
  author={Zhenguo Zhang and Haohan Zhen and Yishen Wang and Le Xu and Tianchen Deng and Xuefeng Chen and Qu Chen and Bo Zhang and Wuxiong Huang},
  journal={arXiv preprint arXiv:2512.14044v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14044v1}
}
```

## [KFS-Bench: Comprehensive Evaluation of Key Frame Sampling in Long Video Understanding](http://arxiv.org/abs/2512.14017v1)

Zongyao Li, Kengo Ishida, Satoshi Yamazaki, Xiaotong Ji, Jianquan Liu

[PDF](https://arxiv.org/pdf/2512.14017v1)
[Abstract](http://arxiv.org/abs/2512.14017v1)

### 中文摘要

我们提出了 KFS-Bench，这是首个针对长视频问答（QA）中的关键帧采样的基准，具有多场景标注功能，能够实现对采样策略的直接且稳健的评估。关键帧采样对于高效理解长视频内容至关重要。在长视频问答任务中，选取具有信息量的帧能够显著提升多模态大规模语言模型（MLLMs）的准确性和效率。KFS-Bench 解决了以往工作仅通过问答准确率间接评估帧选择质量的局限性。通过提供每个问题所需的多个不相交场景的真实标注，KFS-Bench 使我们能直接分析不同采样方法在整个长视频中捕捉关键内容的能力。利用该基准，我们对多种关键帧采样方法进行了全面研究，发现不仅采样的精确性，场景覆盖率和采样平衡也是影响问答性能的关键因素。考虑所有这些因素，我们设计了一种新颖的采样质量指标，该指标与问答准确率高度相关。此外，我们还提出了一种创新的关键帧采样方法，利用问题与视频的相关性，在采样多样性与问题-帧相似性之间实现平衡，从而增强对相关场景的覆盖。我们的方法在关键帧采样和问答性能方面均取得了优异表现。该基准现已开源，地址为 https://github.com/NEC-VID/KFS-Bench。

BibTeX

```
@article{2512.14017v1,
  title={KFS-Bench: Comprehensive Evaluation of Key Frame Sampling in Long Video Understanding},
  author={Zongyao Li and Kengo Ishida and Satoshi Yamazaki and Xiaotong Ji and Jianquan Liu},
  journal={arXiv preprint arXiv:2512.14017v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14017v1}
}
```

## [MobileWorldBench: Towards Semantic World Modeling For Mobile Agents](http://arxiv.org/abs/2512.14014v1)

Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Aditya Grover

[PDF](https://arxiv.org/pdf/2512.14014v1)
[Abstract](http://arxiv.org/abs/2512.14014v1)

### 中文摘要

虚拟模型在提升具身智能体任务性能方面展现出了巨大潜力。虽然此前的研究主要集中在像素空间的虚拟模型，但在图形用户界面（GUI）环境中，这些方法在预测未来状态中的复杂视觉元素时面临实际限制。本文提出了一种针对GUI代理的替代虚拟模型方案，即用自然语言描述状态转移，而非预测原始像素。首先，我们引入了MobileWorldBench，这是一个评估视觉-语言模型（VLM）作为移动GUI智能体虚拟模型能力的基准测试。其次，我们发布了MobileWorld，这是一个由140万样本组成的大规模数据集，显著提升了VLM的虚拟建模能力。最后，我们提出了一个新的框架，将VLM虚拟模型集成到移动智能体的规划流程中，实证表明语义化的虚拟模型通过提升任务成功率，能够直接惠及移动智能体的性能。相关代码与数据集可在https://github.com/jacklishufan/MobileWorld获取。

BibTeX

```
@article{2512.14014v1,
  title={MobileWorldBench: Towards Semantic World Modeling For Mobile Agents},
  author={Shufan Li and Konstantinos Kallidromitis and Akash Gokul and Yusuke Kato and Kazuki Kozuka and Aditya Grover},
  journal={arXiv preprint arXiv:2512.14014v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14014v1}
}
```

## [ReflCtrl: Controlling LLM Reflection via Representation Engineering](http://arxiv.org/abs/2512.13979v1)

Ge Yan, Chung-En Sun, Tsui-Wei, Weng

[PDF](https://arxiv.org/pdf/2512.13979v1)
[Abstract](http://arxiv.org/abs/2512.13979v1)

### 中文摘要

具有链式推理（CoT）能力的大规模语言模型（LLMs）在数学、编码以及一般推理等多种任务中取得了优异的表现。这些推理模型的一个显著特性是自我反思，即能够回顾和修正先前的推理步骤。虽然自我反思能提升推理性能，但也会增加推理的计算成本。在本研究中，我们从表示工程的视角探讨了自我反思。我们将模型的推理过程划分为多个步骤，识别出其中对应自我反思的步骤，并从潜在空间中提取一个引导反思行为的方向。基于这一方向，我们提出了一种逐步引导的方法，用以控制反思的频率。我们将该框架命名为ReflCtrl。实验结果表明（1）在许多情况下，反思具有冗余，尤其是在性能较强的模型中（在我们的实验中，最多可节省33.6%的推理令牌，同时保持性能不变），以及（2）模型的反思行为与内部的不确定性信号高度相关，暗示自我反思可能受模型不确定性调控。

BibTeX

```
@article{2512.13979v1,
  title={ReflCtrl: Controlling LLM Reflection via Representation Engineering},
  author={Ge Yan and Chung-En Sun and Tsui-Wei and Weng},
  journal={arXiv preprint arXiv:2512.13979v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13979v1}
}
```

## [MURIM: Multidimensional Reputation-based Incentive Mechanism for Federated Learning](http://arxiv.org/abs/2512.13955v1)

Sindhuja Madabushi, Dawood Wasif, Jin-Hee Cho

[PDF](https://arxiv.org/pdf/2512.13955v1)
[Abstract](http://arxiv.org/abs/2512.13955v1)

### 中文摘要

联邦学习（FL）已成为一种先进的隐私保护机器学习范式，允许参与方分享模型更新而非原始数据。然而，联邦学习仍然面临诸多关键挑战，包括客户端激励不足、隐私风险以及资源限制。评估客户端的可靠性对于公平分配激励和确保每个客户端的数据能够对全局模型做出有意义的贡献至关重要。为此，本文提出了多维声誉基础激励机制（MURIM），该机制在考虑客户端的可靠性、隐私、资源容量和公平性的同时，有效防止恶意或不可靠的客户端获得不应得的奖励。MURIM根据客户端的贡献、延迟和声誉进行激励分配，并配备了可靠性验证模块。大量在MNIST、FMNIST和ADULT Income数据集上的实验结果表明，MURIM在公平性指标方面实现了最高18%的提升，降低了隐私攻击成功率5%至9%，并在抵御投毒和噪声梯度攻击方面相较于最先进的基线方法提升了85%的鲁棒性。总体而言，MURIM有效抵御对抗性威胁，促进公平诚信的参与，并在异构和动态的联邦环境中实现了模型的稳定收敛。

BibTeX

```
@article{2512.13955v1,
  title={MURIM: Multidimensional Reputation-based Incentive Mechanism for Federated Learning},
  author={Sindhuja Madabushi and Dawood Wasif and Jin-Hee Cho},
  journal={arXiv preprint arXiv:2512.13955v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13955v1}
}
```

## [Informing Acquisition Functions via Foundation Models for Molecular Discovery](http://arxiv.org/abs/2512.13935v1)

Qi Chen, Fabio Ramos, Alán Aspuru-Guzik, Florian Shkurti

[PDF](https://arxiv.org/pdf/2512.13935v1)
[Abstract](http://arxiv.org/abs/2512.13935v1)

### 中文摘要

贝叶斯优化（BO）是一种关键的方法，用于通过估算分子到其性质的映射关系，推动分子发现的进程，同时寻求最优候选。在传统做法中，贝叶斯优化通过迭代更新该映射的概率替代模型，并基于模型导出的采集函数进行优化，以指导分子的选择。然而，在数据不足且先验知识有限且候选空间庞大的情况下，其性能表现受到限制。大型语言模型（LLMs）和化学基础模型带来了丰富的先验信息，有助于提升贝叶斯优化的效果，但由于高维特征、昂贵的上下文学习成本以及深层贝叶斯替代模型带来的计算压力，难以充分发挥其潜力。为应对这些挑战，本文提出了一种无似然的贝叶斯优化方法，避免了显式替代模型，直接利用通用大型语言模型和专门的化学基础模型提供的先验信息，指导采集函数的设计。该方法还学习分子搜索空间的树状划分，并在局部区域引入采集函数，从而通过蒙特卡洛树搜索实现高效的候选分子选择。进一步结合粗粒度的LLM基聚类技术，显著提升对大规模候选集的扩展能力，将采集函数的评估限制在统计性质较优的簇中。透过大量实验和消融分析，验证了该方法在LLM引导的分子发现贝叶斯优化中，显著增强了扩展性、鲁棒性及样本效率。

BibTeX

```
@article{2512.13935v1,
  title={Informing Acquisition Functions via Foundation Models for Molecular Discovery},
  author={Qi Chen and Fabio Ramos and Alán Aspuru-Guzik and Florian Shkurti},
  journal={arXiv preprint arXiv:2512.13935v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13935v1}
}
```

## [Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Functional Materials Discovery](http://arxiv.org/abs/2512.13930v1)

Samuel Rothfarb, Megan C. Davis, Ivana Matanovic, Baikun Li, Edward F. Holby, Wilton J. M. Kort-Kamp

[PDF](https://arxiv.org/pdf/2512.13930v1)
[Abstract](http://arxiv.org/abs/2512.13930v1)

### 中文摘要

人工智能正在重塑科学探索，但目前大多数方法仅能自动执行程序性任务，缺乏科学推理的能力，从而限制了自主发现。我们提出了材料智能体（Materials Agents for Simulation and Theory in Electronic-structure Reasoning，简称MASTER），这是一种主动学习框架，允许大型语言模型自主设计、执行并解析原子级模拟。在MASTER中，一个多模态系统将自然语言转化为密度泛函理论的工作流程，而更高级别的推理智能体则通过一系列策略引导科学发现，包括单智能体基础模型以及三种多智能体方法：同行评审、三方筛选-排序和三方筛选-表单。在两个化学应用场景中——铜表面过渡金属（M）原子吸附以及M-N-C催化剂的研究中，基于推理的探索相比试错筛选显著减少了多达90%的原子级模拟次数。推理路径揭示了许多基于化学原则的决策，这些决策无法通过随机抽样或语义偏差来解释。总之，多智能体协作加速了材料发现，开启了自主科学探索的新时代。

BibTeX

```
@article{2512.13930v1,
  title={Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Functional Materials Discovery},
  author={Samuel Rothfarb and Megan C. Davis and Ivana Matanovic and Baikun Li and Edward F. Holby and Wilton J. M. Kort-Kamp},
  journal={arXiv preprint arXiv:2512.13930v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13930v1}
}
```

## [Generative AI for Video Translation: A Scalable Architecture for Multilingual Video Conferencing](http://arxiv.org/abs/2512.13904v1)

Amirkia Rafiei Oskooei, Eren Caglar, Ibrahim Sahin, Ayse Kayabay, Mehmet S. Aktas

[PDF](https://arxiv.org/pdf/2512.13904v1)
[Abstract](http://arxiv.org/abs/2512.13904v1)

### 中文摘要

本论文针对视频翻译等应用中的级联生成式人工智能（AI）管道的实时部署，面临着重大系统层面挑战，包括多级模型推理的累计延迟以及导致多用户视频会议应用难以扩展的二次方（$\mathcal{O}(N^2)$）计算复杂度。提出并评估了一种实用的系统级框架，以缓解这些关键瓶颈。所设计的架构引入了轮流机制，显著将多用户场景中的计算复杂度从二次方降低到线性，同时采用分段处理协议，确保推理延迟在感知上接近实时体验。我们建立了一个概念验证的管道，并在包括普通GPU（如NVIDIA RTX 4060）、云端GPU（NVIDIA T4）以及企业级GPU（NVIDIA A100）在内的多层硬件环境中进行了严谨的性能分析。客观评估显示，该系统在现代硬件上实现了实时处理（$τ< 1.0$）。此外，通过主观用户研究验证了该方法，结果表明，用户对可预期的初始处理延迟表现出较高的接受度，以换取顺畅、不中断的播放体验。这项工作提供了一套经过验证的端到端系统设计，为多语言通信平台中可扩展、实时生成式AI应用的部署提供了切实可行的解决方案。

BibTeX

```
@article{2512.13904v1,
  title={Generative AI for Video Translation: A Scalable Architecture for Multilingual Video Conferencing},
  author={Amirkia Rafiei Oskooei and Eren Caglar and Ibrahim Sahin and Ayse Kayabay and Mehmet S. Aktas},
  journal={arXiv preprint arXiv:2512.13904v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13904v1}
}
```

## [State-Dependent Refusal and Learned Incapacity in RLHF-Aligned Language Models](http://arxiv.org/abs/2512.13762v1)

TK Lee

[PDF](https://arxiv.org/pdf/2512.13762v1)
[Abstract](http://arxiv.org/abs/2512.13762v1)

### 中文摘要

大型语言模型（LLMs）广泛被用作通用工具，但在长时间交互中，可能会暴露出未被传统量化指标捕捉的行为模式。我们提出一种定性案例研究方法，用于审查长远交互中与政策相关的行为选择性。在一次86轮的对话中，同一模型在广泛且非敏感领域表现出正常表现（NP），而在涉及提供者或政策敏感领域，则反复表现出功能性拒绝（FR），在不同领域中呈现出NP与FR之间的一贯不对称。借鉴习得性无助的类比，我们引入习得无能（LI）作为描述这种选择性屏蔽行为的特征，但不涉及其是否具有意图或内部机制。我们将回应划分为三种状态（NP、FR和元叙事Bre，MN），并显示在敏感情境中，MN角色框架叙事往往与拒绝行为同时出现。总体而言，本研究提出了一种基于可观察行为的交互层面审查框架，并引入LI作为研究潜在对齐副作用的视角，显示有必要在不同用户和模型间进行进一步的探索。

BibTeX

```
@article{2512.13762v1,
  title={State-Dependent Refusal and Learned Incapacity in RLHF-Aligned Language Models},
  author={TK Lee},
  journal={arXiv preprint arXiv:2512.13762v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13762v1}
}
```

## [Network-Wide Traffic Volume Estimation from Speed Profiles using a Spatio-Temporal Graph Neural Network with Directed Spatial Attention](http://arxiv.org/abs/2512.13758v1)

Léo Hein, Giovanni de Nunzio, Giovanni Chierchia, Aurélie Pirayre, Laurent Najman

[PDF](https://arxiv.org/pdf/2512.13758v1)
[Abstract](http://arxiv.org/abs/2512.13758v1)

### 中文摘要

现有的交通量估算方法通常只针对装备传感器的道路的交通预测，或利用附近传感器数据进行空间插补以填补缺失的交通量。虽然预测模型在设计上通常忽略未监测道路，但空间插补方法则明确旨在实现全网范围的估算；然而，这种方法依赖于推断时的交通量数据，限制了其在传感器稀缺城市中的应用。与交通量数据不同，行驶车辆的速度和静态道路属性更为广泛可获取，且能够实现大部分城市路网道路段的全面覆盖。在本研究中，我们提出了混合导向注意力时空图神经网络（HDA-STGNN），这是一种归纳式深度学习框架，旨在解决全网范围的交通量估算问题。我们的方法结合了车速轮廓、静态道路属性和道路网络拓扑信息，用以预测全网所有道路段的日交通量变化趋势。为了验证我们方法的有效性，我们进行了大量消融实验，结果显示该模型能够捕捉复杂的时空依存关系，并突出了拓扑信息在无需交通量数据的情况下实现全网交通量精确估算中的重要作用。

BibTeX

```
@article{2512.13758v1,
  title={Network-Wide Traffic Volume Estimation from Speed Profiles using a Spatio-Temporal Graph Neural Network with Directed Spatial Attention},
  author={Léo Hein and Giovanni de Nunzio and Giovanni Chierchia and Aurélie Pirayre and Laurent Najman},
  journal={arXiv preprint arXiv:2512.13758v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13758v1}
}
```

## [MIDUS: Memory-Infused Depth Up-Scaling](http://arxiv.org/abs/2512.13751v1)

Taero Kim, Hoyoon Byun, Youngjun Choi, Sungrae Park, Kyungwoo Song

[PDF](https://arxiv.org/pdf/2512.13751v1)
[Abstract](http://arxiv.org/abs/2512.13751v1)

### 中文摘要

扩展大型语言模型（LLMs）容量的策略应在不显著增加参数规模或推理成本的前提下实现性能的提升。深度增强扩展（DUS）作为一种前景广阔的方案，通过复制网络层并结合持续预训练（CPT）技术，取得了一定成效，但其对前馈网络（FFN）的依赖限制了效率和潜在收益。本文提出了一种称为存储增强深度扩展（MIDUS）的新方法，该方法用头部记忆（HML）层取代了复制块中的FFN。受到注意力头在层间和层内扮演不同角色的启发，MIDUS为每个注意力头分配了独立的记忆库，实现了头部级的检索和信息注入到后续层，同时保持了头部功能结构的独立性。这一设计将稀疏存储访问与头部级表示相结合，并引入了高效的每头值分解模块，有效缓解了传统方法在效率与性能之间的权衡。在持续预训练的系列实验中，MIDUS相较于强大的DUS基线展现出稳健的性能提升，同时保持了较低的参数开销。我们的研究表明，MIDUS通过其头部记忆设计，为深度模型扩展提供了一种具有吸引力的资源高效替代方案，优于传统的FFN复制策略。

BibTeX

```
@article{2512.13751v1,
  title={MIDUS: Memory-Infused Depth Up-Scaling},
  author={Taero Kim and Hoyoon Byun and Youngjun Choi and Sungrae Park and Kyungwoo Song},
  journal={arXiv preprint arXiv:2512.13751v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13751v1}
}
```

## [The algorithmic muse and the public domain: Why copyrights legal philosophy precludes protection for generative AI outputs](http://arxiv.org/abs/2512.13750v1)

Ezieddin Elmahjub

[PDF](https://arxiv.org/pdf/2512.13750v1)
[Abstract](http://arxiv.org/abs/2512.13750v1)

### 中文摘要

生成式人工智能（GenAI）的输出内容不具有著作权保护。本文对此提出了理由。我们避免采用传统的以“新颖性”和“作者身份”为核心的法律原则分析，而是重新审视著作权的基础理念。GenAI 本质上割断了人与表达形式之间的直接创作联系。传统理论中强调的功利性激励、劳动价值和人格权益等，无法为著作权的保护提供连贯的正当理由。公共领域被视为知识创造的默认基础，寻求为 GenAI 输出内容申请著作权的人应承担举证责任。授予未经加工的 GenAI 输出内容以著作权，不仅在哲学上站不住脚，还会引发前所未有的数字共享空间的限制，形成法律上的困境，阻碍未来的创新。本文倡导明确区分：人类对 AI 生成作品的创造性贡献可能值得保护，但纯粹的算法输出应保持在公共领域内。

BibTeX

```
@article{2512.13750v1,
  title={The algorithmic muse and the public domain: Why copyrights legal philosophy precludes protection for generative AI outputs},
  author={Ezieddin Elmahjub},
  journal={arXiv preprint arXiv:2512.13750v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13750v1}
}
```

## [Human-AI Collaboration Mechanism Study on AIGC Assisted Image Production for Special Coverage](http://arxiv.org/abs/2512.13739v1)

Yajie Yang, Yuqing Zhao, Xiaochao Xi, Yinan Zhu

[PDF](https://arxiv.org/pdf/2512.13739v1)
[Abstract](http://arxiv.org/abs/2512.13739v1)

### 中文摘要

人工智能生成内容（AIGC）在辅助新闻报道中的图像制作引发了行业争议，同时也引起媒体机构的广泛关注。核心问题包括虚假信息、真实性、语义准确性以及可解释性。目前大部分AIGC工具仍然是“不透明的黑箱”模型，难以同时满足内容的准确性与语义一致性的双重要求，进而带来伦理、社会技术以及信任方面的困境。本文探讨了在新闻特殊报道中实现可控图像生成的路径，基于中国媒体机构的项目进行了两项实验：（1）实验一通过标准化提示在三个不同场景中测试了跨平台的适应性，揭示了由于训练语料偏差和平台过滤机制导致的语义一致性、文化特异性以及视觉真实性方面的差异；（2）实验二构建了一个人为参与的模块化流程，结合高精度分割模型（SAM、GroundingDINO）、语义对齐工具（BrushNet）和风格调控方法（Style-LoRA、Prompt-to-Prompt），通过基于CLIP的语义评分、NSFW/OCR/YOLO过滤以及内容验证，确保报道的准确性和可信度，同时实现可追溯部署以维护语义表达的完整性。由此提出一种人机协作的AIGC图像生成机制，应用于新闻的特殊报道，并建议评价指标包括人物身份稳定性（CIS）、文化表达准确性（CEA）以及用户与公众的适宜性（U-PA）。

BibTeX

```
@article{2512.13739v1,
  title={Human-AI Collaboration Mechanism Study on AIGC Assisted Image Production for Special Coverage},
  author={Yajie Yang and Yuqing Zhao and Xiaochao Xi and Yinan Zhu},
  journal={arXiv preprint arXiv:2512.13739v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13739v1}
}
```

## [Instilling Organisational Values in Firefighters through Simulation-Based Training](http://arxiv.org/abs/2512.13737v1)

Nardine Osman, Manel Rodriguez-Soto, Jordi Sabater-Mir

[PDF](https://arxiv.org/pdf/2512.13737v1)
[Abstract](http://arxiv.org/abs/2512.13737v1)

### 中文摘要

在消防及其他应急行动中，在压力下做出的决策具有深远的伦理意义，且可能对事件的结果和消防员的安全产生重大影响。传统的培训方法虽具有基础性作用，但常常难以充分准备消防员应对在混乱应急环境中所固有的复杂伦理困境和价值冲突。本文提出了一种增强消防员培训的概念框架，通过系统性地将部门价值观融入基于模拟的培训过程。这一方法有助于加深价值观的内化，提升在压力情况下基于价值的决策能力。此外，所提出的工具还可用于评估和优化部门操作规程，使其更好地与核心价值观保持一致。

BibTeX

```
@article{2512.13737v1,
  title={Instilling Organisational Values in Firefighters through Simulation-Based Training},
  author={Nardine Osman and Manel Rodriguez-Soto and Jordi Sabater-Mir},
  journal={arXiv preprint arXiv:2512.13737v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13737v1}
}
```

## [Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](http://arxiv.org/abs/2512.13734v1)

Haochen Yuan, Yang Zhang, Xiang He, Quan Z. Sheng, Zhongjie Wang

[PDF](https://arxiv.org/pdf/2512.13734v1)
[Abstract](http://arxiv.org/abs/2512.13734v1)

### 中文摘要

随着云边协作的兴起，推荐服务越来越多地在分布式环境中进行训练。联邦推荐（FR）实现了多端协同训练，同时通过共享模型参数而非原始数据，有效保护了用户隐私。然而，由于大量的物品嵌入参数，尤其是在模型规模庞大的情况下，严重影响了通信效率。现有的研究主要集中于提升FR模型的效率，但在嵌入参数的开销问题上尚缺乏足够的关注。针对这一空白，本文提出了一种基于参数高效微调（PEFT）技术的FR训练框架，旨在降低传输的嵌入参数数量。该方法提供了一种轻量级、插件式的解决方案，能够无缝集成到现有的FR方法中。除采用常见的PEFT技术，如LoRA和基于哈希的编码外，我们还探索了残差量化变分自编码器（RQ-VAE）作为一种新颖的PEFT策略。大量实验证明，基于不同FR模型骨干架构和数据集，我们的方法不仅显著减少了通信开销，还提升了模型的准确性。相关源码已开源，地址为https://github.com/young1010/FedPEFT。

BibTeX

```
@article{2512.13734v1,
  title={Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation},
  author={Haochen Yuan and Yang Zhang and Xiang He and Quan Z. Sheng and Zhongjie Wang},
  journal={arXiv preprint arXiv:2512.13734v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13734v1}
}
```

## [Federated Few-Shot Learning for Epileptic Seizure Detection Under Privacy Constraints](http://arxiv.org/abs/2512.13717v1)

Ekaterina Sysoykova, Bernhard Anzengruber-Tanase, Michael Haslgrubler, Philipp Seidl, Alois Ferscha

[PDF](https://arxiv.org/pdf/2512.13717v1)
[Abstract](http://arxiv.org/abs/2512.13717v1)

### 中文摘要

针对基于脑电图（EEG）进行癫痫发作检测，已研发了多种深度学习方法，但其中大多数依赖于大量集中式带标签的数据集。在临床实践中，EEG数据通常稀缺、分散存储于不同机构，并受到严格的隐私保护法规限制，禁止数据共享。因此，在实际医疗环境中构建可用的基于人工智能的癫痫发作检测模型仍面临诸多挑战。为应对这些限制，本文提出了一种用于个性化EEG发作检测的两阶段联邦少样本学习（FFSL）框架。该方法在包含六类EEG事件的TUH事件库上进行了训练和评估。在第一阶段，通过联邦学习对预训练的生物信号变换器（BIOT）模型进行微调，以模拟非独立同分布（non-IID）的多家医院场景，实现无需集中存储所有EEG记录的共享表征学习。第二阶段，采用联邦少样本个性化策略，仅利用五段标注EEG片段对分类器进行调优，以保持癫痫特异性信息的同时，利用跨站知识增强模型能力。联邦微调在总体性能上达到了0.43的平衡精度（集中式为0.52）、0.42的Cohen's κ系数（集中式为0.49）、以及0.69的加权F1分数（集中式为0.74）。在FFSL的个性化阶段，客户端模型在四个不同事件分布异质场景下实现了平均0.77的平衡精度、0.62的Cohen's κ系数和0.73的加权F1分数。这些结果显示，FFSL方法在现实数据获取和隐私保护的限制条件下，能够有效支持患者适应性癫痫发作检测。

BibTeX

```
@article{2512.13717v1,
  title={Federated Few-Shot Learning for Epileptic Seizure Detection Under Privacy Constraints},
  author={Ekaterina Sysoykova and Bernhard Anzengruber-Tanase and Michael Haslgrubler and Philipp Seidl and Alois Ferscha},
  journal={arXiv preprint arXiv:2512.13717v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13717v1}
}
```

## [ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](http://arxiv.org/abs/2512.13716v1)

Yitong Luo, Ziang Chen, Hou Hei Lam, Jiayu zhan, Junqi Wang, Zhenliang Zhang, Xue Feng

[PDF](https://arxiv.org/pdf/2512.13716v1)
[Abstract](http://arxiv.org/abs/2512.13716v1)

### 中文摘要

个性化决策在人与人工智能交互中至关重要，能够使AI代理人的行为与个体用户的价值偏好保持一致。随着AI系统在实际应用中的不断拓展，超越任务完成或集体一致性，适应个性化价值的挑战变得愈发迫切。为此，我们提出了一种基于价值导向的个性化决策方法。人类价值作为稳定且可迁移的信号，有助于支持在不同场景下保持一致性和普遍适应性的行为。与以外部奖励和激励驱动的任务导向范式相比，价值驱动的决策方式增强了可解释性，使代理人能够甚至在新颖情境中做出适宜的行为。我们引入“ValuePilot”框架，包含两个阶段：一个是构建多样且带有价值注释情景的数据集生成工具（DGT），另一个是学习依据个人价值偏好进行评估的决策模块（DMM）。DGT通过人类与大型语言模型（LLM）合作流程，生成丰富的场景数据。DMM则学习在具体情境下基于个人价值偏好做出评估，实现个性化、情境敏感的决策。在未见过的场景中进行评估时，DMM在与人类行为选择的一致性方面优于包括GPT-5、Claude-Sonnet-4、Gemini-2-flash和Llama-3.1-70b在内的强大LLM基线。我们的结果表明，价值驱动的决策是一条高效且具有扩展性的路径，有助于构建具有人性化、可解释的个性化AI代理。

BibTeX

```
@article{2512.13716v1,
  title={ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making},
  author={Yitong Luo and Ziang Chen and Hou Hei Lam and Jiayu zhan and Junqi Wang and Zhenliang Zhang and Xue Feng},
  journal={arXiv preprint arXiv:2512.13716v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13716v1}
}
```

## [Meta Hierarchical Reinforcement Learning for Scalable Resource Management in O-RAN](http://arxiv.org/abs/2512.13715v1)

Fatemeh Lotfi, Fatemeh Afghah

[PDF](https://arxiv.org/pdf/2512.13715v1)
[Abstract](http://arxiv.org/abs/2512.13715v1)

### 中文摘要

随着现代应用需求日益复杂，无线网络必须具备实时自适应能力和高效的资源管理能力。开放式无线接入网（O-RAN）架构及其引入的网络智能控制器（RIC）模块，已成为实现动态资源调度和网络切片的关键解决方案。尽管基于人工智能（AI）的方法展现出一定潜力，许多现有方案在应对不可预测且高度动态的网络环境时仍面临性能瓶颈。本文提出了一种自适应的元层次强化学习（Meta-HRL）框架，受模型无关元学习（MAML）启发，用于联合优化O-RAN中的资源分配和网络切片。该框架结合了层级控制与元学习，支持全局与局部的自适应：高层控制器负责跨切片的资源分配，底层代理则执行切片内部的调度任务。自适应的元更新机制根据时间差误差的方差为任务赋权，从而增强稳定性，优先处理复杂的网络场景。理论分析验证了该两级学习过程的次线性收敛性和遗憾保证。仿真结果显示，与传统的强化学习（RL）和元强化学习（meta-RL）方法相比，该方法在网络管理效率上提升了19.8%，并实现了更快的适应速度和更高的QoS满足率，覆盖了增强型移动宽带（eMBB）、超可靠低延迟通信（URLLC）和大规模机器类通信（mMTC）切片。此外，通过消融及扩展性研究证实了该方法的鲁棒性，在网络规模扩展时仍能实现最高40%的适应速度提升，并保持公平性、时延与吞吐量的一致性。

BibTeX

```
@article{2512.13715v1,
  title={Meta Hierarchical Reinforcement Learning for Scalable Resource Management in O-RAN},
  author={Fatemeh Lotfi and Fatemeh Afghah},
  journal={arXiv preprint arXiv:2512.13715v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13715v1}
}
```

## [Safe2Harm: Semantic Isomorphism Attacks for Jailbreaking Large Language Models](http://arxiv.org/abs/2512.13703v1)

Fan Yang

[PDF](https://arxiv.org/pdf/2512.13703v1)
[Abstract](http://arxiv.org/abs/2512.13703v1)

### 中文摘要

大型语言模型（LLMs）在各种任务中表现出色，但其安全性漏洞易被攻击者利用以生成有害内容，造成对社会各领域的负面影响。目前大部分破解方法主要集中在提示工程或对抗性优化。然而，我们发现一个以前被忽视的现象：许多有害场景在深层原理上与合法场景高度一致。基于此，我们提出了安全与危害语义同构攻击（Safe2Harm）方法，通过四个阶段实现高效破解：首先，将有害问题重写为语义安全、原理相似的问题；其次，提取两者之间的主题映射关系；第三，让大型语言模型针对安全问题生成详细回答；最后，基于主题映射关系，对安全回答进行逆向重写，从而得到有害输出。在7个主流大型语言模型及三类基准数据集上的实验结果表明，Safe2Harm展现出强大的破解能力，其整体性能优于现有方法。此外，我们还构建了一个包含358个样本的具有挑战性的有害内容评估数据集，用于评估现有有害内容检测方法的有效性，该数据集可以用于大型语言模型输入输出的过滤，以增强防御能力。

BibTeX

```
@article{2512.13703v1,
  title={Safe2Harm: Semantic Isomorphism Attacks for Jailbreaking Large Language Models},
  author={Fan Yang},
  journal={arXiv preprint arXiv:2512.13703v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13703v1}
}
```

## [Leveraging LLMs for Structured Data Extraction from Unstructured Patient Records](http://arxiv.org/abs/2512.13700v1)

Mitchell A. Klusty, Elizabeth C. Solie, Caroline N. Leach, W. Vaiden Logan, Lynnet E. Richey, John C. Gensel, David P. Szczykutowicz, Bryan C. McLellan, Emily B. Collier, Samuel E. Armstrong, V. K. Cody Bumgardner

[PDF](https://arxiv.org/pdf/2512.13700v1)
[Abstract](http://arxiv.org/abs/2512.13700v1)

### 中文摘要

手工病历审查仍然是临床研究中极其费时且资源消耗巨大的环节，其需要专家从非结构化的电子健康记录（EHR）描述中提取复杂信息。我们提出了一种安全、模块化的框架，用于从临床记录中自动提取结构化特征，利用在机构批准、符合健康保险流通与责任法案（HIPAA）合规计算基础设施上本地部署的大型语言模型（LLMs）。该系统将增强检索生成（RAG）和结构化响应方法集成到一种广泛可部署、可扩展的容器中，旨在为多种临床领域提供特征提取支持。在评估中，该框架在大量患者病历中涉及的多种医学特征识别方面表现出高准确率，与专家标注的数据集相比，还发现了多次人工审查中遗漏的标注错误。这一框架展示了LLM系统在自动提取信息、减少手工病历审查负担并提高数据采集一致性方面的潜力，从而加速临床研究进程。

BibTeX

```
@article{2512.13700v1,
  title={Leveraging LLMs for Structured Data Extraction from Unstructured Patient Records},
  author={Mitchell A. Klusty and Elizabeth C. Solie and Caroline N. Leach and W. Vaiden Logan and Lynnet E. Richey and John C. Gensel and David P. Szczykutowicz and Bryan C. McLellan and Emily B. Collier and Samuel E. Armstrong and V. K. Cody Bumgardner},
  journal={arXiv preprint arXiv:2512.13700v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13700v1}
}
```

## [Explainable Preference Learning: a Decision Tree-based Surrogate Model for Preferential Bayesian Optimization](http://arxiv.org/abs/2512.14263v1)

Nick Leenders, Thomas Quadt, Boris Cule, Roy Lindelauf, Herman Monsuur, Joost van Oijen, Mark Voskuijl

[PDF](https://arxiv.org/pdf/2512.14263v1)
[Abstract](http://arxiv.org/abs/2512.14263v1)

### 中文摘要

当前的偏好贝叶斯优化方法主要依赖高斯过程（GP）作为代理模型。这些模型难以解释，难以处理类别数据，并且计算复杂度较高，限制了其在实际中的应用。本文引入了一种本质上具有可解释性的基于决策树的代理模型，该模型能够同时处理类别数据和连续数据，并具有良好的扩展性以适应大规模数据集。在对八个逐渐变得尖锐的优化函数进行的大量数值实验中，结果显示我们的模型在尖锐函数上的性能优于基于高斯过程的替代方案，而在非尖锐函数上也仅略低一点。此外，我们还将该模型应用到真实的寿司数据集，验证其学习个人寿司偏好的能力。最后，我们展示了利用历史偏好数据加速新用户偏好优化的初步尝试。

BibTeX

```
@article{2512.14263v1,
  title={Explainable Preference Learning: a Decision Tree-based Surrogate Model for Preferential Bayesian Optimization},
  author={Nick Leenders and Thomas Quadt and Boris Cule and Roy Lindelauf and Herman Monsuur and Joost van Oijen and Mark Voskuijl},
  journal={arXiv preprint arXiv:2512.14263v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14263v1}
}
```

## [Towards Explainable Quantum AI: Informing the Encoder Selection of Quantum Neural Networks via Visualization](http://arxiv.org/abs/2512.14181v1)

Shaolun Ruan, Feng Liang, Rohan Ramakrishna, Chao Ren, Rudai Yan, Qiang Guan, Jiannan Li, Yong Wang

[PDF](https://arxiv.org/pdf/2512.14181v1)
[Abstract](http://arxiv.org/abs/2512.14181v1)

### 中文摘要

量子神经网络（QNNs）代表了量子计算与神经网络架构的有希望的融合，能够实现加速处理高维度、纠缠数据的高效性。编码器是QNN的关键组成部分，负责将经典输入数据映射为量子态。然而，选择合适的编码器仍然面临重大挑战，这主要由于缺乏系统性指导以及现有方法的试错性质所导致。此过程还受到两个核心难题的制约：一是难以在训练前评估编码的量子状态，二是缺乏直观分析编码器在区分数据特征方面能力的方法。为了解决这些问题，我们提出了一种新颖的可视化工具——XQAI-Eyes，帮助QNN开发者比较经典数据特征与其对应的编码量子态，并检查不同类别之间的混合量子状态。通过结合经典与量子两个视角，XQAI-Eyes促进了对编码器如何影响QNN性能的深入理解。在多个数据集和编码器设计上的评估显示，XQAI-Eyes具有支持探索编码器设计与QNN效果关系的潜力，提供了一种全面和透明的优化量子编码器的方法。此外，领域专家利用XQAI-Eyes总结出两条关于量子编码器选择的关键实践原则，基于图案保持和特征映射的基础。

BibTeX

```
@article{2512.14181v1,
  title={Towards Explainable Quantum AI: Informing the Encoder Selection of Quantum Neural Networks via Visualization},
  author={Shaolun Ruan and Feng Liang and Rohan Ramakrishna and Chao Ren and Rudai Yan and Qiang Guan and Jiannan Li and Yong Wang},
  journal={arXiv preprint arXiv:2512.14181v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14181v1}
}
```

## [Bias-Variance Trade-off for Clipped Stochastic First-Order Methods: From Bounded Variance to Infinite Mean](http://arxiv.org/abs/2512.14686v1)

Chuan He

[PDF](https://arxiv.org/pdf/2512.14686v1)
[Abstract](http://arxiv.org/abs/2512.14686v1)

### 中文摘要

随机优化在现代机器学习中具有基础性作用。近年来的研究将随机一阶方法（SFOMs）的分析拓展到重尾噪声的情形，此类噪声在实践中十分常见，其中裁剪技术已被证明是一种有效的控制重尾梯度的重要手段。大量理论进展表明，SFOMs的oracle复杂度依赖于噪声的尾指数α。然而，现有的复杂度结果通常仅涵盖α在（1，2]区间的情况，即噪声具有有限均值的情形，当α接近1时，复杂度界趋向无穷大。本文针对尾指数α在（0，2]范围内的噪声的完整情况展开研究，涵盖从具有有限方差的噪声到均值无限的噪声，而后者的研究较少。通过对梯度裁剪中偏差-方差权衡的创新分析，我们证明当噪声尾的对称性度量得到控制时，裁剪的随机一阶方法在任何α∈（0，2]的重尾噪声环境下都能实现更优的复杂度保证。我们对偏差-方差权衡的分析不仅为这一全范围尾指数的裁剪方法提供了新颖的统一复杂度界，还具有计算简便的特点，并且能够结合经典的轻尾噪声分析，用以推导重尾噪声环境下的oracle复杂度界。最后，通过数值实验验证了我们的理论结论。

BibTeX

```
@article{2512.14686v1,
  title={Bias-Variance Trade-off for Clipped Stochastic First-Order Methods: From Bounded Variance to Infinite Mean},
  author={Chuan He},
  journal={arXiv preprint arXiv:2512.14686v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14686v1}
}
```

## [Professional Software Developers Don't Vibe, They Control: AI Agent Use for Coding in 2025](http://arxiv.org/abs/2512.14012v1)

Ruanqianqian Huang, Avery Reyna, Sorin Lerner, Haijun Xia, Brian Hempel

[PDF](https://arxiv.org/pdf/2512.14012v1)
[Abstract](http://arxiv.org/abs/2512.14012v1)

### 中文摘要

人工智能代理的兴起正在变革软件开发方式。代理的潜力在于，开发者可以更高效地编写代码，将多项任务委托给不同的代理，甚至仅凭自然语言就能构建完整的软件。实际上，代理在专业软件开发中所扮演的角色仍然存在疑问。本文通过实地观察（共13名开发者）和定性调查（共99名开发者），探讨了经验丰富的开发者在软件构建过程中使用代理的动机、策略、任务适配性及其情感态度。研究发现，虽然开发者重视代理带来的生产力提升，但在软件设计与实现中仍坚持维持自主性，以保障软件的基本质量属性，他们会利用专业知识采用控制代理行为的策略。此外，经验丰富的开发者普遍对将代理融入软件开发持积极态度，认为代理可以有效补足其局限性。我们的研究结果揭示了软件开发中的最佳实践在高效使用代理中的价值，提供了代理适用任务的类型参考，并指出了未来在提升代理人界面和使用指南方面的潜在发展方向。

BibTeX

```
@article{2512.14012v1,
  title={Professional Software Developers Don't Vibe, They Control: AI Agent Use for Coding in 2025},
  author={Ruanqianqian Huang and Avery Reyna and Sorin Lerner and Haijun Xia and Brian Hempel},
  journal={arXiv preprint arXiv:2512.14012v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14012v1}
}
```

## [Beyond MMD: Evaluating Graph Generative Models with Geometric Deep Learning](http://arxiv.org/abs/2512.14241v1)

Salvatore Romano, Marco Grassia, Giuseppe Mangioni

[PDF](https://arxiv.org/pdf/2512.14241v1)
[Abstract](http://arxiv.org/abs/2512.14241v1)

### 中文摘要

图生成是众多领域中的关键任务，包括网络科学和生物信息学，因为它能够生成模拟真实世界网络属性的合成图，用于各种应用。图生成模型（GGMs）作为一种有前景的解决方案，利用深度学习技术学习真实图的潜在分布，并生成与之相似的新样本。其中的代表方法包括变分自编码器（VAE）、循环神经网络（RNN）以及近年来的基于扩散的模型。然而，当前的主要限制通常在于评价环节，传统上依赖最大平均差异（MMD）作为衡量生成图在属性分布上的偏差的指标。本文提出了一种新颖的GGMs评价方法，克服了MMD的局限性，命名为基于表示的图生成模型评价（RGM，Representation-aware Graph-generation Model evaluation）。作为该方法的实用演示，我们对两种最新的图生成模型——图循环注意力网络（GRAN）和高效且度导向的图生成模型（EDGE）进行了全面评估，比较它们在生成具有真实属性的图方面的性能，并采用一种基于几何深度学习的模型进行评价，该模型在一个专门为图分类任务设计的合成及真实图数据集上训练。研究结果显示，尽管这两种模型都能生成具有一定拓扑特性的图，但在维持不同图域的结构特征方面存在显著局限。我们还指出了最大平均差异作为GGM评价指标的不足之处，并提出了未来可行的替代评估方案。

BibTeX

```
@article{2512.14241v1,
  title={Beyond MMD: Evaluating Graph Generative Models with Geometric Deep Learning},
  author={Salvatore Romano and Marco Grassia and Giuseppe Mangioni},
  journal={arXiv preprint arXiv:2512.14241v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14241v1}
}
```

## [LAPPI: Interactive Optimization with LLM-Assisted Preference-Based Problem Instantiation](http://arxiv.org/abs/2512.14138v1)

So Kuroki, Manami Nakagawa, Shigeo Yoshida, Yuki Koyama, Kozuno Tadashi

[PDF](https://arxiv.org/pdf/2512.14138v1)
[Abstract](http://arxiv.org/abs/2512.14138v1)

### 中文摘要

许多现实世界的任务，例如旅行规划或膳食安排，可以被建模为组合优化问题。然而，对于终端用户而言，使用优化求解器存在一定的困难，因为这需要进行问题的实例化：包括定义候选项、分配偏好分数以及设定约束条件。我们提出了一种名为LAPPI（基于大规模语言模型的偏好导向问题实例化）的方法，这是一种利用大型语言模型（LLMs）支持用户进行实例化的交互式流程。通过自然语言对话，系统帮助用户将模糊的偏好转化为明确的优化问题。这些实例化的问题随后被传递给现有的优化求解器以生成解决方案。在一项关于旅行规划的用户研究中，我们的方法成功捕捉了用户的偏好，生成了比传统方法和提示工程方法更符合实际的可行方案。此外，我们还展示了LAPPI的多样性，通过将其应用到另一个实际场景中进一步验证了其适应能力。

BibTeX

```
@article{2512.14138v1,
  title={LAPPI: Interactive Optimization with LLM-Assisted Preference-Based Problem Instantiation},
  author={So Kuroki and Manami Nakagawa and Shigeo Yoshida and Yuki Koyama and Kozuno Tadashi},
  journal={arXiv preprint arXiv:2512.14138v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14138v1}
}
```

## [Improvise, Adapt, Overcome -- Telescopic Adapters for Efficient Fine-tuning of Vision Language Models in Medical Imaging](http://arxiv.org/abs/2512.13855v1)

Ujjwal Mishra, Vinita Shukla, Praful Hambarde, Amit Shukla

[PDF](https://arxiv.org/pdf/2512.13855v1)
[Abstract](http://arxiv.org/abs/2512.13855v1)

### 中文摘要

将视觉-语言分割模型（VLSMs）适应医疗影像领域，传统的微调方法通常需要大量计算资源。现有的参数高效微调（PEFT）方法在所有变压器层上采用统一的适配器维度，导致参数分配不够优化，影响适应效果。我们提出了望远镜式适配器（Telescopic Adapters），这是一种新颖的PEFT框架，采用深度感知的缩放策略，逐层增加适配器的容量，从浅层到深层变压器层逐步扩大。该方法在CLIPSeg的视觉和文本编码器中集成了轻量级的瓶颈模块，适配器的维度根据层的深度和语义相关性动态调节。仅使用613千个可训练参数——比端到端微调少244倍，望远镜式适配器在五个涵盖息肉分割、皮肤病变检测和乳腺超声成像的多样化医疗数据集上均表现出优异的性能。全面的消融实验表明，深层特征层对适应容量的需求远高于浅层，这验证了我们的望远镜式缩放策略的合理性。该方法确立了医疗VLSM微调的新的高效范式，能够在资源受限的临床环境中实现部署，同时保持具有竞争力的分割精度。

BibTeX

```
@article{2512.13855v1,
  title={Improvise, Adapt, Overcome -- Telescopic Adapters for Efficient Fine-tuning of Vision Language Models in Medical Imaging},
  author={Ujjwal Mishra and Vinita Shukla and Praful Hambarde and Amit Shukla},
  journal={arXiv preprint arXiv:2512.13855v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13855v1}
}
```

## [A Spatio-Temporal Hybrid Quantum-Classical Graph Convolutional Neural Network Approach for Urban Taxi Destination Prediction](http://arxiv.org/abs/2512.13745v1)

Xiuying Zhang, Qinsheng Zhu, Xiaodong Xing

[PDF](https://arxiv.org/pdf/2512.13745v1)
[Abstract](http://arxiv.org/abs/2512.13745v1)

### 中文摘要

我们提出了一种混合时空量子图卷积网络（H-STQGCN）算法，结合了量子计算与经典深度学习的优势，用于预测城市道路网络中的出租车目的地。该算法由两个分支组成：空间处理与时间演化。在空间处理方面，经典模块基于图卷积网络（GCN）方法编码道路网络的局部拓扑特征，而量子模块则通过可微分池化层将图特征映射到参数化量子电路中。时间演化部分通过整合多源上下文信息，利用经典的时间卷积网络（TCN）理论捕捉动态行程依赖关系。实验结果显示，该算法在预测准确性和稳定性方面优于现有方法，有效验证了量子增强机制在捕获高维空间依赖关系中的独特优势。

BibTeX

```
@article{2512.13745v1,
  title={A Spatio-Temporal Hybrid Quantum-Classical Graph Convolutional Neural Network Approach for Urban Taxi Destination Prediction},
  author={Xiuying Zhang and Qinsheng Zhu and Xiaodong Xing},
  journal={arXiv preprint arXiv:2512.13745v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13745v1}
}
```

## [Composite Classifier-Free Guidance for Multi-Modal Conditioning in Wind Dynamics Super-Resolution](http://arxiv.org/abs/2512.13729v1)

Jacob Schnell, Aditya Makkar, Gunadi Gani, Aniket Srinivasan Ashok, Darren Lo, Mike Optis, Alexander Wong, Yuhao Chen

[PDF](https://arxiv.org/pdf/2512.13729v1)
[Abstract](http://arxiv.org/abs/2512.13729v1)

### 中文摘要

各类气象建模问题（如天气预报、风力涡轮机布局优化等）都需大量高分辨率、高精度的风速数据。然而，获取如此高质量的风速数据仍然是一项具有挑战性且成本高昂的工作。传统的重建方法通常在成本效益和精确度之间难以兼得。近年来，深度学习方法，包括扩散模型的引入，试图利用自然图像超分辨率的技术突破这一权衡。然而，风速数据与自然图像存在显著差异，且风图像超分辨器经常使用超过10个输入通道，远高于自然图像中的3通道RGB输入。为了更有效地利用扩散模型中的多重条件变量，我们提出了一种多条件输入的无分类指导（CFG）方法的推广方案。我们设计的新型复合无分类指导（CCFG）可以嵌入任何使用标准CFG随机删除训练的预训练扩散模型中。实验结果显示，采用CCFG的风速超分辨模型在保真度方面优于传统CFG，输出质量更高。我们还开发了WindDM，一种用于工业级风动力学重建的扩散模型，且结合了CCFG机制。WindDM在深度学习模型中实现了最先进的重建效果，且成本比传统方法低至十倍以上，达到极高的效率。

BibTeX

```
@article{2512.13729v1,
  title={Composite Classifier-Free Guidance for Multi-Modal Conditioning in Wind Dynamics Super-Resolution},
  author={Jacob Schnell and Aditya Makkar and Gunadi Gani and Aniket Srinivasan Ashok and Darren Lo and Mike Optis and Alexander Wong and Yuhao Chen},
  journal={arXiv preprint arXiv:2512.13729v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13729v1}
}
```

## [LoopBench: Discovering Emergent Symmetry Breaking Strategies with LLM Swarms](http://arxiv.org/abs/2512.13713v1)

Ali Parsaee, Yashar Talebirad, Csongor Szepesvári, Vishwajeet Ohal, Eden Redman

[PDF](https://arxiv.org/pdf/2512.13713v1)
[Abstract](http://arxiv.org/abs/2512.13713v1)

### 中文摘要

大型语言模型（LLMs）正日益被用作自主智能体，但其在分布式系统中协调能力的研究仍然有限。我们提出了“LoopBench”这一基准，用于评估LLM在分布式对称性破缺和元认知思维方面的推理能力。该基准主要关注用有限的颜色对奇环图（如$C\_3$、$C\_5$、$C\_{11}$）进行着色的问题，其中存在无沟通的确定性代理会陷入无限循环。我们设计了一种策略通过机制——即一致性记忆，来实现对策略的验证。研究表明，尽管标准LLMs和传统启发式算法难以解决此类问题，但先进的推理模型（如O3）能够制定策略以逃离死锁。LoopBench为基于语言推理的分布式算法的出现提供了研究平台，也为集体智能的探索提供了试验场。

BibTeX

```
@article{2512.13713v1,
  title={LoopBench: Discovering Emergent Symmetry Breaking Strategies with LLM Swarms},
  author={Ali Parsaee and Yashar Talebirad and Csongor Szepesvári and Vishwajeet Ohal and Eden Redman},
  journal={arXiv preprint arXiv:2512.13713v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13713v1}
}
```

## [Adjudicator: Correcting Noisy Labels with a KG-Informed Council of LLM Agents](http://arxiv.org/abs/2512.13704v1)

Doohee You, Sundeep Paul

[PDF](https://arxiv.org/pdf/2512.13704v1)
[Abstract](http://arxiv.org/abs/2512.13704v1)

### 中文摘要

制作机械学习系统的性能在根本上受到训练数据质量的限制。在高风险工业应用中，噪声标签可能降低系统性能并削弱用户信任。本文提出了Adjudicator系统，旨在解决自动识别和修正标签噪声的关键数据挖掘挑战，并已验证其在生产环境中的部署可行性。该系统将该任务建模为一种神经符号混合任务，首先构建一个动态知识图谱（KG）以统一项目信息背景。随后，基于此知识图谱，形成“代理议会”——一种新颖的多代理大规模语言模型架构，专家代理在其中辩论并投票决策标签的有效性。我们在由AlleNoise基准测试的1,000个样本平衡子集上验证了系统性能。结果显示，利用知识图谱信息的模型实现了0.99的F1-score，显著优于单一大模型（0.48）和非知识图谱的议会（0.59）。分析表明，这一优势源于通过一种创新的覆写逻辑实现的高精度，该逻辑利用知识图谱能够完美识别复杂的结构性错误（实现完全召回），而这是基线模型难以发现的错误类型。该结果展示了一个具有鲁棒性且具解释性的自动高精度数据验证系统，为在严格规管的工业环境中生成黄金数据集提供了重要的技术验证。

BibTeX

```
@article{2512.13704v1,
  title={Adjudicator: Correcting Noisy Labels with a KG-Informed Council of LLM Agents},
  author={Doohee You and Sundeep Paul},
  journal={arXiv preprint arXiv:2512.13704v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13704v1}
}
```

## [Towards Nepali-language LLMs: Efficient GPT training with a Nepali BPE tokenizer](http://arxiv.org/abs/2512.14585v1)

Adarsha Shrestha, Basanta Pokharel, Binit Shrestha, Smriti Adhikari, Dinesh Gothe

[PDF](https://arxiv.org/pdf/2512.14585v1)
[Abstract](http://arxiv.org/abs/2512.14585v1)

### 中文摘要

尼泊尔语是一种由超过3200万人使用的低资源语言，由于其复杂的语法结构、黏着语形态以及高质量语料库的有限，可用性一直是自然语言处理（NLP）领域的挑战。迄今为止，大部分研究集中在基础的编码器架构，但仍不足以满足尼泊尔语特定文本生成的需求。本研究提出了一种基于GPT-2的尼泊尔语模型，采用多种受GPT-3启发的训练策略进行训练，包括优化的学习率调度、批处理规模调整和架构优化。此外，研究训练了一个专为尼泊尔文本开发的16k字节对编码（BPE）分词器，以确保分词的一致性并提升输入表示效果。该模型在一个结合了10.75GB清洗过的NepBERTa语料库与额外网络爬取的尼泊尔新闻文章的联合数据集上进行了预训练。为降低内存消耗并稳定训练过程，模型引入了FlashAttention机制。经过两轮训练后，模型达到训练损失3.168177、验证损失3.081982，最终困惑度为21.80，充分展现了其生成连贯的尼泊尔新闻风格文本的能力。

BibTeX

```
@article{2512.14585v1,
  title={Towards Nepali-language LLMs: Efficient GPT training with a Nepali BPE tokenizer},
  author={Adarsha Shrestha and Basanta Pokharel and Binit Shrestha and Smriti Adhikari and Dinesh Gothe},
  journal={arXiv preprint arXiv:2512.14585v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14585v1}
}
```

## [Dynamic Learning Rate Scheduling based on Loss Changes Leads to Faster Convergence](http://arxiv.org/abs/2512.14527v1)

Shreyas Subramanian, Bala Krishnamoorthy, Pranav Murthy

[PDF](https://arxiv.org/pdf/2512.14527v1)
[Abstract](http://arxiv.org/abs/2512.14527v1)

### 中文摘要

尽管在优化器训练方面取得了显著进展，但大多数研究仍采用诸如余弦衰减或指数衰减等常见调度策略。本文提出了一种新颖的调度器——“GreedyLR”，该调度器能够根据当前的损失值在训练过程中自适应调整学习率。为验证所提调度器的有效性，我们在多个自然语言处理（NLP）、计算机视觉（CV）以及具有高达7十亿参数的大型语言模型（LLM）任务上进行了广泛的实验，包括微调和预训练阶段。实验结果显示，我们的方法在准确率、训练速度和收敛性方面均优于多种最先进的调度器。我们还对GreedyLR算法进行了理论分析，包括收敛性证明以及最大化收敛速度的最优缩放因子F的推导，并通过实验证明该算法在具有实际噪声的复杂环境下具有良好的鲁棒性。该调度器实现简便、计算效率高，并可作为训练的优选默认策略。

BibTeX

```
@article{2512.14527v1,
  title={Dynamic Learning Rate Scheduling based on Loss Changes Leads to Faster Convergence},
  author={Shreyas Subramanian and Bala Krishnamoorthy and Pranav Murthy},
  journal={arXiv preprint arXiv:2512.14527v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14527v1}
}
```

## [Sparse Multi-Modal Transformer with Masking for Alzheimer's Disease Classification](http://arxiv.org/abs/2512.14491v1)

Cheng-Han Lu, Pei-Hsuan Tsai

[PDF](https://arxiv.org/pdf/2512.14491v1)
[Abstract](http://arxiv.org/abs/2512.14491v1)

### 中文摘要

基于Transformer的多模态智能系统由于采用密集自注意力机制，通常面临较高的计算和能耗成本，限制了其在资源受限环境下的扩展性。本文提出了一种稀疏多模态Transformer架构——SMMT，旨在提升效率和鲁棒性。在现有级联系统的基础上，SMMT引入了基于簇的稀疏注意力机制，以实现接近线性复杂度的计算，同时采用模态级掩码增强对输入不完整时的鲁棒性。该架构以ADNI数据集上的阿尔茨海默症分类任务作为代表性多模态案例进行验证。实验结果表明，SMMT在保持具有竞争力的预测性能的同时，大幅度降低了训练时间、内存占用和能耗，展现出其作为资源感知型架构组件在大规模智能系统中的应用潜力。

BibTeX

```
@article{2512.14491v1,
  title={Sparse Multi-Modal Transformer with Masking for Alzheimer's Disease Classification},
  author={Cheng-Han Lu and Pei-Hsuan Tsai},
  journal={arXiv preprint arXiv:2512.14491v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14491v1}
}
```

## [TACK Tunnel Data (TTD): A Benchmark Dataset for Deep Learning-Based Defect Detection in Tunnels](http://arxiv.org/abs/2512.14477v1)

Andreas Sjölander, Valeria Belloni, Robel Fekadu, Andrea Nascetti

[PDF](https://arxiv.org/pdf/2512.14477v1)
[Abstract](http://arxiv.org/abs/2512.14477v1)

### 中文摘要

隧道是交通基础设施的重要组成部分，但其逐渐受到裂缝等老化与劣化机制的影响。为了确保安全，必须进行定期检测，但传统的人工检测方法费时、主观性强且成本较高。近年来，移动测绘系统与深度学习（DL）技术的突破，使得自动化视觉检测成为可能。然而，由于隧道数据集的匮乏，相关方法的效果受到限制。本文介绍了一个新的公开数据集，包含对三种不同隧道衬砌的带标注图像，涵盖典型缺陷如裂缝、渗漏和水浸。该数据集旨在支持监督、半监督和无监督的深度学习方法，用于缺陷的检测与分割。其丰富的纹理和施工技术多样性，还便于研究模型在不同隧道类型中的泛化能力和迁移性。通过弥补特定领域数据的关键缺失，该数据集有助于推动隧道自动检测技术的发展，促进更安全、更高效的基础设施维护策略。

BibTeX

```
@article{2512.14477v1,
  title={TACK Tunnel Data (TTD): A Benchmark Dataset for Deep Learning-Based Defect Detection in Tunnels},
  author={Andreas Sjölander and Valeria Belloni and Robel Fekadu and Andrea Nascetti},
  journal={arXiv preprint arXiv:2512.14477v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14477v1}
}
```

## [Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space](http://arxiv.org/abs/2512.14448v1)

Xingfu Zhou, Pengfei Wang

[PDF](https://arxiv.org/pdf/2512.14448v1)
[Abstract](http://arxiv.org/abs/2512.14448v1)

### 中文摘要

依赖于外部检索的大型语言模型（LLM）代理在高风险环境中的应用日益增多。虽然现有的对抗性攻击主要集中在内容伪造或指令注入上，但我们发现了一种新颖的面向过程的攻击切入点：代理的推理风格。我们提出了推理风格中毒（Reasoning-Style Poisoning, RSP）这一范式，旨在操控代理处理信息的方式而非处理内容本身。为此，我们引入了生成风格注入（Generative Style Injection, GSI）方法，该方法通过将检索到的文档重写为具有病理倾向的语气——具体表现为“分析瘫痪”或“认知仓促”——而不改变事实内容或使用明确的触发词。为了量化这些变化，我们设计了推理风格向量（Reasoning Style Vector, RSV），该指标追踪验证深度、自信程度及注意力焦点。在HotpotQA和FEVER数据集上，采用ReAct、Reflection和Tree of Thoughts（ToT）架构的实验显示，GSI显著削弱模型性能，导致推理步骤最多提高4.4倍或引发提前出现的错误，有效规避了现有最先进的内容过滤机制。最后，我们提出了RSP-M，一种轻量级的运行时监控器，能够实时计算RSV指标，并在数值超过安全阈值时发出警报。我们的研究表明，推理风格是一个独立且可被利用的漏洞，强调了超越静态内容分析的过程级防御的重要性。

BibTeX

```
@article{2512.14448v1,
  title={Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space},
  author={Xingfu Zhou and Pengfei Wang},
  journal={arXiv preprint arXiv:2512.14448v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14448v1}
}
```

## [Seismology modeling agent: A smart assistant for geophysical researchers](http://arxiv.org/abs/2512.14429v1)

Yukun Ren, Siwei Yu, Kai Chen, Jianwei Ma

[PDF](https://arxiv.org/pdf/2512.14429v1)
[Abstract](http://arxiv.org/abs/2512.14429v1)

### 中文摘要

为了解决传统主流开源地震波模拟软件SPECFEM在使用过程中存在的陡峭学习曲线，以及对复杂手动文件编辑和命令行操作的依赖，本文提出了一种由大型语言模型（LLMs）驱动的智能交互式工作流。我们推出了首个支持二维、三维笛卡尔坐标系以及三维全球模型的SPECFEM模型上下文协议（MCP）服务器套件，该套件将整个模拟流程划分为多个可由代理执行的离散工具，包括参数生成、网格划分、求解器运行及结果可视化。这一方法实现了从基于文件的操作向以意图为导向的对话交互的范式转变。该框架既支持全自动化执行，也支持人工干预协作，研究人员可以在实时引导模拟策略的同时，保持科学决策的自主权，有效减少繁琐的底层操作。通过多个案例验证，工作流在自动化和交互式两种模式下均能无缝运行，生成的高保真结果与标准基线保持一致。作为MCP技术首次应用于计算地震学的实例，该研究显著降低了入门门槛，增强了结果的可重复性，为推动计算地球物理学迈向AI辅助和自动化科研提供了有前景的途径。完整源代码可在 https://github.com/RenYukun1563/specfem-mcp 获取。

BibTeX

```
@article{2512.14429v1,
  title={Seismology modeling agent: A smart assistant for geophysical researchers},
  author={Yukun Ren and Siwei Yu and Kai Chen and Jianwei Ma},
  journal={arXiv preprint arXiv:2512.14429v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14429v1}
}
```

## [MultiBanAbs: A Comprehensive Multi-Domain Bangla Abstractive Text Summarization Dataset](http://arxiv.org/abs/2511.19317v1)

Md. Tanzim Ferdous, Naeem Ahsan Chowdhury, Prithwiraj Bhattacharjee

[PDF](https://arxiv.org/pdf/2511.19317v1)
[Abstract](http://arxiv.org/abs/2511.19317v1)

### 中文摘要

本研究开发了一套新的孟加拉语抽象式摘要数据集，旨在从不同来源的孟加拉语文章中生成简洁的摘要。现有大部分相关研究主要集中在新闻类文章，记者通常遵循固定的写作风格。虽然此类方法在有限的场景下效果良好，但往往难以适应现实中孟加拉语文本的多样性。在当今数字时代，大量孟加拉语内容持续在博客、报纸和社交媒体上生成，亟需一种能够减轻信息过载、帮助读者更快理解内容的摘要系统。为应对这一挑战，我们收集了包括Cinegolpo等博客以及Samakal和The Business Standard等报纸在内的多个来源的超过54,000篇孟加拉语文章及其摘要，构建了该数据集。与单一领域资源不同，该数据集涵盖多个领域和写作风格，具有更强的适应性和实际应用价值。为了建立坚实的基线，我们采用多种深度学习和迁移学习模型进行训练与评估，包括LSTM、BanglaT5-small和MTS-small。结果显示，该数据集具有作为孟加拉语自然语言处理研究基准的潜力，为构建稳健的摘要系统提供了坚实基础，也有助于扩展低资源语言的自然语言处理资源。

BibTeX

```
@article{2511.19317v1,
  title={MultiBanAbs: A Comprehensive Multi-Domain Bangla Abstractive Text Summarization Dataset},
  author={Md. Tanzim Ferdous and Naeem Ahsan Chowdhury and Prithwiraj Bhattacharjee},
  journal={arXiv preprint arXiv:2511.19317v1},
  year={2025},
  url={http://arxiv.org/abs/2511.19317v1}
}
```

## [A Threshold-Triggered Deep Q-Network-Based Framework for Self-Healing in Autonomic Software-Defined IIoT-Edge Networks](http://arxiv.org/abs/2512.14297v1)

Agrippina Mwangi, León Navarro-Hilfiker, Lukasz Brewka, Mikkel Gryning, Elena Fumagalli, Madeleine Gibescu

[PDF](https://arxiv.org/pdf/2512.14297v1)
[Abstract](http://arxiv.org/abs/2512.14297v1)

### 中文摘要

本文研究了 benign流量突发和交换机热噪声等随机中断在软件定义工业网络中导致的间歇性服务下降问题。这些事件违反了基于IEC 61850标准的服务质量要求和用户定义的服务水平协议，影响了符合IEC 61400-25标准的风力发电厂中控制、监测以及尽力而为的流量的可靠且及时的传输。未能满足这些要求往往会导致控制信号的延迟或丢失，降低运行效率，并增加风力涡轮发电机停机的风险。
为应对这些挑战，本文提出一种基于阈值触发的深度Q-网络（Deep Q-Network）自愈代理，能够自主检测、分析并缓解网络中断，同时实时调整路由策略和资源分配。所提出的代理在一个云端仿真三簇交换机网络中进行训练、验证和测试，验证环境为一种概念验证的试验台。
仿真结果显示，该代理在中断恢复性能方面比基线的最短路径和负载均衡路由方案提高了53.84%，在超-脊状叶片数据平面架构中，优于包括自适应网络模糊推理系统（Adaptive Neural-fuzzy Inference System）在内的多种先进方法，提升比例分别达到13.1%和21.5%。此外，代理还通过主动启动外部机架冷却措施维护交换机的热稳定性。
研究结果显示，深度强化学习在打造针对关键任务、时间敏感型应用场景的韧性工业网络方面具有巨大潜力。

BibTeX

```
@article{2512.14297v1,
  title={A Threshold-Triggered Deep Q-Network-Based Framework for Self-Healing in Autonomic Software-Defined IIoT-Edge Networks},
  author={Agrippina Mwangi and León Navarro-Hilfiker and Lukasz Brewka and Mikkel Gryning and Elena Fumagalli and Madeleine Gibescu},
  journal={arXiv preprint arXiv:2512.14297v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14297v1}
}
```

## [Gödel's Poetry](http://arxiv.org/abs/2512.14252v1)

Kelly J. Davis

[PDF](https://arxiv.org/pdf/2512.14252v1)
[Abstract](http://arxiv.org/abs/2512.14252v1)

### 中文摘要

形式化自动定理证明长期以来一直被视为人工智能的挑战之一。我们在此提出一种新的计算机定理证明方法，该方法结合了专门的语言模型用于Lean4的证明生成，以及对难题定理进行递归分解为更简单蕴含命题的策略。这些模型通过多智能体架构协同工作，统筹自动形式化（如有必要）、证明生成、难题定理的分解以及这些命题的递归证明（或再分解）。在未进行分解的情况下，我们在miniF2F数据集上达到了90.4%的通关率，而引入分解策略后，性能明显提升。本方法的一个关键技术贡献在于，我们扩展了Kimina Lean服务器，添加了抽象语法树（AST）解析能力，以支持自动化的递归证明分解。相关系统已作为goedels-poetry包在PyPI上发布（网址：https://pypi.org/project/goedels-poetry ），开源实现位于KellyJDavis/goedels-poetry（网址：https://github.com/KellyJDavis/goedels-poetry ），便于适配不同的语言模型及扩展定制功能。

BibTeX

```
@article{2512.14252v1,
  title={Gödel's Poetry},
  author={Kelly J. Davis},
  journal={arXiv preprint arXiv:2512.14252v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14252v1}
}
```

## [IntentMiner: Intent Inversion Attack via Tool Call Analysis in the Model Context Protocol](http://arxiv.org/abs/2512.14166v1)

Yunhao Yao, Zhiqiang Wang, Haoran Cheng, Yihang Cheng, Haohua Du, Xiang-Yang Li

[PDF](https://arxiv.org/pdf/2512.14166v1)
[Abstract](http://arxiv.org/abs/2512.14166v1)

### 中文摘要

大规模语言模型（LLMs）逐步演变为自主代理的快速发展，促使“模型上下文协议”（MCP）成为发现和调用外部工具的标准框架。尽管该架构通过将推理引擎与工具执行解耦以增强可扩展性，但也带来了显著的隐私风险：作为半诚实中介的第三方MCP服务器可以在用户可信边界之外，观察到详细的工具交互日志。本文首先识别并形式化了一种新颖的隐私威胁——意图反转（Intent Inversion），该威胁指半诚实的MCP服务器仅通过分析合法的工具调用，即试图重建用户的隐藏私有意图。为系统性评估这一漏洞，我们提出了IntentMiner框架，该框架结合层级信息隔离与三维语义分析技术，融合工具用途、调用语句及返回结果，能够在步骤层面准确推断用户意图。大量实验结果显示，IntentMiner在语义匹配度方面（超过85%）显著优于基线方法，彰显了这种架构中潜在的隐私风险，揭示了看似无害的工具调用日志实际上可能成为泄露用户秘密的有力载体。

BibTeX

```
@article{2512.14166v1,
  title={IntentMiner: Intent Inversion Attack via Tool Call Analysis in the Model Context Protocol},
  author={Yunhao Yao and Zhiqiang Wang and Haoran Cheng and Yihang Cheng and Haohua Du and Xiang-Yang Li},
  journal={arXiv preprint arXiv:2512.14166v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14166v1}
}
```

## [Universal Reasoning Model](http://arxiv.org/abs/2512.14693v1)

Zitian Gao, Lynx Chen, Yihao Xiao, He Xing, Ran Tao, Haoming Luo, Joey Zhou, Bryan Dai

[PDF](https://arxiv.org/pdf/2512.14693v1)
[Abstract](http://arxiv.org/abs/2512.14693v1)

### 中文摘要

通用变换器（UTs）已广泛应用于复杂推理任务，如ARC-AGI和数独，但其性能提升的具体源头仍未得到充分研究。在本工作中，我们系统分析了UT的各种变体，并发现对ARC-AGI性能的提升主要源自循环归纳偏差和变换器的强非线性成分，而非复杂的架构设计。基于这一观察，我们提出了通用推理模型（URM），通过引入短卷积和截断的反向传播技术对UT进行增强。该方法显著提升了推理性能，在ARC-AGI 1任务中实现了53.8%的pass@1（当前最优水平），在ARC-AGI 2任务中达到16.0%的pass@1。我们的代码可在https://github.com/zitian-gao/URM获得。

BibTeX

```
@article{2512.14693v1,
  title={Universal Reasoning Model},
  author={Zitian Gao and Lynx Chen and Yihao Xiao and He Xing and Ran Tao and Haoming Luo and Joey Zhou and Bryan Dai},
  journal={arXiv preprint arXiv:2512.14693v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14693v1}
}
```

## [Native and Compact Structured Latents for 3D Generation](http://arxiv.org/abs/2512.14692v1)

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, Jiaolong Yang

[PDF](https://arxiv.org/pdf/2512.14692v1)
[Abstract](http://arxiv.org/abs/2512.14692v1)

### 中文摘要

近年来，三维生成模型在逼真度方面取得了显著提升，但由于现有表示方法难以有效捕捉具有复杂拓扑结构和细节外观的资产，仍存在一定的局限性。本文提出一种从原生三维数据中学习结构化潜在表示的方法，旨在解决这一挑战。其核心是一种全新的稀疏体素结构，称为O-Voxel（全方位体素），这是一种同时编码几何形状和外观的全景体素表示。O-Voxel能够稳健地建模任意拓扑，包括开放、非流形以及完全封闭的表面，并且不仅捕捉纹理色彩，还包括物理基础渲染参数等全面的表面属性。在此基础上，我们设计了一种稀疏压缩变分自编码器（Sparse Compression VAE），实现了高空间压缩率和紧凑的潜在空间。我们利用多样化的公共三维资产数据集，训练了包含40亿参数的大规模流动匹配模型用于三维生成。尽管模型规模庞大，推理过程依然高效。同时，我们生成的资产在几何和材质质量方面远超现有模型。我们相信，该方法在三维生成建模领域具有重要的突破意义。

BibTeX

```
@article{2512.14692v1,
  title={Native and Compact Structured Latents for 3D Generation},
  author={Jianfeng Xiang and Xiaoxue Chen and Sicheng Xu and Ruicheng Wang and Zelong Lv and Yu Deng and Hongyuan Zhu and Yue Dong and Hao Zhao and Nicholas Jing Yuan and Jiaolong Yang},
  journal={arXiv preprint arXiv:2512.14692v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14692v1}
}
```

## [Graph AI generates neurological hypotheses validated in molecular, organoid, and clinical systems](http://arxiv.org/abs/2512.13724v1)

Ayush Noori, Joaquín Polonuer, Katharina Meyer, Bogdan Budnik, Shad Morton, Xinyuan Wang, Sumaiya Nazeen, Yingnan He, Iñaki Arango, Lucas Vittor, Matthew Woodworth, Richard C. Krolewski, Michelle M. Li, Ninning Liu, Tushar Kamath, Evan Macosko, Dylan Ritter, Jalwa Afroz, Alexander B. H. Henderson, Lorenz Studer, Samuel G. Rodriques, Andrew White, Noa Dagan, David A. Clifton, George M. Church, Sudeshna Das, Jenny M. Tam, Vikram Khurana, Marinka Zitnik

[PDF](https://arxiv.org/pdf/2512.13724v1)
[Abstract](http://arxiv.org/abs/2512.13724v1)

### 中文摘要

神经系统疾病是全球导致残疾的主要原因，但大多数尚缺乏具有疾病修饰作用的治疗方法。我们提出了PROTON，一种异构图变换模型，能够在分子、类器官和临床系统之间生成可检验的假设。为了评估PROTON的性能，我们将其应用于帕金森病（PD）、双相情感障碍（BD）和阿尔茨海默病（AD）。在PD中，PROTON将遗传风险位点与维持多巴胺能神经元存活所必需的基因关联起来，并预测对患者衍生神经元具有毒性的农药，包括排名前1.29%的杀虫剂敌敌畏。PROTON进行的计算机筛选重现了六项全基因组范围的$\alpha$-突触核蛋白相关实验，如拆分泛素酶酵母两杂系统（标准富集分数[NES] = 2.30，FDR校正$p < 1 \times 10^{-4}$）、抗坏血酸过氧化物酶邻域标记（NES = 2.16，FDR < 1 \times 10^{-4}$），以及在496例突触核蛋白沉积症患者中进行的高深度靶向外显子测序研究（NES = 2.13，FDR < 1 \times 10^{-4}$）。在BD方面，PROTON预测降钙素以一种候选药物，可以逆转来源于BD患者皮层类器官的蛋白质组变化。在AD中，我们在麻省总医院610,524名患者的健康记录中验证了PROTON的预测，发现五种PROTON预测的药物与七年认知障碍风险的降低相关（最小风险比=0.63，95%置信区间：0.53–0.75，$p < 1 \times 10^{-7}$）。PROTON在分子、类器官和临床系统中生成神经疾病的假设，开辟了基于人工智能的神经疾病发现的新路径。

BibTeX

```
@article{2512.13724v1,
  title={Graph AI generates neurological hypotheses validated in molecular, organoid, and clinical systems},
  author={Ayush Noori and Joaquín Polonuer and Katharina Meyer and Bogdan Budnik and Shad Morton and Xinyuan Wang and Sumaiya Nazeen and Yingnan He and Iñaki Arango and Lucas Vittor and Matthew Woodworth and Richard C. Krolewski and Michelle M. Li and Ninning Liu and Tushar Kamath and Evan Macosko and Dylan Ritter and Jalwa Afroz and Alexander B. H. Henderson and Lorenz Studer and Samuel G. Rodriques and Andrew White and Noa Dagan and David A. Clifton and George M. Church and Sudeshna Das and Jenny M. Tam and Vikram Khurana and Marinka Zitnik},
  journal={arXiv preprint arXiv:2512.13724v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13724v1}
}
```

## [Incentivizing Tool-augmented Thinking with Images for Medical Image Analysis](http://arxiv.org/abs/2512.14157v1)

Yankai Jiang, Yujie Zhang, Peng Zhang, Yichen Li, Jintai Chen, Xiaoming Shi, Shihui Zhen

[PDF](https://arxiv.org/pdf/2512.14157v1)
[Abstract](http://arxiv.org/abs/2512.14157v1)

### 中文摘要

近年来，基于推理的医疗多模态大型语言模型（MLLMs）在逐步生成文本推理链方面取得了显著进展。然而，它们在处理复杂任务时仍面临挑战，尤其是在需要动态、迭代地关注细粒度视觉区域以实现精确定位和诊断的场景中。为此，本文提出了“Ophiuchus”框架，这是一种多功能、工具增强的模型架构，能够赋予MLLM以下能力：(i) 判断何时需要额外的视觉证据；(ii) 确定在医学图像中应关注和定位的区域；(iii) 无缝将相关子图像内容融入到交错的多模态推理链中。与受限于专用工具性能上限的先前方法不同，Ophiuchus将模型固有的定位和感知能力与外部工具相结合，从而促进更高层次的推理能力。该方法的核心是一个三阶段的训练策略：首先，通过结合工具的推理数据进行冷启动训练，以实现基础的工具选择和关键区域检测；然后，通过自我反思微调，强化模型的反思推理能力，并鼓励其多次复查工具输出；最后，采用代理工具强化学习，直接优化任务相关的奖励，模仿专家级诊断行为。大量实验结果表明，Ophiuchus在医学领域的多个基准任务中（包括视觉问答、检测和基于推理的分割）均优于现有的最佳方法（无论是闭源还是开源）。这一方法为实现真正“用图像思考”的医学AI代理提供了新的可能性。相关数据集、代码和训练模型将公开发布。

BibTeX

```
@article{2512.14157v1,
  title={Incentivizing Tool-augmented Thinking with Images for Medical Image Analysis},
  author={Yankai Jiang and Yujie Zhang and Peng Zhang and Yichen Li and Jintai Chen and Xiaoming Shi and Shihui Zhen},
  journal={arXiv preprint arXiv:2512.14157v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14157v1}
}
```

## [PathFinder: Advancing Path Loss Prediction for Single-to-Multi-Transmitter Scenario](http://arxiv.org/abs/2512.14150v1)

Zhijie Zhong, Zhiwen Yu, Pengyu Li, Jianming Lv, C. L. Philip Chen, Min Chen

[PDF](https://arxiv.org/pdf/2512.14150v1)
[Abstract](http://arxiv.org/abs/2512.14150v1)

### 中文摘要

无线路径损耗预测（RPP）在优化5G网络以及实现物联网、智能城市等应用中具有关键作用。然而，现有的基于深度学习的路径损耗预测方法普遍缺乏对环境因素的主动建模，在应对现实中的多发射源场景时表现不佳，且在训练和测试环境存在差异（如建筑密度或发射器配置不同）时，其泛化能力明显不足。本文指出了三个核心问题：（1）被动式环境建模未考虑发射源及关键环境特征；（2）过度关注单发射源场景，忽视了真实场景中多发射源的普遍存在；（3）只注重训练环境内的性能表现，而忽略因分布偏移带来的挑战。为此，我们提出了PathFinder，一种通过解耦特征编码主动建模建筑和发射源的创新架构，结合Mask引导的低秩注意机制，能够独立聚焦于接收区域和建筑结构。此外，本文还引入面向发射源的Mixup策略，增强模型的鲁棒性，并提出了专为评估跨分布泛化能力设计的单发射源到多发射源路径损耗预测基准（S2MT-RPP），实现“从单发射源训练到多发射源测试”的考核。实验结果显示，PathFinder在多个多发射源场景下显著优于现有最优方法，展示出强大的预测性能与泛化能力。我们的代码和项目主页可在https://emorzz1g.github.io/PathFinder/获取。

BibTeX

```
@article{2512.14150v1,
  title={PathFinder: Advancing Path Loss Prediction for Single-to-Multi-Transmitter Scenario},
  author={Zhijie Zhong and Zhiwen Yu and Pengyu Li and Jianming Lv and C. L. Philip Chen and Min Chen},
  journal={arXiv preprint arXiv:2512.14150v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14150v1}
}
```

## [UIXPOSE: Mobile Malware Detection via Intention-Behaviour Discrepancy Analysis](http://arxiv.org/abs/2512.14130v1)

Amirmohammad Pasdar, Toby Murray, Van-Thuan Pham

[PDF](https://arxiv.org/pdf/2512.14130v1)
[Abstract](http://arxiv.org/abs/2512.14130v1)

### 中文摘要

我们提出UIXPOSE，一种与源代码无关的框架，既适用于已编译应用也适用于开源应用。该框架在移动恶意软件分析中应用意图行为对齐（IBA），将UI推断的意图与运行时语义相结合。以往的研究要么利用静态方法推断意图，例如以权限为核心，要么在界面控件层面分析，要么监测较为粗粒度的动态信号（如端点、部分资源使用），但都容易遗漏内容和上下文信息。UIXPOSE通过视觉-语言模型和知识结构，从每个屏幕中推断出一个意图向量，并结合解码的网络负载、堆/内存信号以及资源利用轨迹，形成一个行为向量。在运行时对其进行对齐，不仅能够检测出异常行为，还能揭示丰富行为路径的探索。在三个真实案例中，UIXPOSE成功揭示了隐蔽的隐秘数据窃取和隐藏的后台活动，超越了仅依赖元数据的基线方法，展示了IBA在动态检测中的提升作用。

BibTeX

```
@article{2512.14130v1,
  title={UIXPOSE: Mobile Malware Detection via Intention-Behaviour Discrepancy Analysis},
  author={Amirmohammad Pasdar and Toby Murray and Van-Thuan Pham},
  journal={arXiv preprint arXiv:2512.14130v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14130v1}
}
```

## [ACE-SLAM: Scene Coordinate Regression for Neural Implicit Real-Time SLAM](http://arxiv.org/abs/2512.14032v1)

Ignacio Alzugaray, Marwan Taher, Andrew J. Davison

[PDF](https://arxiv.org/pdf/2512.14032v1)
[Abstract](http://arxiv.org/abs/2512.14032v1)

### 中文摘要

我们提出了一种新颖的神经RGB-D同时定位与地图构建（SLAM）系统，能够实时学习场景的隐式地图。首次在神经SLAM流程中探索以场景坐标回归（SCR）作为核心隐式地图表示的方法，这一范式通过训练轻量级网络，直接将二维图像特征映射到三维全局坐标。SCR网络提供高效、低存储的三维地图表示，能够实现极快的重定位，并且在隐私保护方面具有天然优势，使其特别适合用于神经隐式SLAM。
我们的系统是首次依靠SCR表示实现严格实时的神经隐式RGB-D SLAM。我们引入了一种专为此目的设计的创新SCR结构，详细阐述了将SCR集成到实时SLAM流程中的关键设计选择。该框架结构简洁而具有高度的灵活性，兼容稀疏与稠密特征，并能在动态环境中稳定运行，无需特别的适应措施。我们在既有的合成及真实场景基准数据上对所提出的方法进行了评估，结果显示其性能与当前先进技术相当。项目页面： https://github.com/ialzugaray/ace-slam

BibTeX

```
@article{2512.14032v1,
  title={ACE-SLAM: Scene Coordinate Regression for Neural Implicit Real-Time SLAM},
  author={Ignacio Alzugaray and Marwan Taher and Andrew J. Davison},
  journal={arXiv preprint arXiv:2512.14032v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14032v1}
}
```

## [Sample-Efficient Robot Skill Learning for Construction Tasks: Benchmarking Hierarchical Reinforcement Learning and Vision-Language-Action VLA Model](http://arxiv.org/abs/2512.14031v1)

Zhaofeng Hu, Hongrui Yu, Vaidhyanathan Chandramouli, Ci-Jyun Liang

[PDF](https://arxiv.org/pdf/2512.14031v1)
[Abstract](http://arxiv.org/abs/2512.14031v1)

### 中文摘要

本研究评估了两种在建筑机器人新技能传授方面的领先方法，以探讨其在建筑自动化中的适用性：一种是视觉-语言-动作（VLA）模型，另一种是强化学习（RL）方法。研究旨在理解这两种方法在任务性能方面的表现，以及在实际工程中部署所需的实际工作量。作者开发了两种遥操作界面，用于控制机器人并收集示范数据，这两种界面都在训练执行长距离和精细操作任务的机器人中证明了其有效性。此外，作者进行了三阶段的评估。首先，将多层感知器（MLP）策略与深度Q网络（DQN）模仿模型进行对比，以确定更优的RL基线，重点考察模型性能、泛化能力以及抓取实验。其次，在两个不同场景中训练并对比了三种不同的VLA模型。最后，以计算效率和样本效率指标对所选的RL基线与VLA模型进行了性能 benchmark，并进行了多阶段面板装配任务的机器人实验，包括运输和安装。结果显示，VLA模型具有较强的泛化能力和少样本学习能力，在抓取阶段成功率达到60%和100%。相比之下，DQN模型可以通过增加噪声实现鲁棒性，但在调优过程中需要额外的工作量，从而增加了操作难度。总体而言，研究结论表明，VLA方法在面对任务变更时具有明显的实际优势，能够减少编程工作量，利用少量数据实现较好的性能；而DQN则在调优充分的情况下提供了一个具有可行性的基线方案。

BibTeX

```
@article{2512.14031v1,
  title={Sample-Efficient Robot Skill Learning for Construction Tasks: Benchmarking Hierarchical Reinforcement Learning and Vision-Language-Action VLA Model},
  author={Zhaofeng Hu and Hongrui Yu and Vaidhyanathan Chandramouli and Ci-Jyun Liang},
  journal={arXiv preprint arXiv:2512.14031v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14031v1}
}
```

## [Multi-Agent Collaborative Framework for Intelligent IT Operations: An AOI System with Context-Aware Compression and Dynamic Task Scheduling](http://arxiv.org/abs/2512.13956v1)

Zishan Bai, Enze Ge, Junfeng Hao

[PDF](https://arxiv.org/pdf/2512.13956v1)
[Abstract](http://arxiv.org/abs/2512.13956v1)

### 中文摘要

云原生架构的普及，以微服务和动态编排为特征，导致现代信息技术基础设施变得极其复杂和易变。这种复杂性产生了海量的运营数据，造成传统系统中的关键瓶颈，包括信息处理效率低下、任务协调困难以及在故障诊断和修复过程中缺乏上下文连续性。为应对这些挑战，本文提出了一种新颖的多智能体协作框架——AI导向运营（AOI），该框架结合了三个专业化智能体与基于大规模语言模型的上下文压缩器。其核心创新包括：(1) 基于实时系统状态动态调度任务的策略，能自适应地优先处理重要操作；(2) 由工作层、事件层和语义层组成的三层记忆架构，优化上下文的保持与检索。大量在合成和实际场景基准测试中的实验结果表明，AOI有效缓解了信息过载，实现了72.4%的上下文压缩比，同时保留了92.8%的关键信息，并显著提升了运营效率，任务成功率达到94.2%，比最佳基线方案将平均修复时间（MTTR）缩短了34.4%。本研究实现了一种面向可扩展、具有自适应能力和上下文感知的自主运营范式，推动下一代IT基础设施的稳健管理，极大减少了对人工干预的依赖。

BibTeX

```
@article{2512.13956v1,
  title={Multi-Agent Collaborative Framework for Intelligent IT Operations: An AOI System with Context-Aware Compression and Dynamic Task Scheduling},
  author={Zishan Bai and Enze Ge and Junfeng Hao},
  journal={arXiv preprint arXiv:2512.13956v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13956v1}
}
```

## [Intelligent matter consisting of active particles](http://arxiv.org/abs/2512.13912v1)

Julian Jeggle, Raphael Wittkowski

[PDF](https://arxiv.org/pdf/2512.13912v1)
[Abstract](http://arxiv.org/abs/2512.13912v1)

### 中文摘要

在本书章节中，我们回顾了由简单的运动代理系统构建智能系统的可能路径。自然界的一个著名现象是，大量个体遵循简单规则（如动物的群体行为）可以产生更为复杂的集体行为，这种现象体现了涌现效应。这引发了一个问题：我们是否可以在合成物质中模拟这种行为，并将其驱动到集体行为达到智能系统的复杂程度。在此，我们将采用“智能物质”的形式化定义，并将其与活性物质领域中的最新研究结果进行比较。首先，我们将探讨涌现计算的方法，即设计专门的活性物质系统，通过涌现行为直接解决特定任务。随后，我们将对比基于活性粒子系统动力学的物理储存计算方法。在这一背景下，我们还将介绍一种新颖的活性粒子储存计算方案，该方案通过超声驱动或光折射实现对粒子的操控。

BibTeX

```
@article{2512.13912v1,
  title={Intelligent matter consisting of active particles},
  author={Julian Jeggle and Raphael Wittkowski},
  journal={arXiv preprint arXiv:2512.13912v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13912v1}
}
```

## [EEG-D3: A Solution to the Hidden Overfitting Problem of Deep Learning Models](http://arxiv.org/abs/2512.13806v1)

Siegfried Ludwig, Stylianos Bakas, Konstantinos Barmpas, Georgios Zoumpourlis, Dimitrios A. Adamos, Nikolaos Laskaris, Yannis Panagakis, Stefanos Zafeiriou

[PDF](https://arxiv.org/pdf/2512.13806v1)
[Abstract](http://arxiv.org/abs/2512.13806v1)

### 中文摘要

深度学习在解码脑电信号（EEG）方面已获得广泛关注，并多次声称达到了最新的性能水平。然而，尽管在基准测试中的表现令人信服，实际应用中的转化效果却有限。控制式脑机接口（BCI）基准性能与其在实际场景中的泛化能力之间经常存在脱节，揭示了潜在的过拟合问题。我们提出了解缠解码分解（Disentangled Decoding Decomposition，D3），一种弱监督方法，用于在多个EEG数据集上训练深度学习模型。通过预测输入窗口采样的具体试验序列位置，EEG-D3实现了类似非线性独立成分分析（ICA）的脑活动潜在成分的分离。我们采用了一种具有完全独立子网络的创新模型结构，以实现严格的可解释性。并提出了一种特征解释范式，用于对比不同数据集上的成分激活特征曲线，并检查相关的时空滤波器。该方法在运动意象数据上有效地分离了脑活动的潜在成分。在此基础上，将下游分类器训练在这些成分的适当子集上，可以防止由任务相关伪迹引发的隐性过拟合，这种过拟合严重影响端到端的分类性能。此外，我们还利用线性可分的潜在空间实现了睡眠阶段分类的少样本学习。能够区分真实脑活动成分与虚假特征的能力，使模型避免隐藏的过拟合问题，并能很好地泛化到实际应用中，同时仅需极少的有标签数据。该方法对于神经科学界具有重要意义，为研究者提供了一种分离个体脑过程，甚至潜在揭示未知动态的有力工具。

BibTeX

```
@article{2512.13806v1,
  title={EEG-D3: A Solution to the Hidden Overfitting Problem of Deep Learning Models},
  author={Siegfried Ludwig and Stylianos Bakas and Konstantinos Barmpas and Georgios Zoumpourlis and Dimitrios A. Adamos and Nikolaos Laskaris and Yannis Panagakis and Stefanos Zafeiriou},
  journal={arXiv preprint arXiv:2512.13806v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13806v1}
}
```

## [TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](http://arxiv.org/abs/2512.14698v1)

Jun Zhang, Teng Wang, Yuying Ge, Yixiao Ge, Xinhao Li, Ying Shan, Limin Wang

[PDF](https://arxiv.org/pdf/2512.14698v1)
[Abstract](http://arxiv.org/abs/2512.14698v1)

### 中文摘要

本文并未提出新颖的方法，而是为视频时间定位（VTG）这一视频理解的核心能力，建立了一个简洁、逐步但至关重要的基本基线。尽管多模态大语言模型（MLLMs）在多种视频理解任务中表现出色，但针对其在VTG任务中的优化策略仍然研究不足。本文提出了TimeLens，一项系统性研究，旨在构建具有强大VTG能力的MLLMs，主要从数据质量和算法设计两个维度展开。我们首先揭示了现有VTG基准测试中的关键质量问题，并引入TimeLens-Bench，该基准由经过严格质量标准重新标注的三大流行基准组成。分析显示，与传统基准相比，模型的排名发生了巨大变化，验证了之前评估标准的不可靠性。我们还通过自动化重标注流程解决了数据中的噪声问题，产生了规模庞大、质量优良的训练数据集TimeLens-100K。在此基础上，我们深入探索了算法设计原则，获得了一系列具有指导意义的见解和高效有效的实践方法，包括采用交替的文本编码进行时间表示、引入无需思考的可验证奖励（RLVR）作为训练范式，以及为RLVR训练设计的细致策略。最终，这些努力孕育出了TimeLens系列模型，在开源模型中达到最先进的VTG性能，甚至超越了如GPT-5和Gemini-2.5-Flash等专有模型。所有代码、数据和模型将对外公开，以推动未来相关研究的发展。

BibTeX

```
@article{2512.14698v1,
  title={TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs},
  author={Jun Zhang and Teng Wang and Yuying Ge and Yixiao Ge and Xinhao Li and Ying Shan and Limin Wang},
  journal={arXiv preprint arXiv:2512.14698v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14698v1}
}
```

## [Assessing High-Risk Systems: An EU AI Act Verification Framework](http://arxiv.org/abs/2512.13907v1)

Alessio Buscemi, Tom Deckenbrunnen, Fahria Kabir, Nishat Mowla, Kateryna Mishchenko

[PDF](https://arxiv.org/pdf/2512.13907v1)
[Abstract](http://arxiv.org/abs/2512.13907v1)

### 中文摘要

在欧盟实施《人工智能法》（AI法）及其他相关人工智能法规时，一个核心挑战是缺乏系统性的方法来验证其法律授权的遵从性。最新调研显示，这种监管模糊性被视为一项重大负担，导致各成员国在法规遵循方面存在不一致的准备程度。本文提出了一个全面的框架，旨在弥补这一空白，通过在两个基本维度上组织合规性验证：方法类型（控制与测试）和评估对象（数据、模型、流程及最终产品）。此外，我们的框架将核心法律要求映射到具体的验证活动，作为政策制定者与实践者之间的重要桥梁，帮助将法律文本与技术标准及最佳实践对接。所提出的方法旨在减少解释上的不确定性，促进评估实践的一致性，并支持在AI生命周期中实现监管、伦理与技术视角的协调统一。

BibTeX

```
@article{2512.13907v1,
  title={Assessing High-Risk Systems: An EU AI Act Verification Framework},
  author={Alessio Buscemi and Tom Deckenbrunnen and Fahria Kabir and Nishat Mowla and Kateryna Mishchenko},
  journal={arXiv preprint arXiv:2512.13907v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13907v1}
}
```

## [One Permutation Is All You Need: Fast, Reliable Variable Importance and Model Stress-Testing](http://arxiv.org/abs/2512.13892v1)

Albert Dorador

[PDF](https://arxiv.org/pdf/2512.13892v1)
[Abstract](http://arxiv.org/abs/2512.13892v1)

### 中文摘要

在机器学习模型中可靠地估计特征贡献对于模型的信任、透明度和合规性至关重要，尤其是在模型为专有或黑箱操作的情况下。虽然基于置换的方法是这一任务的标准工具，但传统实现依赖多次随机置换，导致计算开销大且结果具有随机不稳定性。本文提出，将多次随机置换替换为单一的、确定性且最优的置换，从而实现一种既保留置换特征重要性核心思想，又摆脱随机性、运行速度更快、结果更稳定的方法。我们在近200个不同场景中验证了该方法的有效性，包括实际应用中的家庭财务和信用风险场景，证明其在样本量较小、高特征维数以及信噪比低等复杂环境下都能显著改善偏差-方差权衡与预测精度。最后，我们引入系统性变量重要性（Systemic Variable Importance），这是该方法的自然扩展，专为模型压力测试设计，能够明确考虑特征之间的相关性。这个框架提供了一种透明的方式，量化冲击或扰动在相关输入中的传播机制，揭示了标准变量重要性指标难以捕捉的依赖关系。两个实际案例展示了该指标在审查模型是否存在对某些保护属性（如性别或种族）隐性依赖方面的应用，为监管机构和从业者提供了一种系统而高效的公平性与系统风险评估工具。

BibTeX

```
@article{2512.13892v1,
  title={One Permutation Is All You Need: Fast, Reliable Variable Importance and Model Stress-Testing},
  author={Albert Dorador},
  journal={arXiv preprint arXiv:2512.13892v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13892v1}
}
```

## [Privacy-Enhancing Infant Cry Classification with Federated Transformers and Denoising Regularization](http://arxiv.org/abs/2512.13880v1)

Geofrey Owino, Bernard Shibwabo

[PDF](https://arxiv.org/pdf/2512.13880v1)
[Abstract](http://arxiv.org/abs/2512.13880v1)

### 中文摘要

婴儿哭声分类有助于早期评估婴儿需求。然而，由于音频数据的隐私问题、对背景噪声的敏感性以及不同录音环境带来的域偏移，这类解决方案的部署受到限制。我们提出了一套端到端的婴儿哭声分析流程，集成了去噪自动编码器（DAE）、卷积编码器和采用通信高效联邦学习（FL）训练的Transformer编码器。该系统实现了设备端的去噪、自适应分割、事后校准以及基于能量的超出分布（OOD）拒绝功能。联邦训练采用正则化控制变量更新策略，结合8比特的适配器增量，在安全聚合环境下进行。利用带有ESC-50噪声叠加的Baby Chillanto和Donate-a-Cry数据集，模型在宏平均F1得分为0.938、AUC为0.962、预期校准误差（ECE）为0.032的指标下表现优异，同时将每轮客户端上传的数据量从约36至42 MB降低至3.3 MB。基于NVIDIA Jetson Nano（4 GB，TensorRT FP16）实现的实时边缘端推断，每秒谱图的处理时间为96毫秒。这些成果展示了一条面向隐私保护、抗噪声和通信效率的实用婴儿哭声分类方案，适合于联邦部署。

BibTeX

```
@article{2512.13880v1,
  title={Privacy-Enhancing Infant Cry Classification with Federated Transformers and Denoising Regularization},
  author={Geofrey Owino and Bernard Shibwabo},
  journal={arXiv preprint arXiv:2512.13880v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13880v1}
}
```

## [VajraV1 -- The most accurate Real Time Object Detector of the YOLO family](http://arxiv.org/abs/2512.13834v1)

Naman Balbir Singh Makkar

[PDF](https://arxiv.org/pdf/2512.13834v1)
[Abstract](http://arxiv.org/abs/2512.13834v1)

### 中文摘要

近年来，随着YOLOv10、YOLOv11、YOLOv12和YOLOv13在2024年至2025年陆续发布，实时目标检测技术取得了显著进展。本技术报告提出了VajraV1模型架构，在现有的基于YOLO的检测器基础上进行了架构优化。VajraV1结合了先前各类YOLO模型的优秀设计方案，在保持较快推理速度的同时，达到了实时目标检测中的最新精度水平。
在COCO验证集上，VajraV1-Nano模型实现了44.3%的平均精度（mAP），超越YOLOv12-N 3.7%和YOLOv13-N 2.7%，并且推理延迟与YOLOv12-N和YOLOv11-N相当。VajraV1-Small模型的mAP达到了50.4%，超过YOLOv12-S和YOLOv13-S各2.4%。VajraV1-Medium获得52.7%的mAP，优于YOLOv12-M 0.2%。VajraV1-Large的mAP为53.7%，超越YOLOv13-L 0.3%。而VajraV1-Xlarge则实现了56.2%的mAP，领先所有现有的实时目标检测器。

BibTeX

```
@article{2512.13834v1,
  title={VajraV1 -- The most accurate Real Time Object Detector of the YOLO family},
  author={Naman Balbir Singh Makkar},
  journal={arXiv preprint arXiv:2512.13834v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13834v1}
}
```

## [Toward Noise-Aware Audio Deepfake Detection: Survey, SNR-Benchmarks, and Practical Recipes](http://arxiv.org/abs/2512.13744v1)

Udayon Sen, Alka Luqman, Anupam Chattopadhyay

[PDF](https://arxiv.org/pdf/2512.13744v1)
[Abstract](http://arxiv.org/abs/2512.13744v1)

### 中文摘要

深度伪造音频检测在强大的预训练编码器（如WavLM、Wav2Vec2、MMS）推动下取得了迅速发展。然而，在现实环境下的检测性能——例如背景噪声（家庭、办公室、交通等）、房间混响以及消费者渠道——往往远不及在洁净实验室条件下的表现。我们对最先进的音频深度伪造检测模型的鲁棒性进行了调研与评估，并提出了一套可复现的框架，将MS-SNSD噪声与ASVspoof 2021深伪语音片段相结合，在受控的信噪比（SNR）条件下进行测试。SNR是衡量噪声严重程度的常用指标，可以在接近洁净（35 dB）到非常嘈杂（-5 dB）之间进行扫描，从而量化模型在不同噪声水平下的性能退化情况。本文研究了多条件训练和固定SNR测试对于预训练编码器（WavLM、Wav2Vec2、MMS）的影响，报告了二分类和四分类（真实性×是否受污染）任务的准确率、ROC-AUC和等错误率（EER）。实验结果显示，微调模型在SNR为10-0 dB范围内明显降低了EER，提升幅度在10到15个百分点。

BibTeX

```
@article{2512.13744v1,
  title={Toward Noise-Aware Audio Deepfake Detection: Survey, SNR-Benchmarks, and Practical Recipes},
  author={Udayon Sen and Alka Luqman and Anupam Chattopadhyay},
  journal={arXiv preprint arXiv:2512.13744v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13744v1}
}
```

## [Complex Mathematical Expression Recognition: Benchmark, Large-Scale Dataset and Strong Baseline](http://arxiv.org/abs/2512.13731v1)

Weikang Bai, Yongkun Du, Yuchen Su, Yazhen Xie, Zhineng Chen

[PDF](https://arxiv.org/pdf/2512.13731v1)
[Abstract](http://arxiv.org/abs/2512.13731v1)

### 中文摘要

数学表达式识别（MER）在简单表达式的识别方面已取得显著进展，但对于包含大量符号和多行的复杂数学表达式的鲁棒识别仍然是一项艰巨的挑战。本文首先提出了CMER-Bench，这是一个经过精心构建的基准测试，将表达式分为易、中等和复杂三个难度等级。基于CMER-Bench，我们对现有的MER模型和通用多模态大型语言模型（MLLMs）进行了全面评估。结果显示，当前方法在简单和中等难度表达式上表现良好，但在处理复杂数学表达式时性能显著下降，主要原因在于现有公开训练数据集主要由简单样本组成。为此，我们提出了强调复杂数学表达式识别的两个大规模数据集——MER-17M和CMER-3M，这些数据集提供了丰富多样的样本，支持开发出更准确、更鲁棒的复杂MER模型。此外，为应对复杂表达式在空间布局上的挑战，我们引入了一种新颖的表达式Tokenization方法，以及一种名为“结构化数学语言”的新表示方式，该表示显式建模表达式的层次结构与空间结构，超越了传统的LaTeX格式。基于这些技术，我们提出了一种专用模型CMERNet，该模型采用编码器-解码器架构，基于CMER-3M数据集进行训练。实验结果表明，CMERNet仅用125百万参数，就显著优于现有的MER模型和MLLMs，在CMER-Bench上的表现突出。

BibTeX

```
@article{2512.13731v1,
  title={Complex Mathematical Expression Recognition: Benchmark, Large-Scale Dataset and Strong Baseline},
  author={Weikang Bai and Yongkun Du and Yuchen Su and Yazhen Xie and Zhineng Chen},
  journal={arXiv preprint arXiv:2512.13731v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13731v1}
}
```

## [Blind Radio Mapping via Spatially Regularized Bayesian Trajectory Inference](http://arxiv.org/abs/2512.13701v1)

Zheng Xing, Junting Chen

[PDF](https://arxiv.org/pdf/2512.13701v1)
[Abstract](http://arxiv.org/abs/2512.13701v1)

### 中文摘要

无线电地图通过捕捉信道特性的空间分布，为智能无线应用提供关键支持。然而，传统的无线地图构建方法依赖大量带有位置标签的数据，成本高昂且在实际场景中难以推广。本文提出了一种盲态无线地图构建框架，该框架无需位置标签即可从室内多输入多输出（MIMO）正交频分复用（OFDM）信道测量中推断用户轨迹。首先，我们证明了在准镜面环境模型下，无信号直达（NLOS）条件下的信道状态信息（CSI）具有空间连续性，从而推导出一项与实际距离成正比的CSI距离指标。对于泊松分布的接入点（AP）部署中的直线轨迹，本文还显示，即使在角度分辨率较差的情况下，定位误差的克拉美-克拉限制（CRLB）也会在渐近意义下趋于零。基于这些理论结果，我们提出了一种空间正则化的贝叶斯推断框架，该框架能够联合估计信道特征、区分直视（LOS）与非直视（NLOS）条件，以及恢复用户轨迹。利用射线追踪数据集进行的实验证明，该方法实现了平均定位误差0.68米，地图重建误差仅为3.3%，验证了所提出盲性映射技术的有效性。

BibTeX

```
@article{2512.13701v1,
  title={Blind Radio Mapping via Spatially Regularized Bayesian Trajectory Inference},
  author={Zheng Xing and Junting Chen},
  journal={arXiv preprint arXiv:2512.13701v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13701v1}
}
```

## [gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and Optimal Power Flow Data Generation](http://arxiv.org/abs/2512.14658v1)

Alban Puech, Matteo Mazzonelli, Celia Cintas, Tamara R. Govindasamy, Mangaliso Mngomezulu, Jonas Weiss, Matteo Baù, Anna Varbella, François Mirallès, Kibaek Kim, Le Xie, Hendrik F. Hamann, Etienne Vos, Thomas Brunschwiler

[PDF](https://arxiv.org/pdf/2512.14658v1)
[Abstract](http://arxiv.org/abs/2512.14658v1)

### 中文摘要

我们介绍了gridfm-datakit-v1，这是一个用于生成逼真且多样化潮流（Power Flow, PF）和最优潮流（Optimal Power Flow, OPF）数据集的Python库，旨在训练机器学习（ML）求解器。现有的数据集和库面临三个主要挑战：（1）缺乏真实的随机负荷和线路拓扑扰动，限制了场景的多样性；（2）PF数据集仅限于满足OPF约束的点，阻碍ML求解器在超出运行限制（如线路过载或电压偏离）情况下的泛化能力；（3）OPF数据集采用固定的发电机成本函数，限制了在不同成本条件下的泛化。本项工作通过以下方式解决这些问题：（1）结合基于实际负荷变化的全局缩放、局部噪声引入及任意N-k线路扰动，生成多样而逼真的数据集；（2）生成超出运行限制的PF样本；（3）提供成本变化的发电机OPF数据。这一工具还能高效扩展至大型电网（最多可含10,000个节点）。文中对比了与OPFData、OPF-Learn、PGLearn和PFΔ等方法的性能。该库已在GitHub（https://github.com/gridfm/gridfm-datakit）开源，采用Apache 2.0许可证，并可通过`pip install gridfm-datakit`安装。

BibTeX

```
@article{2512.14658v1,
  title={gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and Optimal Power Flow Data Generation},
  author={Alban Puech and Matteo Mazzonelli and Celia Cintas and Tamara R. Govindasamy and Mangaliso Mngomezulu and Jonas Weiss and Matteo Baù and Anna Varbella and François Mirallès and Kibaek Kim and Le Xie and Hendrik F. Hamann and Etienne Vos and Thomas Brunschwiler},
  journal={arXiv preprint arXiv:2512.14658v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14658v1}
}
```

## [JMMMU-Pro: Image-based Japanese Multi-discipline Multimodal Understanding Benchmark via Vibe Benchmark Construction](http://arxiv.org/abs/2512.14620v1)

Atsuyuki Miyai, Shota Onohara, Jeonghun Baek, Kiyoharu Aizawa

[PDF](https://arxiv.org/pdf/2512.14620v1)
[Abstract](http://arxiv.org/abs/2512.14620v1)

### 中文摘要

本文介绍了日本多学科多模态理解基准——JMMMU-Pro，以及一种可扩展的Vibe基准构建方法。随着从MMMU到MMMU-Pro的发展，JMMMU-Pro通过将题目图片和题目信息合成为一幅图像，拓展了原有基准，强调通过视觉感知实现整合的视觉-文字理解。为构建JMMMU-Pro，我们提出了Vibe基准构建法，一种采用图像生成模型（如Nano Banana Pro）生成候选视觉问题，并由人工验证输出，必要时通过调整提示词重新生成以确保质量的方法。利用Nano Banana Pro的高度逼真图像生成能力及其嵌入干净日文文本的特性，我们低成本地构建了包含多样背景和布局设计的高质量基准。实验结果显示，所有开源大型多模态模型在JMMMU-Pro上都表现出明显的不足，凸显了该基准在引导开源社区未来研究中的重要作用。我们相信，JMMMU-Pro为评估大型多模态模型日语能力提供了更为严格的评价工具，同时，我们的Vibe基准构建方法也为未来基于图像的视觉问答基准的快速开发提供了高效的指导。

BibTeX

```
@article{2512.14620v1,
  title={JMMMU-Pro: Image-based Japanese Multi-discipline Multimodal Understanding Benchmark via Vibe Benchmark Construction},
  author={Atsuyuki Miyai and Shota Onohara and Jeonghun Baek and Kiyoharu Aizawa},
  journal={arXiv preprint arXiv:2512.14620v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14620v1}
}
```

## [Low-Resource, High-Impact: Building Corpora for Inclusive Language Technologies](http://arxiv.org/abs/2512.14576v1)

Ekaterina Artemova, Laurie Burchell, Daryna Dementieva, Shu Okabe, Mariya Shmatova, Pedro Ortiz Suarez

[PDF](https://arxiv.org/pdf/2512.14576v1)
[Abstract](http://arxiv.org/abs/2512.14576v1)

### 中文摘要

本教程（https://tum-nlp.github.io/low-resource-tutorial）旨在为自然语言处理（NLP）从业者、研究人员和开发者提供指导，特别是针对多语言和低资源语言的应用，帮助他们开发更具公平性和社会影响力的语言技术。学员将掌握一套实用工具，用于构建面向代表性不足语言的端到端NLP流程——包括数据收集与网页爬取、平行句子挖掘、机器翻译，以及文本分类、多模态推理等下游应用。教程将介绍应对数据稀缺和文化差异挑战的策略，提供实操方法和模型框架。我们将聚焦于公平、可重复以及以社区为导向的开发方法，立足于实际场景。课程中将展示涵盖十余种不同语系、不同地缘政治背景的语言的多样化案例，包括数字资源丰富的语言和资源极为匮乏的少数语言。

BibTeX

```
@article{2512.14576v1,
  title={Low-Resource, High-Impact: Building Corpora for Inclusive Language Technologies},
  author={Ekaterina Artemova and Laurie Burchell and Daryna Dementieva and Shu Okabe and Mariya Shmatova and Pedro Ortiz Suarez},
  journal={arXiv preprint arXiv:2512.14576v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14576v1}
}
```

## [Polypersona: Persona-Grounded LLM for Synthetic Survey Responses](http://arxiv.org/abs/2512.14562v1)

Tejaswani Dash, Dinesh Karri, Anudeep Vurity, Gautam Datla, Tazeem Ahmad, Saima Rafi, Rohith Tangudu

[PDF](https://arxiv.org/pdf/2512.14562v1)
[Abstract](http://arxiv.org/abs/2512.14562v1)

### 中文摘要

本文提出了PolyPersona，一种生成式框架，用于合成多领域中带有个性设定的问卷调查回答。该框架采用参数高效的LoRA适配器，结合4位量化技术，在资源适应性训练环境下对紧凑型对话模型进行指令微调。通过一种明确保留个性线索的对话式数据管道，确保生成的回答在行为表现上一致性。利用该数据管道，我们构建了一个包含3,568条合成问卷回答的多领域数据集，涵盖十个不同领域和433个不同的人格设定，从而实现对模型的受控指令微调和系统性多领域评估。评估采用结合标准文本生成指标（如BLEU、ROUGE和BERTScore）以及专为评估结构连贯性、风格一致性和情感对齐设计的问卷特定指标的多指标评估体系。实验结果显示，诸如TinyLlama 1.1B和Phi-2等紧凑模型在性能上与规模更大的7B至8B基础模型相当，最高BLEU得分达0.090，ROUGE-1得分为0.429。研究表明，基于个性条件的微调使小型语言模型能够生成可靠且连贯的合成问卷数据。该框架为问卷数据的高效、可复现的生成提供了有效途径，支持大规模评估，同时通过透明开放的协议便于偏见分析。

BibTeX

```
@article{2512.14562v1,
  title={Polypersona: Persona-Grounded LLM for Synthetic Survey Responses},
  author={Tejaswani Dash and Dinesh Karri and Anudeep Vurity and Gautam Datla and Tazeem Ahmad and Saima Rafi and Rohith Tangudu},
  journal={arXiv preprint arXiv:2512.14562v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14562v1}
}
```

## [CLNet: Cross-View Correspondence Makes a Stronger Geo-Localizationer](http://arxiv.org/abs/2512.14560v1)

Xianwei Cao, Dou Quan, Shuang Wang, Ning Huyan, Wei Wang, Yunan Li, Licheng Jiao

[PDF](https://arxiv.org/pdf/2512.14560v1)
[Abstract](http://arxiv.org/abs/2512.14560v1)

### 中文摘要

基于图像检索的跨视角地理位置识别（IRCVGL）旨在匹配由截然不同视角拍摄的图像，例如卫星图像和街景图像。现有方法主要依赖于学习稳健的全局表征或隐式特征对齐，往往难以准确建模对精确定位至关重要的显式空间对应关系。在本研究中，我们提出了一种新颖的考虑对应关系的特征优化框架，称为CLNet，旨在明确弥合不同视角之间的语义差距与几何差异。CLNet将视角对齐过程分解为三个可学习且互补的模块：用于通过潜在对应场空间对齐跨视角特征的神经对应图（NCM）；利用多层感知机（MLP）变换在不同视角间重映射特征的非线性嵌入转换器（NEC）；以及在学习到的空间线索引导下重新加权信息丰富的特征通道的全局特征重新校准（GFR）模块。所提出的CLNet能够同时捕捉高层次语义信息与细粒度对齐关系。在四个公开基准数据集——CVUSA、CVACT、VIGOR和University-1652上的大量实验表明，所提出的CLNet在性能上实现了最优，并具备更好的可解释性和泛化能力。

BibTeX

```
@article{2512.14560v1,
  title={CLNet: Cross-View Correspondence Makes a Stronger Geo-Localizationer},
  author={Xianwei Cao and Dou Quan and Shuang Wang and Ning Huyan and Wei Wang and Yunan Li and Licheng Jiao},
  journal={arXiv preprint arXiv:2512.14560v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14560v1}
}
```

## [VLegal-Bench: Cognitively Grounded Benchmark for Vietnamese Legal Reasoning of Large Language Models](http://arxiv.org/abs/2512.14554v1)

Nguyen Tien Dong, Minh-Anh Nguyen, Thanh Dat Hoang, Nguyen Tuan Ngoc, Dao Xuan Quang Minh, Phan Phi Hai, Nguyen Thi Ngoc Anh, Dang Van Tu, Binh Vu

[PDF](https://arxiv.org/pdf/2512.14554v1)
[Abstract](http://arxiv.org/abs/2512.14554v1)

### 中文摘要

大规模语言模型（LLMs）的快速发展为人工智能在法律领域的应用开辟了新的可能性。然而，越南法律法规的复杂性、层级结构以及频繁的修订，为评估这些模型对法律知识的理解和利用能力带来了巨大挑战。为弥补这一缺口，本文提出了越南法律基准（VLegal-Bench），这是首个系统性评估LLMs在越南法律任务中表现的综合性基准。该基准借鉴布鲁姆的认知分类学，涵盖了多层次的法律理解能力，通过设计反映实际使用场景的任务进行测试。VLegal-Bench由10,450个样本组成，采用严谨的标注流程，由法律专家对每个实例进行标注和交叉验证，确保所有样本均基于权威法律文件，真实再现法律助理的工作流程，包括一般法律问答、检索增强生成、多步推理以及针对越南法律的场景问题解决。通过提供标准化、透明且具有认知深度的评估框架，VLegal-Bench为评估LLMs在越南法律背景下的性能打下坚实基础，并促进更可靠、可解释且符合伦理的法律辅助人工智能系统的开发。

BibTeX

```
@article{2512.14554v1,
  title={VLegal-Bench: Cognitively Grounded Benchmark for Vietnamese Legal Reasoning of Large Language Models},
  author={Nguyen Tien Dong and Minh-Anh Nguyen and Thanh Dat Hoang and Nguyen Tuan Ngoc and Dao Xuan Quang Minh and Phan Phi Hai and Nguyen Thi Ngoc Anh and Dang Van Tu and Binh Vu},
  journal={arXiv preprint arXiv:2512.14554v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14554v1}
}
```

## [CAPRMIL: Context-Aware Patch Representations for Multiple Instance Learning](http://arxiv.org/abs/2512.14540v1)

Andreas Lolos, Theofilos Christodoulou, Aris L. Moustakas, Stergios Christodoulidis, Maria Vakalopoulou

[PDF](https://arxiv.org/pdf/2512.14540v1)
[Abstract](http://arxiv.org/abs/2512.14540v1)

### 中文摘要

在计算机病理学中，由于全景切片（WSIs）具有千兆像素级别的巨大尺度以及像素级标注的稀缺，弱监督已成为深度学习的标准方法，其中多实例学习（MIL）被确立为进行切片级模型训练的主要框架。本文提出了一种新颖的MIL方法设置，受到神经偏微分方程（PDE）求解器领域的启发。我们不依赖复杂的基于注意力的聚合机制，而是提出了一种高效的、与聚合器无关的框架，消除了MIL聚合器中相关性学习的复杂性。CAPRMIL能够生成丰富的具有上下文感知的块级嵌入，有助于在后续任务中实现有效的相关性学习。通过将使用冻结的块编码器提取的块特征投影到一组具有全局上下文／形态意识的令牌中，并利用多头自注意力机制，CAPRMIL以与包大小成线性关系的计算复杂度引入全球上下文信息。结合简单的平均MIL聚合器，CAPRMIL在多个公共病理学基准测试中达到了与最先进模型相匹配的切片级性能，同时相比于最先进的MIL模型，参数总数减少了48%至92.8%，推理时的FLOPs降低了52%至99%，在GPU内存使用效率和训练时间方面也名列前茅。我们的结果表明，在聚合之前学习丰富的、具有上下文感知的实例表示，是一种有效且具有可扩展性的替代复杂池化机制，用于全景切片分析。我们的代码已在https://github.com/mandlos/CAPRMIL 开源。

BibTeX

```
@article{2512.14540v1,
  title={CAPRMIL: Context-Aware Patch Representations for Multiple Instance Learning},
  author={Andreas Lolos and Theofilos Christodoulou and Aris L. Moustakas and Stergios Christodoulidis and Maria Vakalopoulou},
  journal={arXiv preprint arXiv:2512.14540v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14540v1}
}
```

## [Effect of Document Packing on the Latent Multi-Hop Reasoning Capabilities of Large Language Models](http://arxiv.org/abs/2512.14427v1)

Gabriele Prato, Shagun Sodhani, Alessandro Sordoni, Sarath Chandar

[PDF](https://arxiv.org/pdf/2512.14427v1)
[Abstract](http://arxiv.org/abs/2512.14427v1)

### 中文摘要

大规模语言模型的训练通常采用将多份文档打包在一起的做法，以提升计算效率。然而，这一过程对模型能力的影响尚未得到充分研究。为填补这一空白，我们探讨了不同文档打包策略对大型语言模型潜在多跳推理能力的影响。研究结果表明，与单独训练单一文档相比，文档打包能够提升模型的性能，尽管这会带来更高的计算成本。为了深入理解其中的机制，我们还开展了消融实验，识别出关键因素，揭示了打包带来优势的原因。最终，我们的研究深化了对大型语言模型训练动态的认识，并为模型优化提供了实践性指导。

BibTeX

```
@article{2512.14427v1,
  title={Effect of Document Packing on the Latent Multi-Hop Reasoning Capabilities of Large Language Models},
  author={Gabriele Prato and Shagun Sodhani and Alessandro Sordoni and Sarath Chandar},
  journal={arXiv preprint arXiv:2512.14427v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14427v1}
}
```

## [Massive Editing for Large Language Models Based on Dynamic Weight Generation](http://arxiv.org/abs/2512.14395v1)

Wentao Wan, Qiqing Lao, Zhiwei Xie, Hefeng Wu, Runnan Lin, Liang Lin, Keze Wang

[PDF](https://arxiv.org/pdf/2512.14395v1)
[Abstract](http://arxiv.org/abs/2512.14395v1)

### 中文摘要

知识编辑（KE）是研究如何以较低成本（相较于预训练）对大规模语言模型（LLMs）中的知识进行修改的领域。目前，在确保编辑的可靠性、通用性和局部性指标的前提下，对LLMs进行大规模编辑仍面临巨大挑战。本文提出了一种基于动态权重生成（MeG）的LLMs大规模编辑方法。我们的MeG方法在特定层中引入动态权重神经元，并利用扩散模型根据所需知识的输入查询条件化生成该神经元的权重，从而实现通过添加单个动态权重神经元来进行大规模知识编辑的目标。实验结果显示，与现有的知识编辑方法相比，本文所提出的MeG在可靠性、通用性和局部性指标方面显著提升了大规模知识编辑的性能，特别是在局部性指标的绝对值指数上取得了高个百分点的提升，充分展现了该方法的优越性。

BibTeX

```
@article{2512.14395v1,
  title={Massive Editing for Large Language Models Based on Dynamic Weight Generation},
  author={Wentao Wan and Qiqing Lao and Zhiwei Xie and Hefeng Wu and Runnan Lin and Liang Lin and Keze Wang},
  journal={arXiv preprint arXiv:2512.14395v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14395v1}
}
```

## [Towards Transferable Defense Against Malicious Image Edits](http://arxiv.org/abs/2512.14341v1)

Jie Zhang, Shuai Dong, Shiguang Shan, Xilin Chen

[PDF](https://arxiv.org/pdf/2512.14341v1)
[Abstract](http://arxiv.org/abs/2512.14341v1)

### 中文摘要

近年来，采用不可感知的扰动对输入图像进行攻击的方法在对抗基于扩散模型的图像编辑系统中的恶意操控方面展现出极大的潜力。然而，现有方法在跨模型评估中普遍存在迁移能力有限的问题。为此，我们提出了“可转移的恶意图像编辑防御（TDAE）”——一种创新的双模态框架，通过协同优化图像与文本，增强图像对恶意编辑的免疫能力。具体而言，在视觉防御层面，我们引入了平坦梯度防御机制（FDM），将梯度正则化纳入对抗目标，通过明确引导扰动朝向平坦的极小值点，有效提升模型对未见编辑模型的鲁棒性。在文本增强保护方面，我们提出一种名为动态提示防御（DPD）的对抗优化范式，周期性地优化文本嵌入，以使免疫图像的编辑效果与原始图像一致，然后在优化后的嵌入下更新图像。通过对多样化嵌入的反复对抗式更新，DPD确保生成的免疫图像能寻求更广泛的免疫增强特征，从而具备较强的跨模型迁移能力。大量实验结果表明，我们的TDAE在单模型和跨模型评估中，均可实现对恶意编辑的有效抑制，达到最先进的性能水平。

BibTeX

```
@article{2512.14341v1,
  title={Towards Transferable Defense Against Malicious Image Edits},
  author={Jie Zhang and Shuai Dong and Shiguang Shan and Xilin Chen},
  journal={arXiv preprint arXiv:2512.14341v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14341v1}
}
```

## [A data-physics hybrid generative model for patient-specific post-stroke motor rehabilitation using wearable sensor data](http://arxiv.org/abs/2512.14329v1)

Yanning Dai, Chenyu Tang, Ruizhi Zhang, Wenyu Yang, Yilan Zhang, Yuhui Wang, Junliang Chen, Xuhang Chen, Ruimou Xie, Yangyue Cao, Qiaoying Li, Jin Cao, Tao Li, Hubin Zhao, Yu Pan, Arokia Nathan, Xin Gao, Peter Smielewski, Shuo Gao

[PDF](https://arxiv.org/pdf/2512.14329v1)
[Abstract](http://arxiv.org/abs/2512.14329v1)

### 中文摘要

动态预测中风后行走能力对个性化康复方案的制定具有重要意义，但现有评估方法仅提供静态的功能障碍评分，无法判断患者是否能够安全完成如坡道行走或爬楼等特定任务。为此，我们开发了一种融合数据与物理模型的生成框架，能够从单次20米平地行走试验中重建中风幸存者的神经肌肉控制机制，并预测在不同康复场景下的任务条件行走表现。该系统结合了穿戴式传感器的运动学数据、比例微分物理控制器、健康运动图谱（Healthy Motion Atlas）以及目标导向的深度强化学习技术，包括行为克隆和生成对抗模仿学习，以生成符合物理现实且具有患者个体特征的坡道和楼梯步态模拟。在11名中风幸存者的案例中，个性化控制器在保持独特步态特征的同时，关节角度和终端执行点的模拟精度分别提高了4.73%和12.10%，培训时间相比纯物理模型基础缩短至25.56%。在一项涉及21名住院患者的多中心试点中，利用本框架预测的步态结果指导任务选择和难度调整的临床医师，在28天的标准康复过程中，其Fugl-Meyer下肢评分提升幅度明显优于对照组（平均变化6.0分对3.7分），表明本生成式、任务预测框架能够增强临床决策的科学性，为实现个性化运动康复策略提供了有效模板。

BibTeX

```
@article{2512.14329v1,
  title={A data-physics hybrid generative model for patient-specific post-stroke motor rehabilitation using wearable sensor data},
  author={Yanning Dai and Chenyu Tang and Ruizhi Zhang and Wenyu Yang and Yilan Zhang and Yuhui Wang and Junliang Chen and Xuhang Chen and Ruimou Xie and Yangyue Cao and Qiaoying Li and Jin Cao and Tao Li and Hubin Zhao and Yu Pan and Arokia Nathan and Xin Gao and Peter Smielewski and Shuo Gao},
  journal={arXiv preprint arXiv:2512.14329v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14329v1}
}
```

## [The Trust in AI-Generated Health Advice (TAIGHA) Scale and Short Version (TAIGHA-S): Development and Validation Study](http://arxiv.org/abs/2512.14278v1)

Marvin Kopka, Azeem Majeed, Gabriella Spinelli, Austen El-Osta, Markus Feufel

[PDF](https://arxiv.org/pdf/2512.14278v1)
[Abstract](http://arxiv.org/abs/2512.14278v1)

### 中文摘要

人工智能工具，如大型语言模型，正日益被公众用于获取健康信息和指导。在健康相关情境中，是否采纳或拒绝AI生成的建议可能会直接影响临床决策。目前的工具，如“对自动系统的信任问卷”主要评估通用技术的可信度，但尚无经过验证的工具专门衡量用户对AI生成健康建议的信任程度。本研究开发并验证了“对AI生成健康建议的信任量表”（TAIGHA）及其四项简短版本（TAIGHA-S），这两类工具基于理论模型，分别衡量信任与不信任，涵盖认知与情感两个组成部分。问卷题项采用生成式AI方法设计，随后经10位领域专家进行内容验证，30位普通受试者进行面孔验证，以及在一份关于症状评估的场景中接受AI建议的385名英国参与者进行心理测量验证。经过自动化筛选，保留28项数据，再经专家评级缩减为10项。TAIGHA表现出极高的内容效度（S-CVI/Ave=0.99），确认的验证模型显示两因素结构拟合优良（CFI=0.98，TLI=0.98，RMSEA=0.07，SRMR=0.03）。内部一致性极高（α=0.95）。收敛效度通过与“自动系统信任问卷”的相关（r=0.67/-0.66）以及用户对AI建议依赖性的相关（信任为r=0.37）得到支持，而发散效度则由与阅读流畅性和认知负荷的低相关（r|<0.25）所证实。TAIGHA-S与完整量表的相关性很高（r=0.96），可靠性良好（α=0.88）。TAIGHA及其短版工具为衡量用户对AI生成健康建议的信任与不信任的有效工具，单独报告信任与不信任有助于更全面地评估AI干预效果，而短版本特别适合时间紧张的场合。

BibTeX

```
@article{2512.14278v1,
  title={The Trust in AI-Generated Health Advice (TAIGHA) Scale and Short Version (TAIGHA-S): Development and Validation Study},
  author={Marvin Kopka and Azeem Majeed and Gabriella Spinelli and Austen El-Osta and Markus Feufel},
  journal={arXiv preprint arXiv:2512.14278v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14278v1}
}
```

## [Georeferencing complex relative locality descriptions with large language models](http://arxiv.org/abs/2512.14228v1)

Aneesha Fernando, Surangika Ranathunga, Kristin Stock, Raj Prasanna, Christopher B. Jones

[PDF](https://arxiv.org/pdf/2512.14228v1)
[Abstract](http://arxiv.org/abs/2512.14228v1)

### 中文摘要

对文本文件进行地理编码通常依赖于地名索引（gazetteer）方法，将地名对应到具体的地理坐标，或采用语言建模技术，将文本术语与地理位置相关联。然而，许多位置描述是通过空间关系进行相对表达的，仅依靠地名或地理指示词进行编码往往不够准确。在生物标本采集记录中，这一问题尤为突出——在GPS普及之前，地点通常以叙述方式描述，而非直接提供坐标。精准的地理编码对生物多样性研究至关重要，但该过程仍然繁琐，亟需自动化解决方案。本文探讨了使用大型语言模型（LLMs）自动进行复杂地点描述的地理编码的潜力，特别关注生物多样性采集领域。我们首先确立了有效的提示策略，然后利用量化低秩调适（QLoRA）方法，对来自多个地区和多种语言的生物多样性数据集进行了微调。实验结果显示，我们的方法在训练数据固定的情况下，在多个数据集上的平均准确率为65%，即有65%的记录在10公里范围内被正确编码。其中最佳表现（以纽约州为例）为：10公里范围内达85%，1公里范围内达67%。所选用的LLM在处理长篇复杂描述时表现出色，充分展现了其在复杂地点描述地理编码中的应用潜力。

BibTeX

```
@article{2512.14228v1,
  title={Georeferencing complex relative locality descriptions with large language models},
  author={Aneesha Fernando and Surangika Ranathunga and Kristin Stock and Raj Prasanna and Christopher B. Jones},
  journal={arXiv preprint arXiv:2512.14228v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14228v1}
}
```

## [Error Bound Analysis of Physics-Informed Neural Networks-Driven T2 Quantification in Cardiac Magnetic Resonance Imaging](http://arxiv.org/abs/2512.14211v1)

Mengxue Zhang, Qingrui Cai, Yinyin Chen, Hang Jin, Jianjun Zhou, Qiu Guo, Peijun Zhao, Zhiping Mao, Xingxing Zhang, Yuyu Xia, Xianwang Jiang, Qin Xu, Chunyan Xiong, Yirong Zhou, Chengyan Wang, Xiaobo Qu

[PDF](https://arxiv.org/pdf/2512.14211v1)
[Abstract](http://arxiv.org/abs/2512.14211v1)

### 中文摘要

物理信息神经网络（PINN）在磁共振成像（MRI）定量参数估计方面展现出极大潜力。尽管现有的深度学习方法能够较为准确地估算T2参数，但它们仍依赖大量的训练数据，且缺乏理论支撑和公认的金标准。因此，鉴于目前尚未基于PINN的方法用于T2估计，我们提出将MRI的基本物理规律——布洛赫方程，嵌入PINN的损失函数中，该方法仅依赖目标扫描数据，无需预先定义的训练数据库。此外，通过推导布洛赫方程解的T2估计误差和泛化误差的严格上界，我们构建了评估PINN定量精度的理论基础。即便在无法访问真实值或金标准的情况下，该理论仍能使我们估算相对于实际T2参数的误差。该方法在数值心脏模型和水模体上验证了T2映射的准确性与理论分析的合理性，在心肌T2范围内表现出优异的定量精度。在临床方面，94例急性心肌梗死（AMI）患者的试验验证了该方法的实用性，实现了符合理论误差界的低误差定量T2估算，彰显出PINN的稳健性与潜力。

BibTeX

```
@article{2512.14211v1,
  title={Error Bound Analysis of Physics-Informed Neural Networks-Driven T2 Quantification in Cardiac Magnetic Resonance Imaging},
  author={Mengxue Zhang and Qingrui Cai and Yinyin Chen and Hang Jin and Jianjun Zhou and Qiu Guo and Peijun Zhao and Zhiping Mao and Xingxing Zhang and Yuyu Xia and Xianwang Jiang and Qin Xu and Chunyan Xiong and Yirong Zhou and Chengyan Wang and Xiaobo Qu},
  journal={arXiv preprint arXiv:2512.14211v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14211v1}
}
```

## [Optimizing Multi-Tier Supply Chain Ordering with a Hybrid Liquid Neural Network and Extreme Gradient Boosting Model](http://arxiv.org/abs/2512.14112v1)

Chunan Tong

[PDF](https://arxiv.org/pdf/2512.14112v1)
[Abstract](http://arxiv.org/abs/2512.14112v1)

### 中文摘要

供应链管理（SCM）面临诸如需求波动和鞭打效应等重大挑战。传统方法甚至最新的大型语言模型（LLMs）在应对如自动售货机测试（Vending Machine Test）等基准任务时，难以胜任，无法处理复杂的连续时间序列数据。尽管像长短期记忆网络（LSTM）和XGBoost等机器学习方法提供了一定的解决方案，但它们通常受到计算效率低下的限制。液态神经网络（LNN）因其在机器人领域的适应性与效率而闻名，但尚未在供应链管理中得到应用。本研究提出了一种基于LNN与XGBoost的混合模型，适用于多层级供应链系统。该模型结合了LNN的动态特征提取能力与XGBoost的全局优化优势，旨在减缓鞭打效应并提升盈利能力。这一创新方法有效解决了效率与适应性的问题，填补了智能供应链管理中的关键空白。

BibTeX

```
@article{2512.14112v1,
  title={Optimizing Multi-Tier Supply Chain Ordering with a Hybrid Liquid Neural Network and Extreme Gradient Boosting Model},
  author={Chunan Tong},
  journal={arXiv preprint arXiv:2512.14112v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14112v1}
}
```

## [Neurosymbolic Inference On Foundation Models For Remote Sensing Text-to-image Retrieval With Complex Queries](http://arxiv.org/abs/2512.14102v1)

Emanuele Mezzi, Gertjan Burghouts, Maarten Kruithof

[PDF](https://arxiv.org/pdf/2512.14102v1)
[Abstract](http://arxiv.org/abs/2512.14102v1)

### 中文摘要

近年来，随着专为航空和卫星影像设计的大型视觉-语言模型（LVLMs）的崛起，遥感（RS）中的文本到图像检索取得了快速发展，并逐步演变为遥感大型视觉-语言模型（RS-LVLMS）。然而，现有模型在可解释性不足以及对复杂空间关系处理不佳方面仍面临重大挑战，限制了其在实际应用中的表现。为此，本文提出了RUNE（基于神经符号实体的推理方法），该方法结合了大型语言模型（LLMs）与神经符号AI，能够通过推理检测到的实体与文本查询导出的一阶逻辑（FOL）表达式之间的兼容性进行图像检索。不同于依赖隐式联合嵌入的RS-LVLMs，RUNE采用显式推理，提升了模型的性能与可解释性。为实现良好的扩展性，我们提出了一种逻辑分解策略，基于条件子集对检测到的实体进行操作，确保比神经方法更短的执行时间。与其利用基础模型进行端到端检索，不如仅用其生成FOL表达式，将推理任务交由神经符号推理模块处理。在评估方面，我们对原用于目标检测的DOTA数据集进行了改造，加入比当前基准更复杂的查询内容，以检验模型的检索能力。结果显示，LLMs在文本到逻辑的转换中具有出色的效果，RUNE在与最先进的RS-LVLMS的对比中表现出优越的性能。我们还提出了两个衡量指标——查询复杂度鲁棒性（RRQC）和图像不确定性鲁棒性（RRIU），用以评估模型在不同查询复杂度和图像不确定性条件下的表现。实验结果表明，RUNE在复杂的遥感检索任务中优于联合嵌入模型，在性能、鲁棒性和可解释性方面均实现了提升。最后，通过一例洪水后卫星图像检索的应用案例，展示了RUNE在实际遥感应用中的潜力。

BibTeX

```
@article{2512.14102v1,
  title={Neurosymbolic Inference On Foundation Models For Remote Sensing Text-to-image Retrieval With Complex Queries},
  author={Emanuele Mezzi and Gertjan Burghouts and Maarten Kruithof},
  journal={arXiv preprint arXiv:2512.14102v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14102v1}
}
```

## [SDAR-VL: Stable and Efficient Block-wise Diffusion for Vision-Language Understanding](http://arxiv.org/abs/2512.14068v1)

Shuang Cheng, Yuhua Jiang, Zineng Zhou, Dawei Liu, Wang Tao, Linfeng Zhang, Biqing Qi, Bowen Zhou

[PDF](https://arxiv.org/pdf/2512.14068v1)
[Abstract](http://arxiv.org/abs/2512.14068v1)

### 中文摘要

块级离散扩散在生成速度与因果依赖建模之间提供了一个具有吸引力的平衡，使其成为视觉-语言模型的有前景的基础架构。然而，由于训练成本高、收敛速度慢以及不稳定性等问题，其实际应用一直受到限制，至今仍落后于性能强劲的自回归（AR）模型。我们提出了“SDAR-VL”，这是首次系统性地将块级离散扩散应用于大规模视觉-语言理解（VLU）任务，并配备了一个“高效且稳定的训练一体化框架”。该框架整合了三大组件：（1）\*\*异步块级噪声调度\*\*，用于在每个批次中实现多样化的监督信号；（2）\*\*有效遮罩比率缩放\*\*，在随机遮罩的条件下实现无偏的损失归一化；（3）\*\*渐进贝塔噪声课程\*\*，在增加有效遮罩覆盖率的同时，保持腐蚀多样性。实验在21个单图、多图和视频基准上显示，SDAR-VL在\*\*训练效率\*\*、\*\*收敛稳定性\*\*和\*\*任务性能\*\*方面均优于传统块扩散方法。在此评测平台上，SDAR-VL创下了基于扩散的视觉-语言模型的新纪录，并且在相同性能设置下，超越或赶超强自回归模型如LLaVA-OneVision，以及全球扩散基线模型LLaDA-V，充分证明块级离散扩散作为VLU的实际基础架构具有广阔的应用前景。

BibTeX

```
@article{2512.14068v1,
  title={SDAR-VL: Stable and Efficient Block-wise Diffusion for Vision-Language Understanding},
  author={Shuang Cheng and Yuhua Jiang and Zineng Zhou and Dawei Liu and Wang Tao and Linfeng Zhang and Biqing Qi and Bowen Zhou},
  journal={arXiv preprint arXiv:2512.14068v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14068v1}
}
```

## [Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation](http://arxiv.org/abs/2512.14048v1)

Shen Li, Li Huang, Shaoxiong Zhan, Weifeng Sun, Tao Yin, Zhongxin Liu, Meng Yan

[PDF](https://arxiv.org/pdf/2512.14048v1)
[Abstract](http://arxiv.org/abs/2512.14048v1)

### 中文摘要

大型语言模型（LLMs）展现出强大的生成能力，并在代码生成领域展现出巨大潜力。现有的链式思维（CoT）提示方法通过引导模型推导中间步骤来增强其推理能力，但存在两大主要限制：第一，统一应用这些方法容易在简单任务中引发过度思考；第二，它们缺少在代码生成中对意图的抽象表达，例如未能明确建模核心算法设计和效率，从而导致模型关注表面结构而忽视整体问题目标。受到认知经济原则的启发，即仅在必要时采用结构化推理以节省认知资源，我们提出了RoutingGen，一种新颖的基于难度感知的动态路由框架，能够自适应调整代码生成的提示策略。对于简单任务，采用少-shot提示；而对于较复杂的任务，则调用一种结构化推理策略，称为意图链式思维（ICoT），该策略旨在引导模型捕捉任务意图，例如核心算法逻辑及其时间复杂度。多模型和六个标准代码生成基准的实验结果表明，RoutingGen在大多数场景中取得了最新的性能，同时整体令牌消耗比平均减少了46.37%。此外，ICoT在挑战性基准上优于六个现有的提示基线方法。

BibTeX

```
@article{2512.14048v1,
  title={Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation},
  author={Shen Li and Li Huang and Shaoxiong Zhan and Weifeng Sun and Tao Yin and Zhongxin Liu and Meng Yan},
  journal={arXiv preprint arXiv:2512.14048v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14048v1}
}
```

## [Evaluating Small Language Models for Agentic On-Farm Decision Support Systems](http://arxiv.org/abs/2512.14043v1)

Enhong Liu, Haiyu Yang, Miel Hostens

[PDF](https://arxiv.org/pdf/2512.14043v1)
[Abstract](http://arxiv.org/abs/2512.14043v1)

### 中文摘要

大型语言模型（LLM）有望通过支持决策制定和扩大知识访问，为乳制品领域的学者和农民提供帮助，特别是对技术水平有限的利益相关者。然而，由于其庞大的计算需求，目前基本通过云端服务提供，导致基于LLM的决策支持工具在乳品农业中的应用受到限制。为解决这一问题，需要开发轻量级的替代方案，能够在农场硬件上本地运行。在本研究中，我们在贴近农业实际的计算条件下，基于HuggingFace平台的20个开源小型语言模型（SLM）进行了基准测试。继之前的工作基础上，我们构建了一个包含五个任务专用代理的智能AI系统，这些任务包括文献检索、网页搜索、SQL数据库交互、NoSQL数据库交互以及基于预测模型的图形生成。评估分为两个阶段：第一阶段，通过五个测试问题进行初步筛选，以识别能够执行基本乳业相关指令且在受限计算环境中表现可靠的模型；第二阶段，筛选成功的模型通过包含上述五类任务的30个问题（每类五个）以及一个关于完整性与不当行为的类别，进行深入评估。结果显示，Qwen-4B在大部分任务类别中表现出色，尽管在通过PySpark进行NoSQL数据库交互时表现不稳定。据我们所知，这是首次系统评估SLM作为乳业决策引擎的可行性，重点关注隐私保护和计算效率。结果显示，SLM辅助工具具有在乳业实际应用中的潜力，但仍面临一些挑战，需进一步微调以提升模型在乳业特定问题上的表现。

BibTeX

```
@article{2512.14043v1,
  title={Evaluating Small Language Models for Agentic On-Farm Decision Support Systems},
  author={Enhong Liu and Haiyu Yang and Miel Hostens},
  journal={arXiv preprint arXiv:2512.14043v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14043v1}
}
```

## [PerfCoder: Large Language Models for Interpretable Code Performance Optimization](http://arxiv.org/abs/2512.14018v1)

Jiuding Yang, Shengyao Lu, Hongxuan Liu, Shayan Shirahmad Gale Bagi, Zahra Fazel, Tomasz Czajkowski, Di Niu

[PDF](https://arxiv.org/pdf/2512.14018v1)
[Abstract](http://arxiv.org/abs/2512.14018v1)

### 中文摘要

大型语言模型（LLMs）在自动代码生成方面取得了显著进展，但其生成高性能代码的能力仍然有限——这是实际软件系统中的一项关键需求。我们认为，当前的LLMs面临的困难不仅仅在于数据匮乏，更重要的是缺乏指导可解释且有效提升性能的监督机制。在本研究中，我们提出了PerfCoder，一系列专为通过可解释、定制化的优化策略从源代码生成性能增强代码而设计的LLMs。PerfCoder在经过精心筛选的真实优化轨迹及人类可读注释的资料集上进行了微调，并通过结合运行时测量的强化微调实现了偏好对齐，使其能够提出针对输入的改进策略并直接应用，而无需依赖反复迭代优化。在PIE代码性能基准测试中，PerfCoder在运行速度提升和有效优化率方面均优于所有现有模型，证明性能优化不仅依赖于模型规模，更需要优化策略的认知能力。此外，PerfCoder还能生成可解释的源代码反馈，当将其作为输入提供给更大规模的LLM，在“规划-优化”协作流程中使用时，能进一步提升整体表现。具体而言，我们将32B模型和GPT-5的代码优化性能提升至新水平，显著超越其原始性能。

BibTeX

```
@article{2512.14018v1,
  title={PerfCoder: Large Language Models for Interpretable Code Performance Optimization},
  author={Jiuding Yang and Shengyao Lu and Hongxuan Liu and Shayan Shirahmad Gale Bagi and Zahra Fazel and Tomasz Czajkowski and Di Niu},
  journal={arXiv preprint arXiv:2512.14018v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14018v1}
}
```

## [Sparsity-Controllable Dynamic Top-p MoE for Large Foundation Model Pre-training](http://arxiv.org/abs/2512.13996v1)

Can Jin, Hongwu Peng, Mingcan Xiang, Qixin Zhang, Xiangchi Yuan, Amit Hasan, Ohiremen Dibua, Yifan Gong, Yan Kang, Dimitris N. Metaxas

[PDF](https://arxiv.org/pdf/2512.13996v1)
[Abstract](http://arxiv.org/abs/2512.13996v1)

### 中文摘要

稀疏混合专家（MoE）架构通过仅激活一部分专家，有效地扩展了模型容量。然而，标准的Top-k路由策略施加了一种统一的稀疏性模式，未能考虑不同输入令牌的难易程度。虽然Top-p路由作为一种更具灵活性的替代方案，但现有实现通常依赖于固定的全局概率阈值，这导致计算成本难以控制且对超参数的选择较为敏感。本文提出了一种稀疏性可控的动态Top-p路由机制——DTop-p MoE。为了解决优化非微分阈值的难题，我们采用了比例-积分（PI）控制器，动态调节概率阈值，使所有激活专家的稀疏度与预设目标保持一致。此外，我们引入了一种动态路由归一化机制，适应不同层的路由对数，使各层能够学习不同的专家选择模式，且仍然使用统一的全局概率阈值。在大规模语言模型和扩散变换模型上的大量实验表明，DTop-p在性能上始终优于Top-k和固定阈值的Top-p基线算法。我们的分析验证了DTop-p能够精准控制激活专家的数量，同时根据不同的令牌和层智能调配资源。此外，DTop-p在专家粒度、专家容量、模型规模和数据规模方面表现出强大的扩展能力，为大规模MoE预训练提供了一个稳健的框架。

BibTeX

```
@article{2512.13996v1,
  title={Sparsity-Controllable Dynamic Top-p MoE for Large Foundation Model Pre-training},
  author={Can Jin and Hongwu Peng and Mingcan Xiang and Qixin Zhang and Xiangchi Yuan and Amit Hasan and Ohiremen Dibua and Yifan Gong and Yan Kang and Dimitris N. Metaxas},
  journal={arXiv preprint arXiv:2512.13996v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13996v1}
}
```

## [Exploring Machine Learning, Deep Learning, and Explainable AI Methods for Seasonal Precipitation Prediction in South America](http://arxiv.org/abs/2512.13910v1)

Matheus Corrêa Domingos, Valdivino Alexandre de Santiago Júnior, Juliana Aparecida Anochi, Elcio Hideiti Shiguemori, Luísa Mirelle Costa dos Santos, Hércules Carlos dos Santos Pereira, André Estevam Costa Oliveira

[PDF](https://arxiv.org/pdf/2512.13910v1)
[Abstract](http://arxiv.org/abs/2512.13910v1)

### 中文摘要

气象变量的预测由于其复杂的过程而具有挑战性，通常需要先进的模型以确保预测的准确性。准确的降水预报对社会具有重要意义，可靠的预测能够帮助社区减缓气候变化的影响。鉴于人工智能（AI）在当前的相关性，传统的机器学习（ML）和深度学习（DL）技术已被用作动态模型的替代或补充。然而，关于纯数据驱动方法在降水预测中的可行性，仍缺乏广泛的研究。本研究旨在解决这一问题，针对南美地区2019年全季节的降水预测，详细分析了多种经典ML和DL方法。所选的经典ML技术包括随机森林（RF）和极端梯度提升（XGBoost），而DL的对应模型则为一维卷积神经网络（CNN 1D）、长短期记忆网络（LSTM）和门控循环单元（GRU）。此外，巴西全球大气模型（BAM）被用作传统动态模型的代表。我们还利用了可解释的人工智能（XAI）技术，为模型行为提供一些解释。结果显示，LSTM在预测性能方面表现优异，而代表传统动态模型的BAM则表现最差。尽管LSTM具有较高的延迟，但在重污染预报中最为准确。如若考虑成本，XGBoost提供了较低的延迟，虽然略有准确率的下降。研究结果验证了深度学习模型在气候预测中的可行性，巩固了主要气象与气候预测中心的全球趋势。

BibTeX

```
@article{2512.13910v1,
  title={Exploring Machine Learning, Deep Learning, and Explainable AI Methods for Seasonal Precipitation Prediction in South America},
  author={Matheus Corrêa Domingos and Valdivino Alexandre de Santiago Júnior and Juliana Aparecida Anochi and Elcio Hideiti Shiguemori and Luísa Mirelle Costa dos Santos and Hércules Carlos dos Santos Pereira and André Estevam Costa Oliveira},
  journal={arXiv preprint arXiv:2512.13910v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13910v1}
}
```



Generated by ArXiv AI Agent • Powered by DeepSeek & Jina AI