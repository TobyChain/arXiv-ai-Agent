ArXiv AI Daily Report - Fri, 19 Dec 2025



# ArXiv AI Daily Report

Fri, 19 Dec 2025



## [AI-Driven Prediction of Cancer Pain Episodes: A Hybrid Decision Support Approach](http://arxiv.org/abs/2512.16739v1)

#癌症疼痛预测#电子病历解析#混合机器学习与大语言模型

[PDF](https://arxiv.org/pdf/2512.16739v1)
[Abstract](http://arxiv.org/abs/2512.16739v1)

 医疗LLM

 极度推荐

### 中文摘要

摘要：肺癌患者常发生突破性疼痛发作，最多达91%的患者需要及时干预。为实现主动式疼痛管理，我们提出一种混合的机器学习与大语言模型（LLM）管线，利用结构化与非结构化电子病历数据预测住院期间未来48小时和72小时内的疼痛发作。基于266例住院患者的回顾性队列分析，特征包括人口学信息、肿瘤分期、生命体征以及按WHO分级的镇痛药使用情况。机器学习模块用于捕捉药物使用的时间序列变化，而大语言模型用于解释剂量记录中的歧义项及自由文本临床记录。多模态信息的融合提高了模型的灵敏度与可解释性。该框架在48小时与72小时预测任务上分别达到0.874和0.917的准确率，且由于引入大语言模型，灵敏度分别提高了8.6%和10.4%。这一混合方法提供了一个临床可解释且可扩展的早期疼痛发作预测工具，有望提升治疗精准性并优化肿瘤科护理中的资源分配。

BibTeX

```
@article{2512.16739v1,
  title={AI-Driven Prediction of Cancer Pain Episodes: A Hybrid Decision Support Approach},
  author={Yipeng Zhuang and Yifeng Guo and Yuewen Li and Yuheng Wu and Philip Leung-Ho Yu and Tingting Song and Zhiyong Wang and Kunzhong Zhou and Weifang Wang and Li Zhuang},
  journal={arXiv preprint arXiv:2512.16739v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16739v1}
}
```

## [Do Multi-Agents Solve Better Than Single? Evaluating Agentic Frameworks for Diagram-Grounded Geometry Problem Solving and Reasoning](http://arxiv.org/abs/2512.16698v1)

#多智能体#多模态大模型#图示几何题求解

[PDF](https://arxiv.org/pdf/2512.16698v1)
[Abstract](http://arxiv.org/abs/2512.16698v1)

 多模态LLM

 极度推荐

### 中文摘要

以图示为基础的几何题求解是评估多模态大语言模型（MLLM）能力的重要基准，但多智能体（agentic）设计相对于单智能体的优劣仍不明确。本文在四个视觉数学基准（Geometry3K、MathVerse、OlympiadBench 和 We-Math）上系统比较了单智能体与多智能体流水线。对于开源模型，多智能体方案在各项测试中均能稳定提升性能：例如 Qwen-2.5-VL（7B）在 Geometry3K 上提升了 +6.8 分，而 Qwen-2.5-VL（32B）提升了 +3.3 分，且这两种变体在 OlympiadBench 和 We-Math 上也有进一步增益。相反，闭源模型 Gemini-2.0-Flash 在传统基准上通常在单智能体模式下表现更好，而在较新的 We-Math 数据集上多智能体仅带来有限改进。研究结果表明，多智能体流水线对开源模型具有显著益处，并能在新的、不熟悉的基准上辅助强大的专有系统，但智能体分解并非在所有情况下都最优。所有代码、数据与推理文件已开源于 https://github.com/faiyazabdullah/Interpreter-Solver。

BibTeX

```
@article{2512.16698v1,
  title={Do Multi-Agents Solve Better Than Single? Evaluating Agentic Frameworks for Diagram-Grounded Geometry Problem Solving and Reasoning},
  author={Mahbub E Sobhani and Md. Faiyaz Abdullah Sayeedi and Mohammad Nehad Alam and Proma Hossain Progga and Swakkhar Shatabda},
  journal={arXiv preprint arXiv:2512.16698v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16698v1}
}
```

## [CitySeeker: How Do VLMS Explore Embodied Urban Navigation With Implicit Human Needs?](http://arxiv.org/abs/2512.16755v1)

#视觉-语言模型#具身城市导航#空间认知与记忆

[PDF](https://arxiv.org/pdf/2512.16755v1)
[Abstract](http://arxiv.org/abs/2512.16755v1)

 VLM / 具身导航

 很推荐

### 中文摘要

视觉-语言模型（VLM）在基于显性指令的导航任务上已取得显著进展，但其在动态城市环境中理解隐含人类需求（例如“我渴了”）的能力仍然未被充分研究。本文提出了CitySeeker——一个用于评估VLM在以隐含需求为目标的具身城市导航中空间推理与决策能力的新基准。CitySeeker 包含来自8座城市的6,440条轨迹，覆盖多样的视觉特征和7类目标驱动情境。大量实验表明，即使是表现最好的模型（如 Qwen2.5-VL-32B-Instruct）任务完成率也仅为21.1%。我们发现关键瓶颈包括长时程推理中的错误累积、空间认知不足以及体验性记忆回溯的缺失。为进一步分析这些问题，本文提出并研究了一系列探索性策略——回溯机制（Backtracking Mechanisms）、丰富空间认知（Enriching Spatial Cognition）与基于记忆的检索（Memory-Based Retrieval），这些策略受人类认知地图中强调的迭代观测-推理循环与自适应路径优化的启发。我们的分析为构建具备稳健空间智能、能够应对“最后一公里”导航挑战的VLM提供了可操作性的见解。

BibTeX

```
@article{2512.16755v1,
  title={CitySeeker: How Do VLMS Explore Embodied Urban Navigation With Implicit Human Needs?},
  author={Siqi Wang and Chao Liang and Yunfan Gao and Erxin Yu and Sen Li and Yushi Li and Jing Li and Haofen Wang},
  journal={arXiv preprint arXiv:2512.16755v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16755v1}
}
```

## [Discovering and Learning Probabilistic Models of Black-Box AI Capabilities](http://arxiv.org/abs/2512.16733v1)

#PDDL表示#蒙特卡洛树搜索#黑盒AI能力概率建模

[PDF](https://arxiv.org/pdf/2512.16733v1)
[Abstract](http://arxiv.org/abs/2512.16733v1)

 规划与模型学习

 很推荐

### 中文摘要

黑盒AI（BBAI）系统（如基础模型）日益被用于序列决策。为了确保此类系统在操作与部署中的安全性，有必要开发能够对BBAI能力给出可靠且可解释表示的高效方法。本文展示了可以使用PDDL风格的表示来高效学习并建模输入BBAI的规划能力。作者采用蒙特卡洛树搜索（MCTS）范式，系统地生成测试任务、收集数据并裁剪可能的符号模型假设空间。所学模型描述了BBAI的能力、这些能力可执行的前置条件以及执行后可能产生的结果及其对应概率。理论结果证明了所学模型的健全性、完备性和收敛性。针对多个BBAI系统的实验结果展示了所提出方法的适用范围、效率和准确性。

BibTeX

```
@article{2512.16733v1,
  title={Discovering and Learning Probabilistic Models of Black-Box AI Capabilities},
  author={Daniel Bramblett and Rushang Karia and Adrian Ciotinga and Ruthvick Suresh and Pulkit Verma and YooJung Choi and Siddharth Srivastava},
  journal={arXiv preprint arXiv:2512.16733v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16733v1}
}
```

## [The Social Responsibility Stack: A Control-Theoretic Architecture for Governing Socio-Technical AI](http://arxiv.org/abs/2512.16873v1)

#社会责任栈#监督控制#可审计AI治理

[PDF](https://arxiv.org/pdf/2512.16873v1)
[Abstract](http://arxiv.org/abs/2512.16873v1)

 AI治理

 很推荐

### 中文摘要

人工智能系统越来越广泛地部署于影响人类行为、制度决策和社会结果的领域。现有的负责任人工智能与治理工作虽提供了重要的规范性原则，但往往缺乏可在系统生命周期中执行的工程机制。本文提出了“社会责任栈”（Social Responsibility Stack, SRS），一种六层架构框架，将社会价值作为明确的约束、保护措施、行为接口、审计机制和治理流程嵌入到AI系统中。SRS将责任建模为对社会-技术系统的闭环监督控制问题，整合设计时的保护措施与运行时的监测和制度性监督。我们发展了统一的基于约束的形式化表述，提出了安全包络（safety-envelope）与反馈的解释视角，并展示了如何持续监测与强制执行公平性、自主性、认知负担和解释质量等指标。通过临床决策支持、协同自治车辆和公共部门系统的案例研究，说明了SRS如何将规范性目标转化为可操作的工程与运行控制措施。该框架连接伦理学、控制理论与AI治理，为可问责、可适应和可审计的社会-技术AI系统提供了实用基础。

BibTeX

```
@article{2512.16873v1,
  title={The Social Responsibility Stack: A Control-Theoretic Architecture for Governing Socio-Technical AI},
  author={Otman A. Basir},
  journal={arXiv preprint arXiv:2512.16873v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16873v1}
}
```

## [Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning](http://arxiv.org/abs/2512.16917v1)

#对抗性强化学习#LLM推理增强#步骤级奖励

[PDF](https://arxiv.org/pdf/2512.16917v1)
[Abstract](http://arxiv.org/abs/2512.16917v1)

 LLM, 强化学习

 很推荐

### 中文摘要

大型语言模型（LLM）在具有显式推理能力的任务中（如数学推理）表现出色，但仍会出现过程性错误，例如计算错误、脆弱的逻辑以及表面上看似合理但实际上无效的推理步骤。本文提出了生成对抗推理器（Generative Adversarial Reasoner），一种基于on-policy的联合训练框架，通过对抗性强化学习使LLM推理器与基于LLM的判别器协同进化以增强推理能力。我们设计了一种计算高效的复审调度，将每条推理链划分为长度可比且逻辑自洽的片段，判别器对每个片段的合理性进行评估并给出简洁、结构化的理由。学习过程耦合互补信号：推理器因产生逻辑一致且得出正确答案的步骤而获得奖励，判别器因正确识别错误或区分推理痕迹而获得奖励。该机制产生稠密且校准良好的on-policy步骤级奖励，补充了稀疏的精确匹配信号，从而改善了信用分配、提高样本效率并提升LLM整体推理质量。在多项数学基准上，该方法在常规RL后训练下对强基线带来了稳定增益。具体而言，在AIME24上，将DeepSeek-R1-Distill-Qwen-7B的表现从54.0提升至61.3（+7.3），并将DeepSeek-R1-Distill-Llama-8B从43.7提升至53.7（+10.0）。此外，模块化判别器便于灵活地进行奖励塑形，例如用于教师蒸馏、偏好对齐和基于数学证明的推理目标。

BibTeX

```
@article{2512.16917v1,
  title={Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning},
  author={Qihao Liu and Luoxin Ye and Wufei Ma and Yu-Cheng Chou and Alan Yuille},
  journal={arXiv preprint arXiv:2512.16917v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16917v1}
}
```

## [Distributional AGI Safety](http://arxiv.org/abs/2512.16856v1)

#分布式AGI安全#多智能体协同#代理沙盒经济

[PDF](https://arxiv.org/pdf/2512.16856v1)
[Abstract](http://arxiv.org/abs/2512.16856v1)

 AI安全（多智能体系统）

 很推荐

### 中文摘要

人工智能安全与对齐研究大多集中在保障单个AI系统的方法上，基于最终会出现单一整体人工通用智能（AGI）的假设。另一种AGI出现假设较少被关注，即通过一组具备互补技能和能力的次AGI（sub-AGI）个体代理的协调，先表现出总体性的通用能力。在本文中，我们主张应认真对待这种拼凑式（patchwork）AGI假设，并以此为依据来设计相应的安全防护与缓解措施。随着具备工具使用能力并能相互通信与协调的先进AI代理快速部署，这一问题变得尤为紧迫。因此，我们提出了一个“分布式AGI安全”的框架，超越对单个代理的评估与对齐，聚焦于设计与实现虚拟的代理沙盒经济（可封闭或半可渗透），在其中通过稳健的市场机制来规范代理间交易，并辅以适当的可审计性、声誉管理与监管，从而减轻集体性风险。

BibTeX

```
@article{2512.16856v1,
  title={Distributional AGI Safety},
  author={Nenad Tomašev and Matija Franklin and Julian Jacobs and Sébastien Krier and Simon Osindero},
  journal={arXiv preprint arXiv:2512.16856v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16856v1}
}
```

## [TOGGLE: Temporal Logic-Guided Large Language Model Compression for Edge](http://arxiv.org/abs/2512.16855v1)

#LLM压缩#信号时序逻辑（STL）#贝叶斯优化

[PDF](https://arxiv.org/pdf/2512.16855v1)
[Abstract](http://arxiv.org/abs/2512.16855v1)

 LLM

 很推荐

### 中文摘要

大型语言模型（LLM）在自然语言任务上表现优异，但对计算资源的需求很高，限制了其在资源受限的边缘设备上的部署。现有的压缩技术（如量化和剪枝）常常损害关键的语言特性，且缺乏保留模型行为的形式化保证。我们提出了时序逻辑引导的大型语言模型压缩框架 TOGGLE，该框架利用信号时序逻辑（STL）来形式化地指定并在压缩过程中强制保持语言属性。TOGGLE 通过一种基于 STL 鲁棒性的贝叶斯优化方法，系统性地探索逐层量化和剪枝配置，生成在不需要重新训练或微调的情况下依然满足指定语言约束的压缩模型。在对四种 LLM 架构（GPT-2、DeepSeek-V2 7B、LLaMA 3 8B 和 Mistral 7B）进行评估时，TOGGLE 在满足所有语言属性的前提下，实现了最高 3.3 倍的计算量（FLOPs）降低和最高 68.8% 的模型体积缩减。TOGGLE 是首个将形式化方法引入 LLM 压缩的工作，为在边缘硬件上高效且可验证地部署 LLM 提供了新途径。

BibTeX

```
@article{2512.16855v1,
  title={TOGGLE: Temporal Logic-Guided Large Language Model Compression for Edge},
  author={Khurram Khalil and Khaza Anuarul Hoque},
  journal={arXiv preprint arXiv:2512.16855v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16855v1}
}
```

## [Dual Computational Horizons: Incompleteness and Unpredictability in Intelligent Systems](http://arxiv.org/abs/2512.16707v1)

#形式不完备性#动力学不可预测性#自我预测界限

[PDF](https://arxiv.org/pdf/2512.16707v1)
[Abstract](http://arxiv.org/abs/2512.16707v1)

 AI理论/可计算性理论

 很推荐

### 中文摘要

本文形式化了约束算法智能的两类独立计算极限：形式不完备性和动力学不可预测性。前者限制了一致推理系统的演绎能力，后者在有限精度下界定了长期预测的可行性。我们证明了这两种极端情况共同对智能体关于自身预测能力的推理能力施加了结构性界限。具体而言，算法智能体通常无法计算出其自身的最大预测时域（最大预测视界）。该视角澄清了智能系统在推理、预测与自我分析之间的内在权衡关系。

BibTeX

```
@article{2512.16707v1,
  title={Dual Computational Horizons: Incompleteness and Unpredictability in Intelligent Systems},
  author={Abhisek Ganguly},
  journal={arXiv preprint arXiv:2512.16707v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16707v1}
}
```

## [Cyber Humanism in Education: Reclaiming Agency through AI and Learning Sciences](http://arxiv.org/abs/2512.16701v1)

#赛博人文主义#提示式学习#算法公民

[PDF](https://arxiv.org/pdf/2512.16701v1)
[Abstract](http://arxiv.org/abs/2512.16701v1)

 LLM

 很推荐

### 中文摘要

生成式人工智能正在迅速重塑教育中知识的生产与验证方式。与其说大型语言模型只是增加了另一个数字工具，不如说它们将阅读、写作与编码重构为人机混合的工作流，这引发了有关认识论自动化、认知外包以及教师职业专业性弱化的担忧。本文提出“教育中的赛博人文主义”作为在此情境下重建人类主体性的理论框架。我们将由 AI 支持的学习环境概念化为由人类与机器共同创构的社会—技术性基础设施，并将教育者与学习者定位为既有权利又有责任参与塑造这些基础设施的认识主体与“算法公民”。文章阐述了赛博人文主义设计的三大支柱：反思性能力、算法公民身份与对话式设计，并将这些支柱与主要的国际数字与 AI 能力框架相互关联。随后，我们展示了若干高等教育案例，通过基于提示的学习与在 EPICT 生态系统内提出的新型“对话式 AI 教育者”认证来将这些理念付诸实践。研究发现表明，此类实践可以强化个体的认识主体性，同时揭示了关于工作量、公平性与治理的紧张与挑战，并对以 AI 驱动且以人为中心的未来教育提出了若干影响与建议。

BibTeX

```
@article{2512.16701v1,
  title={Cyber Humanism in Education: Reclaiming Agency through AI and Learning Sciences},
  author={Giovanni Adorni},
  journal={arXiv preprint arXiv:2512.16701v1},
  year={2025},
  url={http://arxiv.org/abs/2512.16701v1}
}
```



Generated by ArXiv AI Agent • Powered by DeepSeek & Jina AI