ArXiv AI Daily Report - Tue, 16 Dec 2025



# ArXiv AI Daily Report

Tue, 16 Dec 2025



## [Hypergame Rationalisability: Solving Agent Misalignment In Strategic Play](http://arxiv.org/abs/2512.11942v1)

Vince Trencsenyi

[PDF](https://arxiv.org/pdf/2512.11942v1)
[Abstract](http://arxiv.org/abs/2512.11942v1)

### 中文摘要

由于感知差异、信息不对称以及有限理性，博弈论中的参与者会形成对游戏的主观私人认知，这种认知可能与实际的客观事实存在偏差，也可能与其他参与者的解读不一致。尽管传统的博弈论假设通常忽略了这种异质性，超博弈理论（hypergame theory）提供了一个数学框架，用于推理认知模型的偏差。尽管近年来超博弈在处理不确定性相关的动态应用中逐渐受到关注，但其在多智能体系统研究中的实际应用一直受到限制，原因在于缺乏一种统一、形式化且实用的表达语言，以及能高效管理复杂超博弈结构和均衡的可扩展算法。我们的工作正是弥补这一空白，提出了一种声明式、基于逻辑的面向领域的语言，用于编码超博弈结构和超博弈的解概念。通过运用答集编程（answer-set programming），我们开发出一个自动化流程，用于实例化超博弈结构并运行我们创新的超博弈合理化机制——一种寻找能够解释看似非理性结果的信念结构的方法。该语言为超博弈提供了一种统一的形式化框架，并可作为开发细致、基于信念的异质推理器的基础，具有可验证的逻辑保证。这些贡献共同建立了超博弈理论、多智能体系统与战略人工智能之间的联系。

BibTeX

```
@article{2512.11942v1,
  title={Hypergame Rationalisability: Solving Agent Misalignment In Strategic Play},
  author={Vince Trencsenyi},
  journal={arXiv preprint arXiv:2512.11942v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11942v1}
}
```

## [A Monad-Based Clause Architecture for Artificial Age Score (AAS) in Large Language Models](http://arxiv.org/abs/2512.11835v1)

Seyma Yaman Kayadibi

[PDF](https://arxiv.org/pdf/2512.11835v1)
[Abstract](http://arxiv.org/abs/2512.11835v1)

### 中文摘要

大型语言模型（LLMs）常被部署为强大但不透明的系统，关于其内部记忆以及“自我”行为应当如何以原则性和可审计的方式进行管理尚未明确。此前引入的人工时代评分（AAS）通过三个定理提供了数学依据，将其定义为衡量人工记忆老化的指标。在此基础上，本文提出了一种面向工程、基于子句的架构，该架构对LLM的记忆与控制施加类法律约束。从莱布尼茨的 Monadology 中精选的二十个单子（monads）被分为六个束：本体论、动力学、表征与意识、和谐与理性、身体与组织，以及目的论，每个束在AAS内核上实现为可执行的规范。在六个简洁的Python实现中，这些子句族在数值实验中被应用于通道层级的指标，如回忆得分、冗余度和权重。每个实现遵循四个步骤：输入与设置、子句实现、数值结果以及对LLM设计的启示，强调该框架不仅具有哲学基础，而且具有直接的可实现性。实验表明，该子句系统表现出界限明确且具有可解释性的行为：AAS轨迹保持连续且速率受限，矛盾和无支持的陈述会触发明确的惩罚，层级细化机制以有机的结构方式进行揭示。和谐项协调了双重视角与目标-行动对，而窗口漂移则在完美评分中区分了持续改进与持续退化。总体而言，基于单子的子句框架以AAS为支撑，为约束和分析人工智能系统内部动态提供了透明且具备代码层级的蓝图。

BibTeX

```
@article{2512.11835v1,
  title={A Monad-Based Clause Architecture for Artificial Age Score (AAS) in Large Language Models},
  author={Seyma Yaman Kayadibi},
  journal={arXiv preprint arXiv:2512.11835v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11835v1}
}
```

## [Rethinking Label Consistency of In-Context Learning: An Implicit Transductive Label Propagation Perspective](http://arxiv.org/abs/2512.12175v1)

Haoyang Chen, Richong Zhang, Junfan Chen

[PDF](https://arxiv.org/pdf/2512.12175v1)
[Abstract](http://arxiv.org/abs/2512.12175v1)

### 中文摘要

大型语言模型（LLMs）在少量有监督示例的条件下进行上下文学习（ICL），这有助于多种自然语言处理（NLP）任务的实现。一个关键的研究方向是示范提示的选择。现有方法通常采用检索模型，选取语义相似度最高的前K个示例作为示范。然而，我们认为现有方法存在局限性，因为在示范选择过程中未能保证标签的一致性。我们的认知基础源自于对ICL的贝叶斯视角，并结合从传导标注传播角度重新审视ICL。我们将ICL视为一种传导学习方法，融入贝叶斯观点中的潜在概念，推导出相似示范能够引导查询的概念，而具有一致标签的示范则作为估计依据。基于此理解，我们建立了一个标签传播框架，将标签一致性与传播误差上界相联系。为建模标签一致性，我们提出一种数据合成方法，结合语义和标签信息，并采用带有合成数据的TopK采样（TopK-SD）策略，获得标签一致的示范。实验结果显示，TopK-SD在多个基准测试中优于原始的TopK采样方法。我们的工作为理解ICL的内部机制提供了新的视角。

BibTeX

```
@article{2512.12175v1,
  title={Rethinking Label Consistency of In-Context Learning: An Implicit Transductive Label Propagation Perspective},
  author={Haoyang Chen and Richong Zhang and Junfan Chen},
  journal={arXiv preprint arXiv:2512.12175v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12175v1}
}
```

## [Large Language Newsvendor: Decision Biases and Cognitive Mechanisms](http://arxiv.org/abs/2512.12552v1)

Jifei Liu, Zhi Chen, Yuanguang Zhong

[PDF](https://arxiv.org/pdf/2512.12552v1)
[Abstract](http://arxiv.org/abs/2512.12552v1)

### 中文摘要

问题定义：尽管大型语言模型（LLMs）日益被整合到企业决策中，但其潜在的模仿甚至放大人类认知偏差的能力引发了严重且尚未充分理解的风险，这在供应链管理等高风险操作环境中尤为关键。为此，我们以动态情境中的标准新闻商问题（newsvendor problem）为研究对象，探讨领先的LLMs的决策模式，旨在识别它们认知偏差的本质及其源头。方法与结果：通过对GPT-4、GPT-4o和LLaMA-8B进行多轮动态实验，我们检测了五种已确立的决策偏差。结果显示，LLMs持续复制经典的“过低/过高”订购偏差，并且在需求追逐等其他偏向方面明显放大，与人类基准相比尤为显著。我们的分析揭示了一种“智慧的悖论”：越复杂的GPT-4表现出越大的非理性，表现为过度思考，而以效率优化为目标的GPT-4o则几乎表现出最优。由于这些偏差即使在提供最优公式的情况下仍然存在，我们认为它们源于模型架构的限制而非知识上的不足。管理启示：首先，管理者应根据具体任务选择模型——我们的研究显示，效率优化的模型在某些优化问题上可以优于更复杂的模型。第二，LLMs显著放大的偏差凸显了在高风险决策中引入强有力的人机协作监督的迫切需求，以防止昂贵的错误。第三，我们的研究表明，设计结构化、规则导向的提示是限制模型启发式偏差、提升AI辅助决策可靠性的切实有效策略。

BibTeX

```
@article{2512.12552v1,
  title={Large Language Newsvendor: Decision Biases and Cognitive Mechanisms},
  author={Jifei Liu and Zhi Chen and Yuanguang Zhong},
  journal={arXiv preprint arXiv:2512.12552v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12552v1}
}
```

## [AI Transparency Atlas: Framework, Scoring, and Real-Time Model Card Evaluation Pipeline](http://arxiv.org/abs/2512.12443v1)

Akhmadillo Mamirov, Faiaz Azmain, Hanyu Wang

[PDF](https://arxiv.org/pdf/2512.12443v1)
[Abstract](http://arxiv.org/abs/2512.12443v1)

### 中文摘要

人工智能模型文档散见于多个平台，结构不统一，导致政策制定者、审核人员和用户难以可靠评估安全声明、数据来源及版本变更。我们分析了五款前沿模型（Gemini 3、Grok 4.1、Llama 4、GPT-5 和 Claude 4.5）以及100份Hugging Face模型卡，发现共有947个不同的章节名称，命名差异极大。仅使用信息使用方面的内容就出现了97个不同的标签。以欧盟AI法案附录IV和斯坦福透明度指数为基准，我们构建了一个包含8个部分、23个子部分的加权透明度评估框架，优先突出安全关键披露（安全评估占25%、关键风险占20%），而非技术规格。我们开发了一个自动化多智能体管道，从公开渠道提取文档，并利用大规模语言模型的共识对内容完整性进行评分。对50个模型（涵盖视觉、多模态、开源和闭源系统）的评估，总成本不足3美元，揭示了系统性差距。前沿实验室（xAI、微软、Anthropic）的合规率大约为80%，而大多数供应商低于60%。在安全关键类别中，表现出最大差距：误导行为、幻觉以及儿童安全评估，分别在所有评估模型中累计失分为148、124和116分。

BibTeX

```
@article{2512.12443v1,
  title={AI Transparency Atlas: Framework, Scoring, and Real-Time Model Card Evaluation Pipeline},
  author={Akhmadillo Mamirov and Faiaz Azmain and Hanyu Wang},
  journal={arXiv preprint arXiv:2512.12443v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12443v1}
}
```

## [DiffusionBrowser: Interactive Diffusion Previews via Multi-Branch Decoders](http://arxiv.org/abs/2512.13690v1)

Susung Hong, Chongjian Ge, Zhifei Zhang, Jui-Hsien Wang

[PDF](https://arxiv.org/pdf/2512.13690v1)
[Abstract](http://arxiv.org/abs/2512.13690v1)

### 中文摘要

视频扩散模型在生成式视频合成领域带来了革命性突破，但它们存在不够精确、速度较慢、生成过程不透明的问题，导致用户在长时间内无法掌握生成细节。在本工作中，我们提出了DiffusionBrowser，一种模型无关的轻量级解码器框架，允许用户在去噪过程中任意时刻（时间步骤或变换器块）交互式预览生成内容。我们的模型能够以超过4倍的实时时速（不足1秒即可生成一段4秒的视频）生成多模态预览表示，内容包括RGB图像和场景内在参数，确保预览与最终视频在外观和运动上的一致性。借助训练好的解码器，我们展示了通过引入随机性和模态引导，在中间噪声步骤动态指导生成的可能性，开启了一种新的控制方式。此外，我们还系统性地利用学习到的解码器对模型进行了探究，揭示了在原本黑箱式的去噪过程中，场景、物体及其他细节内容是如何被组合和组装的。

BibTeX

```
@article{2512.13690v1,
  title={DiffusionBrowser: Interactive Diffusion Previews via Multi-Branch Decoders},
  author={Susung Hong and Chongjian Ge and Zhifei Zhang and Jui-Hsien Wang},
  journal={arXiv preprint arXiv:2512.13690v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13690v1}
}
```

## [From Code to Field: Evaluating the Robustness of Convolutional Neural Networks for Disease Diagnosis in Mango Leaves](http://arxiv.org/abs/2512.13641v1)

Gabriel Vitorino de Andrade, Saulo Roberto dos Santos, Itallo Patrick Castro Alves da Silva, Emanuel Adler Medeiros Pereira, Erick de Andrade Barboza

[PDF](https://arxiv.org/pdf/2512.13641v1)
[Abstract](http://arxiv.org/abs/2512.13641v1)

### 中文摘要

通过鲁棒性评估对人工智能（AI）模型进行验证与确认，对于确保智能系统在面对现实世界挑战（如噪声、模糊和天气变化等图像损坏）时的可靠性能至关重要。尽管芒果（Mangifera indica L.）在全球具有重要价值，但关于其叶片疾病诊断模型鲁棒性的研究仍然缺乏。本文提出了一种在恶劣条件下评估卷积神经网络（CNN）的方法。我们对MangoleafDB数据集进行了改造，生成了包含19种人工损坏类型、五个严重程度等级的MangoleafDB-C。我们基准测试了五种架构：ResNet-50、ResNet-101、VGG-16、Xception以及专为芒果叶诊断设计的轻量级架构LCNN。评估指标包括F1得分、损坏错误（CE）以及相对平均损坏错误（相对mCE）。结果显示，LCNN在如散焦模糊、运动模糊等实际场景常见的损坏类型上优于复杂模型，并且实现了最低的mCE。尽管现代架构（如ResNet-101）在理想状态下表现出色，但在受损场景中性能明显下降。这些发现表明，轻量化和专用模型可能更适合在边缘设备的实际应用中使用，在此类环境中，鲁棒性和效率尤为关键。研究强调，在农业智能系统的开发过程中，融入鲁棒性评估以应对具有技术限制的地区尤为重要。

BibTeX

```
@article{2512.13641v1,
  title={From Code to Field: Evaluating the Robustness of Convolutional Neural Networks for Disease Diagnosis in Mango Leaves},
  author={Gabriel Vitorino de Andrade and Saulo Roberto dos Santos and Itallo Patrick Castro Alves da Silva and Emanuel Adler Medeiros Pereira and Erick de Andrade Barboza},
  journal={arXiv preprint arXiv:2512.13641v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13641v1}
}
```

## [Memory in the Age of AI Agents](http://arxiv.org/abs/2512.13564v1)

Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yanbin Yin, Jiongnan Liu, Zeyu Zhang, Zhongxiang Sun, Yutao Zhu, Hao Sun, Boci Peng, Zhenrong Cheng, Xuanbo Fan, Jiaxin Guo, Xinlei Yu, Zhenhong Zhou, Zewen Hu, Jiahao Huo, Junhao Wang, Yuwei Niu, Yu Wang, Zhenfei Yin, Xiaobin Hu, Yue Liao, Qiankun Li, Kun Wang, Wangchunshu Zhou, Yixin Liu, Dawei Cheng, Qi Zhang, Tao Gui, Shirui Pan, Yan Zhang, Philip Torr, Zhicheng Dou, Ji-Rong Wen, Xuanjing Huang, Yu-Gang Jiang, Shuicheng Yan

[PDF](https://arxiv.org/pdf/2512.13564v1)
[Abstract](http://arxiv.org/abs/2512.13564v1)

### 中文摘要

记忆已成为生成型基础模型代理的核心能力之一，并将在未来持续扮演重要角色。随着代理记忆研究的快速发展及其受到前所未有的关注，该领域也变得日益碎片化。现有属于代理记忆范畴的研究在动机、实现方式和评估协议等方面存在显著差异，而对记忆概念的定义也因用语模糊而加剧了概念的混乱。传统的长短期记忆等分类方法已无法充分描述当代代理记忆系统的多样性。本研究旨在提供当前代理记忆研究的最新全景。我们首先明确划定代理记忆的范围，并将其与相关概念如大型语言模型记忆、检索增强生成（RAG）以及上下文工程等区分开来。随后，利用形式、功能和动态三个统一视角对代理记忆进行分析。从形式角度，我们识别出三种主要的记忆实现方式：符号级、参数化和潜在记忆。从功能角度，我们提出更细粒度的分类，区分事实性记忆、体验性记忆和工作记忆。从动态角度，我们探讨记忆的形成、演变和检索机制。为了推动实际应用，我们整理了全面的记忆基准和开源框架总结。除了梳理现有研究，我们还展望了未来的研究前沿，包括记忆自动化、强化学习的集成、多模态记忆、多智能体记忆以及可信性等问题。希望本综述不仅能成为现有工作的参考，也能为未来智能体设计中将记忆定位为一等公民的思考提供理论基础。

BibTeX

```
@article{2512.13564v1,
  title={Memory in the Age of AI Agents},
  author={Yuyang Hu and Shichun Liu and Yanwei Yue and Guibin Zhang and Boyang Liu and Fangyi Zhu and Jiahang Lin and Honglin Guo and Shihan Dou and Zhiheng Xi and Senjie Jin and Jiejun Tan and Yanbin Yin and Jiongnan Liu and Zeyu Zhang and Zhongxiang Sun and Yutao Zhu and Hao Sun and Boci Peng and Zhenrong Cheng and Xuanbo Fan and Jiaxin Guo and Xinlei Yu and Zhenhong Zhou and Zewen Hu and Jiahao Huo and Junhao Wang and Yuwei Niu and Yu Wang and Zhenfei Yin and Xiaobin Hu and Yue Liao and Qiankun Li and Kun Wang and Wangchunshu Zhou and Yixin Liu and Dawei Cheng and Qi Zhang and Tao Gui and Shirui Pan and Yan Zhang and Philip Torr and Zhicheng Dou and Ji-Rong Wen and Xuanjing Huang and Yu-Gang Jiang and Shuicheng Yan},
  journal={arXiv preprint arXiv:2512.13564v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13564v1}
}
```

## [From User Interface to Agent Interface: Efficiency Optimization of UI Representations for LLM Agents](http://arxiv.org/abs/2512.13438v1)

Dezhi Ran, Zhi Gong, Yuzhe Guo, Mengzhou Wu, Yuan Cao, Haochuan Lu, Hengyu Zhang, Xia Zeng, Gang Cao, Liangchao Yao, Yuetang Deng, Wei Yang, Tao Xie

[PDF](https://arxiv.org/pdf/2512.13438v1)
[Abstract](http://arxiv.org/abs/2512.13438v1)

### 中文摘要

虽然大型语言模型（LLM）代理在自动化界面（UI）导航（如自动化UI测试和AI助手）方面展现出巨大潜力，但其效率问题却被广泛忽视。我们的研究发现，低效的界面表示方式成为关键的性能瓶颈。然而，将UI表示优化作为自动生成变换UI表示的程序任务，面临两大独特挑战。首先，缺乏布尔类的验证器（oracle），而传统程序合成依赖这些验证器来果断验证语义正确性，这对同时优化Token效率和表达完整性构成了根本性障碍。其次，在处理庞大且复杂的UI树作为输入，同时生成长且结构化的变换程序时，搜索空间庞大且易出错。为解决上述问题，我们提出了UIFormer，这是首个通过结构化分解复杂合成任务、基于约束优化的方法，实现UI变换程序自动合成的优化框架。UIFormer首先利用捕获UI特定操作的领域专属语言（DSL）限制程序空间；其次，采用基于LLM的迭代式细化策略，结合正确性和效率奖励，引导实现效率与完整性共优化。UIFormer作为一种轻量级插件，能够调用变换程序，实现无缝集成到现有的LLM代理中，仅需极少的核心逻辑调整。在涵盖Android与Web平台的三个UI导航基准测试，以及五种不同的LLM上进行的评估显示，UIFormer在保持或提升代理性能的同时，Token数量减少了48.7%至55.8%，且运行时开销极小。在微信的实际工业部署进一步验证了UIFormer的实用价值。

BibTeX

```
@article{2512.13438v1,
  title={From User Interface to Agent Interface: Efficiency Optimization of UI Representations for LLM Agents},
  author={Dezhi Ran and Zhi Gong and Yuzhe Guo and Mengzhou Wu and Yuan Cao and Haochuan Lu and Hengyu Zhang and Xia Zeng and Gang Cao and Liangchao Yao and Yuetang Deng and Wei Yang and Tao Xie},
  journal={arXiv preprint arXiv:2512.13438v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13438v1}
}
```

## [ALIGN-FL: Architecture-independent Learning through Invariant Generative component sharing in Federated Learning](http://arxiv.org/abs/2512.13316v1)

Mayank Gulati, Benedikt Groß, Gerhard Wunder

[PDF](https://arxiv.org/pdf/2512.13316v1)
[Abstract](http://arxiv.org/abs/2512.13316v1)

### 中文摘要

我们提出了一种新颖的分布式学习方法——ALIGN-FL，旨在解决来自高度异质数据分布的学习挑战，通过选择性共享生成组件实现数据隐私保护。与传统交换完整模型参数不同，我们的框架仅传输生成能力，以保证隐私安全，而服务器则利用合成样本进行全局训练。通过结合两种互补的隐私保护机制：具有自适应裁剪的差分隐私随机梯度下降（DP-SGD）以及具有Lipschitz正则化的变分自编码器（VAE）解码器和支持异构客户端的有状态架构，我们在包含跨域离群点的MNIST和Fashion-MNIST数据集上验证了该方法的有效性。实验结果表明，这两种隐私机制能有效将敏感的离群点映射为典型数据，同时在典型的跨机构合作中的极端非独立同分布（Non-IID）场景下保持模型的实用性。
关键词：客户端不变学习、联邦学习（FL）、隐私保护生成模型、非独立同分布（Non-IID）、异构架构

BibTeX

```
@article{2512.13316v1,
  title={ALIGN-FL: Architecture-independent Learning through Invariant Generative component sharing in Federated Learning},
  author={Mayank Gulati and Benedikt Groß and Gerhard Wunder},
  journal={arXiv preprint arXiv:2512.13316v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13316v1}
}
```

## [SACn: Soft Actor-Critic with n-step Returns](http://arxiv.org/abs/2512.13165v1)

Jakub Łyskawa, Jakub Lewandowski, Paweł Wawrzyński

[PDF](https://arxiv.org/pdf/2512.13165v1)
[Abstract](http://arxiv.org/abs/2512.13165v1)

### 中文摘要

软演员-批评家（SAC）在实际应用中被广泛采用，现已成为最具代表性的离策略在线无模型强化学习（RL）方法之一。与基于一阶回报的方法相比，采用 n 步回报技术能够显著加快RL算法的收敛速度。然而，由于动作分布的变化，通常将n步回报与SAC结合时会引入偏差，导致离策略算法难以有效融合该技术。虽然重要采样方法可以解决这一问题，即利用来自一个分布的样本估算另一个分布的期望值，但其可能引发数值不稳定问题。在本研究中，我们提出了一种结合SAC与n步回报的方案，有效克服了上述难题，并引入了数值稳定的重要采样方法，简化了超参数的设置。此外，我们还分析了在n步最大熵框架下，Soft Actor-Critic中熵估计的方法，提出了 τ 样本熵估计，以降低学习目标的方差。最后，我们设计了基于n步回报的软演员-批评家（SAC$n$）算法，并在MuJoCo模拟环境中进行了实验证明其有效性。

BibTeX

```
@article{2512.13165v1,
  title={SACn: Soft Actor-Critic with n-step Returns},
  author={Jakub Łyskawa and Jakub Lewandowski and Paweł Wawrzyński},
  journal={arXiv preprint arXiv:2512.13165v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13165v1}
}
```

## [Diffusion-Based Restoration for Multi-Modal 3D Object Detection in Adverse Weather](http://arxiv.org/abs/2512.13107v1)

Zhijian He, Feifei Liu, Yuwei Li, Zhanpeng Liu, Jintao Cheng, Xieyuanli Chen, Xiaoyu Tang

[PDF](https://arxiv.org/pdf/2512.13107v1)
[Abstract](http://arxiv.org/abs/2512.13107v1)

### 中文摘要

多模态三维目标检测对于机器人和自动驾驶中的可靠感知至关重要。然而，由于天气引起的畸变以及不同数据模态之间的错位，其在恶劣天气条件下的性能仍然受到限制。本论文提出了一种新颖的框架——DiffFusion，旨在通过基于扩散模型的图像恢复和自适应跨模态融合，提升在复杂天气环境下的鲁棒性。我们的核心观点是扩散模型具备强大的降噪和生成能力，能够适应各种天气条件。在此基础上，DiffFusion引入了Diffusion-IR，用于恢复受到天气影响退化的图像；同时，提出点云修复（PCR）模块，利用图像中的目标线索补偿受污染的激光雷达数据。为了应对两种模态之间的空间对齐问题，我们设计了双向自适应融合与对齐模块（BAFAM），实现动态多模态融合与双向鸟瞰图（BEV）对齐，保持空间的一致性。大量实验在三个公开数据集上表明，DiffFusion在恶劣天气条件下实现了最先进的鲁棒性，同时在干净数据环境中仍保持优异性能。在真实场景的DENSE数据集上的零样本测试结果进一步验证了其良好的泛化能力。本框架的实现将以开源形式发布。

BibTeX

```
@article{2512.13107v1,
  title={Diffusion-Based Restoration for Multi-Modal 3D Object Detection in Adverse Weather},
  author={Zhijian He and Feifei Liu and Yuwei Li and Zhanpeng Liu and Jintao Cheng and Xieyuanli Chen and Xiaoyu Tang},
  journal={arXiv preprint arXiv:2512.13107v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13107v1}
}
```

## [WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment](http://arxiv.org/abs/2512.12692v1)

Mahir Labib Dihan, Tanzima Hashem, Mohammed Eunus Ali, Md Rizwan Parvez

[PDF](https://arxiv.org/pdf/2512.12692v1)
[Abstract](http://arxiv.org/abs/2512.12692v1)

### 中文摘要

基于大型模型的智能体通常采用贪婪的逐步选择策略，依据当前观察结果单一地选择行动，而未考虑长远后果或其他潜在路径。这种缺乏前瞻性的行为在网页环境中特尤为突出，该环境具有部分可观测性——内容仅限于浏览器可见部分（如DOM和UI元素），而一次错误的操作常常需要复杂且脆弱的导航步骤才能逆转。若没有明确的回溯机制，智能体在纠正错误或系统性探索备用路径方面会遇到困难。树搜索方法提供了一种结构化探索的原则框架，但现有方法缺乏安全回溯的机制，容易产生意外副作用，并假设所有行动都是可逆的，忽视了不可逆操作的存在，从而在实际网页任务中其效果受限。为应对这些挑战，我们提出了WebOperator——一种支持可靠回溯和策略性探索的树搜索框架。该方法采用最佳优先搜索策略，基于奖励估计和安全性考量对行动进行排序，并配备了稳健的回溯机制，在重演先前路径前先验证其可行性，以防止不良副作用。为了进一步引导探索，WebOperator从多个多样化的推理上下文中生成行动候选，确保探索的多样性和鲁棒性，并通过过滤无效动作和合并语义等价动作，构建高质量的行动集。实验结果在WebArena和WebVoyager平台上验证了WebOperator的有效性。在WebArena上，WebOperator结合GPT-4O模型达到了54.6%的成功率，创下了新纪录，充分体现了将策略性前瞻与安全执行相结合的关键优势。

BibTeX

```
@article{2512.12692v1,
  title={WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment},
  author={Mahir Labib Dihan and Tanzima Hashem and Mohammed Eunus Ali and Md Rizwan Parvez},
  journal={arXiv preprint arXiv:2512.12692v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12692v1}
}
```

## [Mirror Mode in Fire Emblem: Beating Players at their own Game with Imitation and Reinforcement Learning](http://arxiv.org/abs/2512.11902v1)

Yanna Elizabeth Smid, Peter van der Putten, Aske Plaat

[PDF](https://arxiv.org/pdf/2512.11902v1)
[Abstract](http://arxiv.org/abs/2512.11902v1)

### 中文摘要

敌方策略在回合制游戏中应具有出其不意和难以预测的特点。本研究引入了一种新型游戏模式——镜像模式（Mirror Mode），在该模式中，敌方AI模仿玩家的个人策略，从而激励玩家不断调整自己的游戏玩法。我们基于Unity平台开发了《火焰之纹章：英雄传》这款简化版的策略视频游戏，包含标准模式和镜像模式。第一组实验旨在选择合适的模型，以模仿玩家示范行为，实验中结合了强化学习（Reinforcement Learning）与模仿学习（Imitation Learning），采用生成对抗模仿学习（Generative Adversarial Imitation Learning）、行为克隆（Behavioral Cloning）和近端策略优化（Proximal Policy Optimization）的方法。第二组实验通过玩家测试来评估所构建的模型，模型均以参与者提供的示范数据进行训练。测试结果显示，模型在防御行为方面模仿效果良好，但在进攻策略方面尚有提升空间。问卷调查显示，玩家能够识别自己采用的撤退策略，整体上对镜像模式的满意度更高。当模型得到进一步优化后，有望提升模仿质量，并增强玩家的体验，尤其是在面对自己的策略时。完整的代码和问卷结果已存放于：https://github.com/YannaSmid/MirrorMode

BibTeX

```
@article{2512.11902v1,
  title={Mirror Mode in Fire Emblem: Beating Players at their own Game with Imitation and Reinforcement Learning},
  author={Yanna Elizabeth Smid and Peter van der Putten and Aske Plaat},
  journal={arXiv preprint arXiv:2512.11902v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11902v1}
}
```

## [Intrinsic-Motivation Multi-Robot Social Formation Navigation with Coordinated Exploration](http://arxiv.org/abs/2512.13293v2)

Hao Fu, Wei Liu, Shuai Zhou

[PDF](https://arxiv.org/pdf/2512.13293v2)
[Abstract](http://arxiv.org/abs/2512.13293v2)

### 中文摘要

本文研究了强化学习（RL）在多机器人社会队形导航中的应用，这是一项实现人机无缝共处的关键能力。尽管RL提供了一种具有潜力的范式，但行人行为本身的不可预测性和常常表现出不配合的动态特性，给机器人之间的协同探索效率带来了巨大挑战。为此，本文提出了一种新颖的协同探索多机器人强化学习算法——引入了内在动机探索机制。该算法的核心部分是一个自我学习的内在奖励机制，旨在共同缓解策略的保守性。此外，该算法在集中训练与分散执行的框架中结合了双重采样方式，以增强对导航策略和内在奖励的表现能力，并采用两时间尺度更新规则，以解耦参数更新过程。实验证明，该算法在社会队形导航基准测试中优于现有的最先进方法，在关键指标上表现出色。我们的代码和视频演示可在：https://github.com/czxhunzi/CEMRRL 获取。

BibTeX

```
@article{2512.13293v2,
  title={Intrinsic-Motivation Multi-Robot Social Formation Navigation with Coordinated Exploration},
  author={Hao Fu and Wei Liu and Shuai Zhou},
  journal={arXiv preprint arXiv:2512.13293v2},
  year={2025},
  url={http://arxiv.org/abs/2512.13293v2}
}
```

## [Efficient Adaptive Rejection Sampling for Accelerating Speculative Decoding in Large Language Models](http://arxiv.org/abs/2512.13194v2)

Chendong Sun, mingmin Chen, Lei Xu

[PDF](https://arxiv.org/pdf/2512.13194v2)
[Abstract](http://arxiv.org/abs/2512.13194v2)

### 中文摘要

推测性解码（Speculative Decoding）是一种用于加速大规模语言模型（LLMs）自回归推理的主要技术。它通过使用一个快速的草稿模型提出候选词序列，并由一个大型目标模型并行验证。然而，其核心组件——拒绝采样机制——依赖于一个固定的、与上下文无关的随机阈值。这在高不确定性生成场景中会带来显著的“随机拒绝”问题，即合理的候选词因随机因素而频繁被拒绝，从而降低推理效率。本文提出了高效自适应拒绝采样（EARS）方法，该方法通过引入目标模型自身的预测不确定性（以1 - 最大预测概率P\_target衡量）动态调整接受阈值。利用与不确定性成比例的容差项，EARS可以在模型不确定时智能放宽接受标准，从而有效减少随机拒绝，同时在模型置信时保持严格要求。在创意写作和开放域问答任务中的实验结果表明，EARS显著提升了推测性解码的效率，在GSM8K基准测试中实现了最高18.12%的吞吐量提升，精度仅下降0.84%。该方法无需修改模型结构，能无缝集成到现有的推测性解码框架中。

BibTeX

```
@article{2512.13194v2,
  title={Efficient Adaptive Rejection Sampling for Accelerating Speculative Decoding in Large Language Models},
  author={Chendong Sun and mingmin Chen and Lei Xu},
  journal={arXiv preprint arXiv:2512.13194v2},
  year={2025},
  url={http://arxiv.org/abs/2512.13194v2}
}
```

## [Behavior and Representation in Large Language Models for Combinatorial Optimization: From Feature Extraction to Algorithm Selection](http://arxiv.org/abs/2512.13374v1)

Francesca Da Ros, Luca Di Gaspero, Kevin Roitero

[PDF](https://arxiv.org/pdf/2512.13374v1)
[Abstract](http://arxiv.org/abs/2512.13374v1)

### 中文摘要

近年来，大型语言模型（LLMs）的发展为优化自动化开辟了新前景。尽管已有多项研究探讨了LLMs如何生成或求解优化模型，但对于这些模型在问题结构或算法行为方面的实际学习内容仍知之甚少。本研究旨在分析LLMs如何在内部表征组合优化问题，以及此类表征是否能够支持后续的决策任务。我们采用双重方法：一方面通过直接查询评估LLMs明确提取实例特征的能力，另一方面通过探测分析检验这些信息是否隐含编码在模型的隐藏层中。探测框架还被拓展应用于逐个实例的算法选择任务，以评估基于LLM表征是否能有效预测表现最佳的求解器。实验涵盖四个基准问题和三种实例表示方法。结果表明，LLMs在从问题实例中恢复特征信息方面表现出中等能力，无论是通过直接查询还是探测方法。值得注意的是，LLM隐藏层表征的预测能力与传统特征提取方法相当，表明LLMs能够捕捉与优化性能密切相关的重要结构信息。

BibTeX

```
@article{2512.13374v1,
  title={Behavior and Representation in Large Language Models for Combinatorial Optimization: From Feature Extraction to Algorithm Selection},
  author={Francesca Da Ros and Luca Di Gaspero and Kevin Roitero},
  journal={arXiv preprint arXiv:2512.13374v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13374v1}
}
```

## [Can AI Understand What We Cannot Say? Measuring Multilevel Alignment Through Abortion Stigma Across Cognitive, Interpersonal, and Structural Levels](http://arxiv.org/abs/2512.13142v2)

Anika Sharma, Malavika Mampally, Chidaksh Ravuru, Kandyce Brennan, Neil Gaikwad

[PDF](https://arxiv.org/pdf/2512.13142v2)
[Abstract](http://arxiv.org/abs/2512.13142v2)

### 中文摘要

随着大型语言模型（日益成为偏见健康决策的中介工具，其在真正理解复杂心理与生理现象方面的能力仍然缺乏充分评估。人工智能是否能理解我们无法表达的内容？我们探讨大型语言模型是否在其运作的认知、 interpersonal和结构层面上连贯地表现出对堕胎歧视的理解。我们使用经过验证的个人层级堕胎歧视量表（ILAS）系统性测试了五个领先的大型语言模型，对627个具有人口多样性的虚拟用户进行了评估。多层次分析考察了模型是否在认知（自我评价）、人际（预期的他评与孤立）和结构（社区谴责与信息披露模式）层面，以及整体歧视方面，连贯地呈现出歧视现象。结果显示，各模型在所有层面上都未能实现真正的理解：它们高估了人际层面的歧视，低估了认知层面的歧视；假设社区谴责具有一致性；引入了在人类验证数据中不存在的人口偏见；未能捕捉到实证验证的歧视与隐私关系；甚至在理论框架内出现自相矛盾的现象。这些Pattern表明，现行的对齐方法虽能确保模型输出符合恰当的语言表达，但并未实现多层次的连贯理解。本研究提供了实证证据，表明当前的大型语言模型缺乏对心理和生理构念的连贯多层次理解。在高风险场景下的AI安全，亟需新的设计（实现多层次的连贯性）、评估（持续审查）、治理与监管（强制审查、责任追究、部署限制），以及在理解“人们无法表达的内容”对支持的成效是否有益或有害方面，提升相关领域的AI素养。

BibTeX

```
@article{2512.13142v2,
  title={Can AI Understand What We Cannot Say? Measuring Multilevel Alignment Through Abortion Stigma Across Cognitive, Interpersonal, and Structural Levels},
  author={Anika Sharma and Malavika Mampally and Chidaksh Ravuru and Kandyce Brennan and Neil Gaikwad},
  journal={arXiv preprint arXiv:2512.13142v2},
  year={2025},
  url={http://arxiv.org/abs/2512.13142v2}
}
```

## [Solving Parallel Machine Scheduling With Precedences and Cumulative Resource Constraints With Calendars](http://arxiv.org/abs/2512.11864v1)

Christoph Einspieler, Matthias Horn, Marie-Louise Lackner, Patrick Malik, Nysret Musliu, Felix Winter

[PDF](https://arxiv.org/pdf/2512.11864v1)
[Abstract](http://arxiv.org/abs/2512.11864v1)

### 中文摘要

在大多数工业制造领域中，寻找高效的多机并行作业调度方案是一项具有挑战性的任务。由于现代工厂的规模庞大，通过自动化调度技术最大限度地降低生产成本具有巨大潜力。过去，针对许多不同类型的机械调度问题，研究者已经提出了多种解法，然而即使是一些基础变体也被证明是NP-hard问题。然而，在当今的实际生产环境中，额外引入了复杂的前置关系约束、资源限制及日历约束，这些约束必须得到满足。此外，现有的解题技术在处理这些附加约束时效率有限。因此，迫切需要开发和分析能够解决此类实际多机调度场景的自动化方法。
本文引入了一种结合作业前置关系和基于日历的资源累计约束的多机调度新变体，这种问题在实际工业应用中较为常见。我们提出了一种约束建模方法，作为用于小型调度问题的精确解法，并结合最先进的约束求解技术。同时，本文还设计了一种启发式构造方法以及基于局部搜索的定制元启发式算法，以高效处理大规模问题实例。这一元启发式方法已在工业场景中部署使用，目前正在实际应用中发挥作用。

BibTeX

```
@article{2512.11864v1,
  title={Solving Parallel Machine Scheduling With Precedences and Cumulative Resource Constraints With Calendars},
  author={Christoph Einspieler and Matthias Horn and Marie-Louise Lackner and Patrick Malik and Nysret Musliu and Felix Winter},
  journal={arXiv preprint arXiv:2512.11864v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11864v1}
}
```

## [A Geometric Theory of Cognition](http://arxiv.org/abs/2512.12225v1)

Laha Ale

[PDF](https://arxiv.org/pdf/2512.12225v1)
[Abstract](http://arxiv.org/abs/2512.12225v1)

### 中文摘要

人类认知涵盖感知、记忆、直觉判断、深思熟虑、行动选择与社会推理，然而这些能力常被用不同的计算模型加以解释。在本文中，我们提出了一个统一的数学框架，在该框架中，各种认知过程源于一个单一的几何原则。我们将认知状态表示为一个可微流形上的点，并配备一个经过学习的黎曼度量，该度量编码了表征约束、计算成本以及认知变量之间的结构关系。一个标量认知势结合了预测准确性、结构的简约性、任务的效用以及规范或逻辑上的要求。认知的展开过程被描述为该认知势的黎曼梯度流，提供了一种普遍的动力学规律，从而产生了广泛的心理现象。经典的双过程效应——即快速的直觉反应与较慢的深思熟虑——自然地由度量引起的各向异性导致的内在时间尺度差异和几何相变产生，无需借助模块化或混合架构。我们推导了这些状态的解析条件，并通过模拟典型认知任务展示了其行为特征。综上，这些研究成果奠定了认知的几何基础，并为开发更具普遍性和类人智能的人工智能系统提供了指导原则。

BibTeX

```
@article{2512.12225v1,
  title={A Geometric Theory of Cognition},
  author={Laha Ale},
  journal={arXiv preprint arXiv:2512.12225v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12225v1}
}
```

## [Floorplan2Guide: LLM-Guided Floorplan Parsing for BLV Indoor Navigation](http://arxiv.org/abs/2512.12177v1)

Aydin Ayanzadeh, Tim Oates

[PDF](https://arxiv.org/pdf/2512.12177v1)
[Abstract](http://arxiv.org/abs/2512.12177v1)

### 中文摘要

室内导航对于视障人士而言仍然是一项关键挑战。现有的解决方案主要依赖于基础设施系统，限制了其在动态环境中安全导航的能力。我们提出了一种新颖的导航方法，利用基础模型将平面图转化为可导航的知识图谱，并生成人类可读的导航指令。Floorplan2Guide 将大型语言模型（LLM）集成到系统中，以提取建筑布局中的空间信息，减少了以往平面图解析方法所需的人工预处理步骤。实验结果表明，在模拟和实际环境评估中，少样本学习（few-shot learning）相较零样本学习（zero-shot learning）能显著提升导航精度。在所测试的模型中，Claude 3.7 Sonnet 在MP-1平面图的5次引导下，短、中、长路线的准确率分别达到92.31%、76.92%和61.54%，表现最佳。基于图的空间结构方法在所有模型中比直接视觉推理的成功率高出15.4%，这一结果验证了图形表示和上下文学习能够增强导航性能，从而使我们的方法在盲人及低视力（Blind and Low Vision, BLV）用户的室内导航中更加精准。

BibTeX

```
@article{2512.12177v1,
  title={Floorplan2Guide: LLM-Guided Floorplan Parsing for BLV Indoor Navigation},
  author={Aydin Ayanzadeh and Tim Oates},
  journal={arXiv preprint arXiv:2512.12177v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12177v1}
}
```

## [Modular and Multi-Path-Aware Offline Benchmarking for Mobile GUI Agents](http://arxiv.org/abs/2512.12634v1)

Youngmin Im, Byeongung Jo, Jaeyoung Wi, Seungwoo Baek, Tae Hoon Min, Joo Hyung Lee, Sangeun Oh, Insik Shin, Sunjae Lee

[PDF](https://arxiv.org/pdf/2512.12634v1)
[Abstract](http://arxiv.org/abs/2512.12634v1)

### 中文摘要

移动图形用户界面（GUI）代理，即能够代表用户与移动应用交互的智能代理，有望彻底改变人机交互方式。然而，当前关于GUI代理的评估方法存在两大根本性局限性。首先，评估通常要么依赖于单路径的离线基准测试，要么采用在线实时评测。离线基准使用静态、单路径的带注释数据集，这种方式不公平地惩罚了其他有效的备选操作；而在线评测由于其动态且不可预测的特性，导致规模难以扩展且难以保证结果的可重复性。其次，现有基准将代理视为一个单一的黑箱，忽略了各个组成部分的贡献，这常常导致不公平的比较，或无法揭示关键的性能瓶颈。为了解决这些问题，我们提出了MobiBench——首个面向移动GUI代理的模块化、多路径感知的离线评测框架，实现了高保真度、可扩展性和可重复性的离线评估。实验结果显示，MobiBench的评估结果与人工评估者的一致性达94.72%，与精心设计的在线基准相当，同时保持了静态离线基准的可扩展性和可重复性。此外，我们通过全面的模块级分析揭示了多项关键见解，包括对移动GUI代理中采用的多样技术的系统性评估、不同模型规模下的最优模块配置、当前大规模语言模型的固有限制，以及面向更强大且成本更优的移动代理设计的指导原则。

BibTeX

```
@article{2512.12634v1,
  title={Modular and Multi-Path-Aware Offline Benchmarking for Mobile GUI Agents},
  author={Youngmin Im and Byeongung Jo and Jaeyoung Wi and Seungwoo Baek and Tae Hoon Min and Joo Hyung Lee and Sangeun Oh and Insik Shin and Sunjae Lee},
  journal={arXiv preprint arXiv:2512.12634v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12634v1}
}
```

## [World Models Unlock Optimal Foraging Strategies in Reinforcement Learning Agents](http://arxiv.org/abs/2512.12548v1)

Yesid Fonseca, Manuel S. Ríos, Nicanor Quijano, Luis F. Giraldo

[PDF](https://arxiv.org/pdf/2512.12548v1)
[Abstract](http://arxiv.org/abs/2512.12548v1)

### 中文摘要

补丁觅食涉及有意识且有计划的过程，旨在确定最合适的时机离开资源丰富的区域，并探索潜在更有益的替代地点。边际价值定理（MVT）常被用来描述这一过程，提供了关于此类觅食行为的最优性模型。尽管该模型已被广泛应用于行为生态学中的预测，但在生物觅食者中促使最优补丁觅食决策出现的计算机制仍在研究中。本文表明，配备有学习型世界模型的人工觅食者自然会趋向符合MVT的策略。我们采用一种基于模型的强化学习智能体，该智能体通过获得简洁的预测性表征来理解环境，证明了预见能力——而非单纯的奖励最大化——是推动高效补丁离开行为的关键。与传统的无模型强化学习智能体相比，这些基于模型的智能体展现出与许多生物体类似的决策模式，暗示预测性世界模型可以成为AI系统中更具可解释性和生物学基础决策的基础。总体而言，我们的研究突显了生态学最优性原则在推动可解释性与适应性AI发展中的价值。

BibTeX

```
@article{2512.12548v1,
  title={World Models Unlock Optimal Foraging Strategies in Reinforcement Learning Agents},
  author={Yesid Fonseca and Manuel S. Ríos and Nicanor Quijano and Luis F. Giraldo},
  journal={arXiv preprint arXiv:2512.12548v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12548v1}
}
```

## [Forgetful but Faithful: A Cognitive Memory Architecture and Benchmark for Privacy-Aware Generative Agents](http://arxiv.org/abs/2512.12856v1)

Saad Alqithami

[PDF](https://arxiv.org/pdf/2512.12856v1)
[Abstract](http://arxiv.org/abs/2512.12856v1)

### 中文摘要

随着生成型智能体在复杂交互场景中的应用日益增强，其记忆管理能力成为影响系统性能和隐私保护的关键瓶颈。目前的方法要么采用无限存储的策略，导致计算难题和隐私风险，要么使用简单的遗忘机制，牺牲了智能体的连贯性和功能性。本文提出了一种面向人类中心的记忆管理新框架——记忆感知保留方案（MaRS），并配备六种具有理论依据的遗忘策略，以在性能、隐私保护和计算效率之间实现平衡。我们还提出了“遗忘但忠实的智能体（FiFA）”基准测评体系，这一全面的评估框架涵盖叙事连贯性、目标完成度、社会回忆的准确性、隐私保障和成本效率等指标。通过涉及多种记忆预算和智能体配置的300次广泛实验，我们验证了我们的混合遗忘策略在保持计算可行性的同时，取得了卓越的整体表现（综合得分：0.911）。本研究为记忆预算型智能体的评估树立了新标杆，并为在资源有限、隐私敏感的环境中部署生成型智能体提供了实用指导。这些理论基础、实现框架和实验结果共同推进了以人为本的人工智能领域的发展，解决了影响用户信任、系统扩展性和合规性的核心记忆管理挑战。

BibTeX

```
@article{2512.12856v1,
  title={Forgetful but Faithful: A Cognitive Memory Architecture and Benchmark for Privacy-Aware Generative Agents},
  author={Saad Alqithami},
  journal={arXiv preprint arXiv:2512.12856v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12856v1}
}
```

## [Intrinsic Image Fusion for Multi-View 3D Material Reconstruction](http://arxiv.org/abs/2512.13157v1)

Peter Kocsis, Lukas Höllein, Matthias Nießner

[PDF](https://arxiv.org/pdf/2512.13157v1)
[Abstract](http://arxiv.org/abs/2512.13157v1)

### 中文摘要

我们提出了一种内在图像融合方法，能够从多视角图像中重建高质量的基于物理的材质模型。材质重建本身具有高度欠定性，通常依赖分析-合成（analysis-by-synthesis）的方法，而这种方法需要昂贵且带有噪声的路径追踪。为了更有效地限制优化过程，我们在重建过程中引入了单视图先验信息。具体地，我们利用一种基于扩散的材质估计器，该估计器每个视角会产生多个候选分解结果，但这些结果往往不一致。为了减少不一致性，我们对这些预测结果拟合了一种明确的低维参数函数。随后，我们提出了一种鲁棒的优化框架，结合软性每视图预测选择和基于置信度的多视图软内点集，将最具一致性且最具置信度的视图预测融合成一个一致的参数化材质空间。最终，我们通过逆路径追踪方法，优化低维参数。在合成场景和真实场景中进行的实验结果显示，该方法在材质解缠方面优于现有最先进的技术，能够生成锐利、干净的重建结果，适用于高质量的光照重放。

BibTeX

```
@article{2512.13157v1,
  title={Intrinsic Image Fusion for Multi-View 3D Material Reconstruction},
  author={Peter Kocsis and Lukas Höllein and Matthias Nießner},
  journal={arXiv preprint arXiv:2512.13157v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13157v1}
}
```

## [Towards Open Standards for Systemic Complexity in Digital Forensics](http://arxiv.org/abs/2512.12970v1)

Paola Di Maio

[PDF](https://arxiv.org/pdf/2512.12970v1)
[Abstract](http://arxiv.org/abs/2512.12970v1)

### 中文摘要

人工智能（AI）与数字取证（DF）交叉领域正变得日益复杂、普及和无处不在，相关的技术和方法已广泛应用于各种科学与技术研究中。尽管取得了令人瞩目的进展，取证科学仍无法避免错误的发生，且依然存在易错的风险。为应对数字取证中错误带来的局限性，本文识别并解决了系统复杂性的问题，通过采用人可读的文档和开源标准进行应对。本文还提出了一种基于最新技术的数字取证AI模型框架。

BibTeX

```
@article{2512.12970v1,
  title={Towards Open Standards for Systemic Complexity in Digital Forensics},
  author={Paola Di Maio},
  journal={arXiv preprint arXiv:2512.12970v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12970v1}
}
```

## [Feedforward 3D Editing via Text-Steerable Image-to-3D](http://arxiv.org/abs/2512.13678v1)

Ziqi Ma, Hongqiao Chen, Yisong Yue, Georgia Gkioxari

[PDF](https://arxiv.org/pdf/2512.13678v1)
[Abstract](http://arxiv.org/abs/2512.13678v1)

### 中文摘要

近年来，图像到三维（image-to-3D）技术的进展为设计、增强现实/虚拟现实（AR/VR）以及机器人学等领域带来了巨大潜力。然而，若要将AI生成的三维模型应用于实际场景，关键之一在于能够方便地对其进行编辑。本文提出了一种前馈式方法——Steer3D，旨在为图像到三维模型引入文本引导能力，从而实现基于语言对生成的三维资产进行修改。我们的方法受到ControlNet的启发，并对其进行了适配，允许在一次前向计算中实现文本引导的三维生成。我们构建了一个可扩展的自动数据生成引擎，并提出了基于流匹配训练（flow-matching）与直接偏好优化（DPO）的两阶段训练方案。与现有方法相比，Steer3D在更忠实于文本指令的同时，更能保持与原始三维模型的一致性，且运行速度提升了2.4到28.5倍。实验结果表明，借助约10万个数据，Steer3D能够成功实现引入文本模态，以引导预训练图像到三维生成模型的生成过程。项目网址：https://glab-caltech.github.io/steer3d/

BibTeX

```
@article{2512.13678v1,
  title={Feedforward 3D Editing via Text-Steerable Image-to-3D},
  author={Ziqi Ma and Hongqiao Chen and Yisong Yue and Georgia Gkioxari},
  journal={arXiv preprint arXiv:2512.13678v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13678v1}
}
```

## [Embedding-Based Rankings of Educational Resources based on Learning Outcome Alignment: Benchmarking, Expert Validation, and Learner Performance](http://arxiv.org/abs/2512.13658v1)

Mohammadreza Molavi, Mohammad Moein, Mohammadreza Tavakoli, Abdolali Faraji, Stefan T. Mol, Gábor Kismihók

[PDF](https://arxiv.org/pdf/2512.13658v1)
[Abstract](http://arxiv.org/abs/2512.13658v1)

### 中文摘要

随着在线学习环境的不断发展，个性化的需求日益凸显。虽然教育资源不断丰富，但教育工作者在选择既能实现预期学习目标又能满足不同学习者需求的材料时仍面临挑战。大型语言模型（LLMs）由于具有生成更贴合个性化支持的学习资源的潜力，受到越来越多的关注，但验证这些资源是否覆盖预期目标仍需人工进行内容符合性审查，这既耗费人力又限制了推广规模。我们提出了一种框架，支持以低成本实现教育资源与预定学习目标之间符合度的自动化评估。通过使用人工生成的材料对基于LLM的文本嵌入模型进行了基准测试，发现最为准确的模型（Voyage）在检测符合度方面达到79%的准确率。随后，我们将这一最优模型应用于由LLM生成的资源，并通过专家评审确认其在评估内容与预期目标的一致性方面具有可靠性，准确率为83%。最后，在一项涉及360名学习者的三组实验中，较高的符合度评分与更好的学习表现呈正相关，卡方检验值为15.39（自由度2，样本数360），p<0.001。这些发现表明，基于嵌入的符合度评分可以促进规模化的个性化学习，验证其与学习目标的一致性，从而帮助教师将更多时间用于根据不同学习者的需求定制内容。

BibTeX

```
@article{2512.13658v1,
  title={Embedding-Based Rankings of Educational Resources based on Learning Outcome Alignment: Benchmarking, Expert Validation, and Learner Performance},
  author={Mohammadreza Molavi and Mohammad Moein and Mohammadreza Tavakoli and Abdolali Faraji and Stefan T. Mol and Gábor Kismihók},
  journal={arXiv preprint arXiv:2512.13658v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13658v1}
}
```

## [Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models](http://arxiv.org/abs/2512.13607v1)

Boxin Wang, Chankyu Lee, Nayeon Lee, Sheng-Chieh Lin, Wenliang Dai, Yang Chen, Yangyi Chen, Zhuolin Yang, Zihan Liu, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping

[PDF](https://arxiv.org/pdf/2512.13607v1)
[Abstract](http://arxiv.org/abs/2512.13607v1)

### 中文摘要

构建具有通用推理能力的模型依赖于强化学习（RL），这涉及到跨领域的巨大异质性，包括推理时响应长度和验证延迟的显著差异。这种变异性增加了RL基础架构的复杂性，放缓了训练速度，并使训练课程（如延长响应长度）和超参数的选择变得更加困难。在本文中，我们提出了级联系统的逐领域强化学习（Cascade RL）方法，以开发具有通用推理能力的模型——Nemotron-Cascade，该模型能够在指令理解和深度思考两种模式下运行。不同于传统方法将来自不同领域的异构提示进行混合，级联系统的RL通过逐领域、顺序执行，有效降低了工程复杂度，同时在多项基准测试中展现出最先进的性能。值得注意的是，将RLHF（偏好导向的强化学习）用于模型对齐作为预处理步骤，极大地提升了模型的推理能力，远超单纯的偏好优化，而后续的逐领域RLVR阶段很少削弱或甚至还会提升早期领域的基准表现（见图1示意）。我们14亿参数的模型在RL训练后，超越了其有监督微调（SFT）教师模型DeepSeek-R1-0528，在LiveCodeBench v5/v6/Pro上表现优异，并在2025年国际信息学奥林匹克（IOI）中获得银牌。我们坦诚分享了训练方案和数据配置，欢迎参考。

BibTeX

```
@article{2512.13607v1,
  title={Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models},
  author={Boxin Wang and Chankyu Lee and Nayeon Lee and Sheng-Chieh Lin and Wenliang Dai and Yang Chen and Yangyi Chen and Zhuolin Yang and Zihan Liu and Mohammad Shoeybi and Bryan Catanzaro and Wei Ping},
  journal={arXiv preprint arXiv:2512.13607v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13607v1}
}
```

## [ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](http://arxiv.org/abs/2512.13586v1)

Jia-Nan Li, Jian Guan, Wei Wu, Chongxuan Li

[PDF](https://arxiv.org/pdf/2512.13586v1)
[Abstract](http://arxiv.org/abs/2512.13586v1)

### 中文摘要

自回归模型（ARMs）在推理过程中受到较慢的序贯推断的制约。尽管掩码扩散模型（MDMs）提供了一种并行替代方案，但它们也存在关键缺陷：由于无法缓存关键值（KV），带来了较高的计算开销，以及在学习对不可解空间中的令牌组合依赖关系时导致生成结果不连贯。为了解决这些限制，我们提出了ReFusion，一种新型掩码扩散模型，通过将并行解码从令牌级提升到更高的槽（slot）级，其中每个槽是一个固定长度的连续子序列，从而实现了更优的性能和效率。这一改进通过一种迭代的“规划与填充”解码过程得以实现：首先进行基于扩散的规划步骤，识别出一组依赖较弱的槽；随后进行自回归的填充步骤，逐步并行解码这些选中的槽。基于槽的设计不仅实现了全KV缓存的复用，结合统一的因果关系框架，还将学习的复杂度从令牌组合空间降低到可管理的槽级排列空间。在七个不同的基准任务上进行的广泛实验显示，ReFusion不仅以34%的性能提升和平均18倍以上的速度提升彻底超越了现有MDMs，还缩小了与强大ARMs之间的性能差距，同时保持平均2.33倍的加速效果。

BibTeX

```
@article{2512.13586v1,
  title={ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding},
  author={Jia-Nan Li and Jian Guan and Wei Wu and Chongxuan Li},
  journal={arXiv preprint arXiv:2512.13586v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13586v1}
}
```

## [Error-Driven Prompt Optimization for Arithmetic Reasoning](http://arxiv.org/abs/2512.13323v1)

Árpád Pándy, Róbert Lakatos, András Hajdu

[PDF](https://arxiv.org/pdf/2512.13323v1)
[Abstract](http://arxiv.org/abs/2512.13323v1)

### 中文摘要

近年来，人工智能的快速发展激发了在金融、医疗等受监管行业中，能够支持分析师进行表格数据工作流程的工业智能代理的研究兴趣。这类系统的一个关键能力是能够在确保敏感信息绝不离开安全的本地环境的前提下，针对结构化数据进行精确的算术运算。为此，本文提出了一种基于错误驱动的算术推理优化框架，用于增强代码生成代理（CGA），特别适用于本地部署的小型语言模型（SLMs）。通过对主流小型语言模型（Qwen3 4B）的系统性评估发现，尽管基础模型在算术任务中存在根本性限制，我们提出的误差驱动方法——通过聚类错误预测以迭代优化提示规则——显著提升了模型性能，将准确率提升至70.8%。研究表明，构建可靠、可解释且具工业应用能力的AI助手，不仅仅依赖于昂贵的微调，还可以通过系统性、基于错误的提示优化，使得小型模型在保证隐私合规的前提下，超越更大规模的语言模型（如GPT-3.5 Turbo）。

BibTeX

```
@article{2512.13323v1,
  title={Error-Driven Prompt Optimization for Arithmetic Reasoning},
  author={Árpád Pándy and Róbert Lakatos and András Hajdu},
  journal={arXiv preprint arXiv:2512.13323v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13323v1}
}
```

## [MedInsightBench: Evaluating Medical Analytics Agents Through Multi-Step Insight Discovery in Multimodal Medical Data](http://arxiv.org/abs/2512.13297v1)

Zhenghao Zhu, Chuxue Cao, Sirui Han, Yuanfeng Song, Xing Chen, Caleb Chen Cao, Yike Guo

[PDF](https://arxiv.org/pdf/2512.13297v1)
[Abstract](http://arxiv.org/abs/2512.13297v1)

### 中文摘要

在医学数据分析中，从复杂的多模态数据集中提取深层次的见解对于改善患者护理、提升诊断准确率以及优化医疗运营具有重要意义。然而，目前尚缺乏专门设计的高质量数据集，用于评估大型多模态模型（LMMs）在发现医学洞见方面的能力。本文提出了MedInsightBench，这是首个包含332个经过精心筛选、每个都附有深思熟虑的洞见注释的医学案例的基准测试平台。该基准旨在评估LMMs及其代理框架分析多模态医学影像数据的能力，包括提出相关问题、解读复杂的诊断发现，以及整合可操作的洞见和建议。我们的分析结果表明，现有的LMMs在MedInsightBench上的表现有限，主要原因在于它们难以进行多步的深层次洞察提取，以及缺乏医学专业知识的支持。因此，我们提出了MedInsightAgent，一种用于医学数据分析的自动化代理框架，由视觉根寻器、分析性洞察代理和跟进问题生成三个模块组成。在MedInsightBench上的实验显示了这些挑战的普遍性，并证明了MedInsightAgent能够提升通用LMM在医学数据洞见获取方面的性能。

BibTeX

```
@article{2512.13297v1,
  title={MedInsightBench: Evaluating Medical Analytics Agents Through Multi-Step Insight Discovery in Multimodal Medical Data},
  author={Zhenghao Zhu and Chuxue Cao and Sirui Han and Yuanfeng Song and Xing Chen and Caleb Chen Cao and Yike Guo},
  journal={arXiv preprint arXiv:2512.13297v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13297v1}
}
```

## [CORE: Contrastive Masked Feature Reconstruction on Graphs](http://arxiv.org/abs/2512.13235v1)

Jianyuan Bo, Yuan Fang

[PDF](https://arxiv.org/pdf/2512.13235v1)
[Abstract](http://arxiv.org/abs/2512.13235v1)

### 中文摘要

在快速发展的图自监督学习领域，生成式方法和对比学习方法已成为两大主流。我们的研究聚焦于掩码特征重构（MFR），这是一种生成技术，模型通过自监督方式学习恢复被掩码节点的原始特征。我们观察到，MFR和图对比学习（GCL）都旨在最大化相似元素之间的一致性。在此基础上，我们揭示了一项新的理论观点：在特定条件下，尽管操作机制各不相同，MFR的目标与节点级GCL的目标会趋于一致。这一理论联系表明，这两种方法是互补的而非根本对立的，从而促使我们探索其融合以提升图的自监督学习能力。我们的研究提出了对比掩码特征重构（CORE）——一种将对比学习整合入MFR的创新图自监督学习框架。具体而言，我们仅在原始特征与重构特征的掩码节点之间构建正样本对，促使编码器更关注上下文信息而非节点本身的特征。此外，我们还将掩码节点本身用作负样本，结合MFR的重构能力与GCL的判别能力，更有效地捕捉图的固有结构。实验结果显示，所提框架CORE在节点分类和图分类任务中显著优于单纯的MFR，取得了最优表现。具体而言，CORE在节点分类任务中分别比GraphMAE和GraphMAE2高出最多2.80%和3.72%，在图分类任务中则高出最多3.82%和3.76%。

BibTeX

```
@article{2512.13235v1,
  title={CORE: Contrastive Masked Feature Reconstruction on Graphs},
  author={Jianyuan Bo and Yuan Fang},
  journal={arXiv preprint arXiv:2512.13235v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13235v1}
}
```

## [KidsArtBench: Multi-Dimensional Children's Art Evaluation with Attribute-Aware MLLMs](http://arxiv.org/abs/2512.12503v1)

Mingrui Ye, Chanjin Zheng, Zengyi Yu, Chenyu Xiang, Zhixue Zhao, Zheng Yuan, Helen Yannakoudakis

[PDF](https://arxiv.org/pdf/2512.12503v1)
[Abstract](http://arxiv.org/abs/2512.12503v1)

### 中文摘要

多模态大型语言模型（MLLMs）在多项视觉-语言任务中取得了显著进展；然而，它们在评估艺术表达方面的能力仍然有限。审美观念具有本质的抽象性和开放性，而描述多模态艺术作品的标注数据相对匮乏。我们推出了 KidsArtBench，这是一个新的基准数据集，包括超过1000件由5至15岁儿童创作的艺术作品，拥有由12位专家教育者基于九个与评分标准一致的维度进行的标注，并配有专家评论作为反馈。与以往提供单一标量评分的成人图像审美数据集不同，KidsArtBench以儿童作品为对象，将多维度标注与评论监督相结合，实现了序数评估与形成性反馈的结合。基于此资源，我们提出了一种面向属性的多LoRA（Multi-LoRA）方法，每个属性对应评分标准中的一个独特评价维度（如真实性、想象力），并采用了具有回归感知的微调（RAFT）技术，以使预测结果与序数尺度保持一致。在Qwen2.5-VL-7B模型上，我们的方法将相关性从0.468提升至0.653，尤其在感知维度上获得了最大改进，并在高阶属性上逐步缩小差距。这些结果表明，符合教育者标准的监督与属性感知的训练方式能够提供具有教育意义的评估，有助于推动教育人工智能的持续发展。我们同时发布了相关数据、代码及伦理说明文件。

BibTeX

```
@article{2512.12503v1,
  title={KidsArtBench: Multi-Dimensional Children's Art Evaluation with Attribute-Aware MLLMs},
  author={Mingrui Ye and Chanjin Zheng and Zengyi Yu and Chenyu Xiang and Zhixue Zhao and Zheng Yuan and Helen Yannakoudakis},
  journal={arXiv preprint arXiv:2512.12503v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12503v1}
}
```

## [SafeGen: Embedding Ethical Safeguards in Text-to-Image Generation](http://arxiv.org/abs/2512.12501v1)

Dang Phuong Nam, Nguyen Kieu, Pham Thanh Hieu

[PDF](https://arxiv.org/pdf/2512.12501v1)
[Abstract](http://arxiv.org/abs/2512.12501v1)

### 中文摘要

生成式人工智能（AI）为创造性表达、教育和科研带来了前所未有的机遇。诸如DALL.E、Stable Diffusion和Midjourney等文本转图像系统能够在数秒内将构思转化为视觉效果，但同时也引发了双重用途的伦理困境，诸如放大社会偏见、生成高保真的虚假信息以及侵犯知识产权等问题。本文提出了SafeGen框架，将伦理保障直接嵌入到文本转图像生成流程中，并以可信赖人工智能的既定原则为设计基础。SafeGen融合了两个互补的组件：一是经过微调的文本分类器BGE-M3，用于筛查有害或误导性的提示语；二是经过优化的扩散模型Hyper-SD，能够生成高保真、语义一致的图像。该框架基于多语种（英语-越南语）精心策划的数据集和公平性意识的训练过程，证明了创造自由与伦理责任可以在单一工作流程中共存。定量评估结果显示其卓越表现，Hyper-SD的inception Score（IS）为3.52，Fréchet Inception Distance（FID）为22.08，结构相似性指数（SSIM）为0.79，BGE-M3的F1-Score达到了0.81。此外，通过消融实验进一步验证了领域特定微调在两个模块中的关键作用。案例研究亦展示了SafeGen在阻挡不安全提示、生成包容性教学资料以及维护学术诚信方面的实际应用影响。

BibTeX

```
@article{2512.12501v1,
  title={SafeGen: Embedding Ethical Safeguards in Text-to-Image Generation},
  author={Dang Phuong Nam and Nguyen Kieu and Pham Thanh Hieu},
  journal={arXiv preprint arXiv:2512.12501v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12501v1}
}
```

## [MetaHGNIE: Meta-Path Induced Hypergraph Contrastive Learning in Heterogeneous Knowledge Graphs](http://arxiv.org/abs/2512.12477v1)

Jiawen Chen, Yanyan He, Qi Shao, Mengli Wei, Duxin Chen, Wenwu Yu, Yanlong Zhao

[PDF](https://arxiv.org/pdf/2512.12477v1)
[Abstract](http://arxiv.org/abs/2512.12477v1)

### 中文摘要

异构知识图中的节点重要性估计（Node Importance Estimation, NIE）是一项关键且富有挑战性的任务，对于推荐系统、知识推理和问答等应用具有重要意义。现有方法多依赖于节点之间的成对连接，忽视了多个实体及关系之间的高阶依赖关系，同时将结构信息与语义信息分开处理，限制了不同模态信息的有效融合。为了解决这些问题，本文提出了MetaHGNIE，一种基于元路径诱导的超图对比学习框架，旨在对结构信息和语义信息进行解耦和对齐。该框架通过元路径序列构建高阶知识图，其中类型化超边捕捉多实体间的关系环境；结构依赖关系通过局部注意力机制进行聚合，语义表示则采用配备稀疏块处理的超图Transformer编码，以减少冗余。最终，结合对比学习和辅助监督的多模态融合模块，有效实现了结构与语义编码的跨模态对齐。大量在标准NIE数据集上的实验证明，MetaHGNIE持续优于现有最优方法，验证了在异构知识图中显式建模高阶交互关系和跨模态对齐的有效性。我们的代码已开源，地址为 https://github.com/SEU-WENJIA/DualHNIE

BibTeX

```
@article{2512.12477v1,
  title={MetaHGNIE: Meta-Path Induced Hypergraph Contrastive Learning in Heterogeneous Knowledge Graphs},
  author={Jiawen Chen and Yanyan He and Qi Shao and Mengli Wei and Duxin Chen and Wenwu Yu and Yanlong Zhao},
  journal={arXiv preprint arXiv:2512.12477v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12477v1}
}
```

## [Value-Aware Multiagent Systems](http://arxiv.org/abs/2512.12652v1)

Nardine Osman

[PDF](https://arxiv.org/pdf/2512.12652v1)
[Abstract](http://arxiv.org/abs/2512.12652v1)

### 中文摘要

本文引入了人工智能中“价值意识”的概念，超越了传统的价值对齐问题。我们对价值意识的定义为工程化具有价值意识的人工智能提供了一条简洁而清晰的路线图。该路线图以三个核心支柱为基础：(1) 利用形式化语义学习和表征人类价值观；(2) 确保单个智能体及多智能体系统的价值对齐；(3) 提供基于价值的行为解释。论文展示了我们在这些主题上的部分持续研究工作，以及其在实际应用领域中的应用实例。

BibTeX

```
@article{2512.12652v1,
  title={Value-Aware Multiagent Systems},
  author={Nardine Osman},
  journal={arXiv preprint arXiv:2512.12652v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12652v1}
}
```

## [Feeling the Strength but Not the Source: Partial Introspection in LLMs](http://arxiv.org/abs/2512.12411v1)

Ely Hahami, Lavik Jain, Ishaan Sinha

[PDF](https://arxiv.org/pdf/2512.12411v1)
[Abstract](http://arxiv.org/abs/2512.12411v1)

### 中文摘要

近期，Anthropic公司声称前沿模型有时能够检测并命名作为激活方向注入的“概念”。我们对这些结论的鲁棒性进行了测试。首先，我们在Meta-Llama-3.1-8B-Instruct模型上复现了Anthropic关于“突现内省”的多轮对话结果，发现该模型在Anthropic原始流程下能准确识别并命名注入的概念，成功率为20%，与他们报道的数据完全一致，表明内省能力并非仅由极大或极强的模型所具备。其次，我们系统性地变换推理提示，发现内省能力较为脆弱：在与之密切相关的任务中表现显著下降，例如多项选择识别注入概念或不同提示下判断是否注入了概念的二分类任务。第三，我们发现一种对比性较强的部分内省状态：相同模型能以高达70%的准确率，可靠地对归一化注入概念向量的系数强度（如弱/中等/强/非常强）进行分类，显著高于随机猜测的25%。综上，这些结果为Anthropic的观点提供了更多证据，即语言模型在内省时实际上在计算其基础的内部表征的某种函数；但这些模型对自身表征的自我报告仍然有限、对提示敏感。这些研究代码已开放，地址为https://github.com/elyhahami18/CS2881-Introspection。

BibTeX

```
@article{2512.12411v1,
  title={Feeling the Strength but Not the Source: Partial Introspection in LLMs},
  author={Ely Hahami and Lavik Jain and Ishaan Sinha},
  journal={arXiv preprint arXiv:2512.12411v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12411v1}
}
```

## [Log Anomaly Detection with Large Language Models via Knowledge-Enriched Fusion](http://arxiv.org/abs/2512.11997v1)

Anfeng Peng, Ajesh Koyatan Chathoth, Stephen Lee

[PDF](https://arxiv.org/pdf/2512.11997v1)
[Abstract](http://arxiv.org/abs/2512.11997v1)

### 中文摘要

系统日志是监控和管理分布式系统的重要资源，能够提供关于故障和异常行为的深刻洞察。传统的日志分析方法，包括基于模板的分析和基于序列的方法，常常会丢失关键的语义信息或难以处理模糊的日志模式。为了解决这些问题，我们提出了EnrichLog，一种无需训练、基于条目的异常检测框架，能够利用语料库特定和样本特定的知识对原始日志条目进行丰富。EnrichLog结合了上下文信息，包括历史示例和由语料库推导出的推理，从而实现更精准且具有可解释性的异常检测。该框架采用增强检索的生成技术，有效整合相关的上下文知识，而无需进行额外的模型再训练。我们在四个大规模系统日志基准数据集上对EnrichLog进行了评估，并与五个基线方法进行了比较。结果显示，EnrichLog在异常检测性能上持续提升，能够有效处理模糊的日志条目，并保持高效的推理速度。此外，结合语料库和样本特定的知识不仅增强了模型的信心，还提升了检测的准确性，使得EnrichLog非常适合实际部署应用。

BibTeX

```
@article{2512.11997v1,
  title={Log Anomaly Detection with Large Language Models via Knowledge-Enriched Fusion},
  author={Anfeng Peng and Ajesh Koyatan Chathoth and Stephen Lee},
  journal={arXiv preprint arXiv:2512.11997v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11997v1}
}
```

## [AGAPI-Agents: An Open-Access Agentic AI Platform for Accelerated Materials Design on AtomGPT.org](http://arxiv.org/abs/2512.11935v1)

Jaehyung Lee, Justin Ely, Kent Zhang, Akshaya Ajith, Charles Rhys Campbell, Kamal Choudhary

[PDF](https://arxiv.org/pdf/2512.11935v1)
[Abstract](http://arxiv.org/abs/2512.11935v1)

### 中文摘要

人工智能正在重塑科学发现，但在材料研究中的应用仍受限于碎片化的计算体系、可复现性挑战以及对商业大型语言模型（LLMs）的依赖。在此，我们推出了AGAPI（AtomGPT.org API）——一个开源的智能AI平台，集成了超过八个开源大型语言模型和二十余个材料科学API接口，通过一个统一的调度框架整合了数据库、模拟工具和机器学习模型。AGAPI采用“Agent-规划器-执行器-总结器”架构，能够自主构建并执行涵盖材料数据检索、图神经网络性质预测、机器学习力场优化、紧束缚计算、衍射分析及逆向设计等多步骤工作流程。我们通过端到端的工作流程示范了AGAPI的应用，包括异质结构构建、粉末X射线衍射分析以及半导体缺陷工程，整个过程最多需要十个连续操作。此外，我们利用30多个示例提示作为测试案例，对AGAPI的预测效果进行了评估，并将智能体在有无工具访问的情况下的预测结果与实验数据进行了对比。目前，AGAPI已拥有超过1000名活跃用户，为可重复、AI加速的材料发现提供了一个可扩展且透明的基础。AGAPI-Agents的代码库可在https://github.com/atomgptlab/agapi 获取。

BibTeX

```
@article{2512.11935v1,
  title={AGAPI-Agents: An Open-Access Agentic AI Platform for Accelerated Materials Design on AtomGPT.org},
  author={Jaehyung Lee and Justin Ely and Kent Zhang and Akshaya Ajith and Charles Rhys Campbell and Kamal Choudhary},
  journal={arXiv preprint arXiv:2512.11935v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11935v1}
}
```

## [Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](http://arxiv.org/abs/2512.11907v1)

Daniel Platnick, Marjan Alirezaie, Hossein Rahnama

[PDF](https://arxiv.org/pdf/2512.11907v1)
[Abstract](http://arxiv.org/abs/2512.11907v1)

### 中文摘要

个性化大型语言模型（LLM）代理需要根据用户特定数据进行条件设计，这在任务效用与数据泄露之间造成了关键的权衡。虽然引入用户数据通常会带来递减的边际收益（即次模性质），从而可以采用近似最优的贪心算法进行选择，但在实际应用中，个性化过程受到结构性约束的复杂制约。这些约束包括逻辑依赖（例如，选择事实A必须同时选择事实B）、类别配额（例如，最多选择一种写作风格）以及层级规则（例如，最多选择两个社交媒体偏好，其中最多一个可以是专业网络偏好）。这些约束违反了传统子集选择算法的假设。我们提出了一种具有理论基础的建模方法，系统地描述了此类限制。具体方法是将具有依赖关系的用户知识图谱转化为一组抽象的宏观要素（macro-facets）。我们的核心结果是证明，基于这些宏观要素的层级和配额型约束构成了有效的层叠模态（laminar matroid）。这一理论特性使我们能够将结构化个性化问题转化为在模态约束下的次模最大化问题，从而利用贪心算法实现具有常数因子保证的近似最优（以及通过连续贪心算法实现的（1 - 1/e）逼近），应对更丰富、更贴近实际的个性化任务场景。

BibTeX

```
@article{2512.11907v1,
  title={Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents},
  author={Daniel Platnick and Marjan Alirezaie and Hossein Rahnama},
  journal={arXiv preprint arXiv:2512.11907v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11907v1}
}
```

## [Personalized QoE Prediction: A Demographic-Augmented Machine Learning Framework for 5G Video Streaming Networks](http://arxiv.org/abs/2512.12736v1)

Syeda Zunaira Ahmed, Hejab Tahira Beg, Maryam Khalid

[PDF](https://arxiv.org/pdf/2512.12736v1)
[Abstract](http://arxiv.org/abs/2512.12736v1)

### 中文摘要

质量体验（QoE）预测是现代多媒体系统中的一个关键组成部分，尤其在5G网络中的自适应视频流中具有重要意义。准确的QoE估算能够实现智能资源管理，支持以用户为中心的服务交付。现有的QoE预测方法主要依赖有限的数据集，并假设用户感知一致，这限制了其在多样化现实环境中的应用。
本文提出了一种面向个性化QoE预测的考虑人口统计特征的机器学习框架。我们引入了一种基于行为学的真实人口统计数据增强策略，通过模拟用户对缓冲重缓冲、码率变化和画质下降等流媒体干扰的不同敏感性，将一个较小的QoE数据集扩展了六倍。在此扩充的数据集基础上，我们对多种经典机器学习模型以及先进的深度学习架构进行了评估，包括基于注意力机制的多层感知机（MLP）和TabNet。
实验结果显示，与基线模型相比，各项指标（RMSE、MAE和R）在预测精度方面均有显著提升。在所有评估方法中，TabNet表现最优，得益于其固有的特征选择和注意力机制。这些结果验证了人口统计特征感知增强策略能够大幅提升QoE预测的稳健性，为5G视频流网络中实现个性化、鲁棒的QoE感知智能提供了可扩展的方向。

BibTeX

```
@article{2512.12736v1,
  title={Personalized QoE Prediction: A Demographic-Augmented Machine Learning Framework for 5G Video Streaming Networks},
  author={Syeda Zunaira Ahmed and Hejab Tahira Beg and Maryam Khalid},
  journal={arXiv preprint arXiv:2512.12736v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12736v1}
}
```

## [Satisfiability Modulo Theory Meets Inductive Logic Programming](http://arxiv.org/abs/2512.12918v1)

Nijesh Upreti, Vaishak Belle

[PDF](https://arxiv.org/pdf/2512.12918v1)
[Abstract](http://arxiv.org/abs/2512.12918v1)

### 中文摘要

归纳逻辑程序设计（ILP）在关系域中提供了可解释的规则学习，但在引入和推理数值约束方面仍存在一定局限。传统的ILP系统以离散谓词为操作对象，通常依赖离散化或手工设计的数值谓词，这使得难以推断必须共同满足的阈值或算术关系。近年来，一些研究开始通过将ILP与 satisfactoriness 模论（SMT）或专门的数值推理机制更紧密地结合，以解决这些限制。本文提出一种模块化的替代方案，将ILP系统PyGol与SMT求解器Z3相结合。PyGol提出的候选条款被解释为在线性或非线性实数演算等背景理论上的无量化公式，使得数值参数可以由SMT求解器实例化和验证，同时保持ILP的声明式关系偏向性。这允许诱导出结合符号谓词与学习到的数值约束（包括阈值、区间和多文字的算术关系）的混合规则。我们形式化了这一SMT-ILP框架，并在一系列旨在测试线性、关系、非线性以及多跳推理的合成数据集上进行了评估。结果显示，模块化的SMT-ILP架构能够扩展符号规则学习的表达能力，补充了现有的数值ILP方法，并为未来向更丰富的理论感知诱导方向拓展提供了灵活的基础。

BibTeX

```
@article{2512.12918v1,
  title={Satisfiability Modulo Theory Meets Inductive Logic Programming},
  author={Nijesh Upreti and Vaishak Belle},
  journal={arXiv preprint arXiv:2512.12918v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12918v1}
}
```

## [Carrot, stick, or both? Price incentives for sustainable food choice in competitive environments](http://arxiv.org/abs/2512.13174v1)

Francesco Salvi, Giuseppe Russo, Adam Barla, Vincent Moreau, Robert West

[PDF](https://arxiv.org/pdf/2512.13174v1)
[Abstract](http://arxiv.org/abs/2512.13174v1)

### 中文摘要

肉类消费是导致全球温室气体排放的主要因素之一。虽然价格干预措施在降低肉类摄入方面展现出潜力，但此前的研究多集中在消费者选择受限的高度约束环境中。本文首次在真实的竞争性环境中开展大规模现场实验，评估多种定价干预措施的效果。我们采用瑞士某大学校园内匹配菜单的连续交叉设计，系统比较了素食餐折扣（-2.5瑞士法郎）、肉类附加费（+2.5瑞士法郎）以及两者结合（-1.2瑞士法郎＋1.2瑞士法郎）在四个餐厅中的实施效果。结果显示，肉类附加费和组合方案均显著提高了素食餐的需求，分别增长26.4%和16.6%，同时每餐的二氧化碳排放量分别减少7.4%和11.3%。虽然肉类附加费在干预地点非常有效，但也导致当地销售额下降12.3%，而非干预地点的销售则相应增加了14.9%，形成了溢出效应，完全抵消了环境上的改善。相比之下，组合方案在实现显著减排的同时，并未对整体销售或收入产生显著影响，表现出更好的经济与环境兼顾性。值得注意的是，价格干预对素食倾向明显的消费者和习惯性食肉者均同样有效，即使在根深蒂固的饮食习惯中也能激发变化。我们的研究表明，合理的定价策略有助于降低现实食品环境的碳足迹，但其效果依赖于协调统一的实施，以最大化气候益处并避免不良的溢出效应。

BibTeX

```
@article{2512.13174v1,
  title={Carrot, stick, or both? Price incentives for sustainable food choice in competitive environments},
  author={Francesco Salvi and Giuseppe Russo and Adam Barla and Vincent Moreau and Robert West},
  journal={arXiv preprint arXiv:2512.13174v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13174v1}
}
```

## [From Overfitting to Reliability: Introducing the Hierarchical Approximate Bayesian Neural Network](http://arxiv.org/abs/2512.13111v1)

Hayk Amirkhanian, Marco F. Huber

[PDF](https://arxiv.org/pdf/2512.13111v1)
[Abstract](http://arxiv.org/abs/2512.13111v1)

### 中文摘要

近年来，神经网络在多个领域引发了革命性变革，但超参数调优和过拟合等问题仍然是亟需解决的重要挑战。贝叶斯神经网络通过在模型中引入不确定性，为应对这些挑战提供了有效框架，特别是在处理分布外数据时能够产生更可靠的预测结果。本文提出了一种层次近似贝叶斯神经网络（Hierarchical Approximate Bayesian Neural Network，HABNN），这是一种创新的方法，采用高斯-逆威沙特分布作为网络权重的超先验，以提升模型的鲁棒性和性能。我们提供了预测分布和权重后验的解析表达式，这些表达式转化为学生t分布参数的封闭式计算，且其计算复杂度线性依赖于权重数。我们的方案在实验中展示了优异的性能，能够有效缓解过拟合问题，并为分布外任务提供可靠的不确定性估计。结果显示，HABNN不仅达到了甚至超越了现有的先进模型，为在安全关键环境中的未来应用开辟了具有潜力的研究方向。

BibTeX

```
@article{2512.13111v1,
  title={From Overfitting to Reliability: Introducing the Hierarchical Approximate Bayesian Neural Network},
  author={Hayk Amirkhanian and Marco F. Huber},
  journal={arXiv preprint arXiv:2512.13111v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13111v1}
}
```

## [Socratic Students: Teaching Language Models to Learn by Asking Questions](http://arxiv.org/abs/2512.13102v1)

Rajeev Bhatt Ambati, Tianyi Niu, Aashu Singh, Shlok Mishra, Shashank Srivastava, Snigdha Chaturvedi

[PDF](https://arxiv.org/pdf/2512.13102v1)
[Abstract](http://arxiv.org/abs/2512.13102v1)

### 中文摘要

大型语言模型（LLMs）在静态交互中表现优异，能够通过检索其参数中编码的知识来回答用户提问。然而，在许多实际应用场景中，例如教育辅导或医疗援助，相关信息并非直接可用，而需要通过动态交互主动获取。一个具有交互能力的智能体应能识别自身的不确定性，提出有针对性的问题，并高效地获取新知识。此前的研究主要集中在教师如何有效指导学生，即教师识别学生的知识空白并提供指导。而在本研究中，我们将焦点转向学生，探讨学生主动向教师提问以获取有用信息的有效策略。在数学和编程等基准测试中，基础学生模型起始性能几乎为零，我们的实验显示，学生主导的方法在静态基线基础上，平均至少提高0.5的绝对Pass@k成绩。为了提升提问质量，我们采用直接偏好优化（DPO）对学生进行训练，并辅以来自自我或更强学生的指导。研究发现，这种指导性训练使得较小模型也能学会提出更优的问题，从而进一步提高学习效率。

BibTeX

```
@article{2512.13102v1,
  title={Socratic Students: Teaching Language Models to Learn by Asking Questions},
  author={Rajeev Bhatt Ambati and Tianyi Niu and Aashu Singh and Shlok Mishra and Shashank Srivastava and Snigdha Chaturvedi},
  journal={arXiv preprint arXiv:2512.13102v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13102v1}
}
```

## [DA-SSL: self-supervised domain adaptor to leverage foundational models in turbt histopathology slides](http://arxiv.org/abs/2512.13600v1)

Haoyue Zhang, Meera Chappidi, Erolcan Sayar, Helen Richards, Zhijun Chen, Lucas Liu, Roxanne Wadia, Peter A Humphrey, Fady Ghali, Alberto Contreras-Sanz, Peter Black, Jonathan Wright, Stephanie Harmon, Michael Haffner

[PDF](https://arxiv.org/pdf/2512.13600v1)
[Abstract](http://arxiv.org/abs/2512.13600v1)

### 中文摘要

近年来，结合组织病理学中的多实例学习（MIL）与基础病理模型（PFMs）的深度学习框架在该领域展现出优异的性能。然而，由于域迁移问题，PFMs在某些癌症类型或样本类型上存在一定限制——这些癌症类型在预训练中很少使用，或样本中存在在预训练人群中罕见的组织碎片或热烧蚀伪影。例如，经尿道膀胱肿瘤切除术（TURBT）在肌层浸润性膀胱癌（MIBC）诊断中至关重要，但其样本常含有碎片化的组织块和电灼伪影，且在公开的PFMs中应用较少。为此，我们提出了一种简洁而有效的域适应自监督转换器（DA-SSL），无需微调基础模型，即能将预训练PFM的特征重新调整到TURBT样本域中。我们在TURBT的治疗反应预测任务中进行了试点，该任务中组织形态特征尚未得到充分利用，且识别受益于新辅助化疗（NAC）的患者具有较大挑战。在多中心研究中，DA-SSL在五折交叉验证中达到了0.77±0.04的AUC，且在外部测试中通过多数投票实现了0.84的准确率、0.71的敏感性和0.91的特异性。结果表明，轻量级的自监督域适应方法能够有效增强基于PFM的MIL管道在临床实际中面临的组织病理学任务中的表现。相关代码已在https://github.com/zhanghaoyue/DA\_SSL\_TURBT开源。

BibTeX

```
@article{2512.13600v1,
  title={DA-SSL: self-supervised domain adaptor to leverage foundational models in turbt histopathology slides},
  author={Haoyue Zhang and Meera Chappidi and Erolcan Sayar and Helen Richards and Zhijun Chen and Lucas Liu and Roxanne Wadia and Peter A Humphrey and Fady Ghali and Alberto Contreras-Sanz and Peter Black and Jonathan Wright and Stephanie Harmon and Michael Haffner},
  journal={arXiv preprint arXiv:2512.13600v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13600v1}
}
```

## [Verifying Rumors via Stance-Aware Structural Modeling](http://arxiv.org/abs/2512.13559v1)

Gibson Nkhata, Uttamasha Anjally Oyshi, Quan Mai, Susan Gauch

[PDF](https://arxiv.org/pdf/2512.13559v1)
[Abstract](http://arxiv.org/abs/2512.13559v1)

### 中文摘要

在社交媒体上验证谣言的真实性对于抑制虚假信息的传播至关重要。对话回复的立场倾向通常提供了判断谣言真实性的重要线索。然而，现有模型在同时捕捉语义内容、立场信息和对话结构方面存在困难，尤其是在基于变换器编码器的序列长度限制下。本研究提出了一种基于立场感知的结构建模方法，该方法对话中的每个帖子都编码其立场信号，并通过立场类别对回复嵌入进行聚合，从而实现对整个讨论串的可扩展且语义丰富的表示。为了增强结构感知能力，我们引入了立场分布和层级深度作为协变量，以捕捉立场失衡和回复深度的影响。大量在基准数据集上的实验证明，该方法在谣言真实性预测方面显著优于现有方法。此外，我们还展示了该模型在早期检测和跨平台泛化方面的强大适应能力。

BibTeX

```
@article{2512.13559v1,
  title={Verifying Rumors via Stance-Aware Structural Modeling},
  author={Gibson Nkhata and Uttamasha Anjally Oyshi and Quan Mai and Susan Gauch},
  journal={arXiv preprint arXiv:2512.13559v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13559v1}
}
```

## [Defending the Hierarchical Result Models of Precedential Constraint](http://arxiv.org/abs/2512.13505v1)

Henry Prakken, Wijnand van Woerkom

[PDF](https://arxiv.org/pdf/2512.13505v1)
[Abstract](http://arxiv.org/abs/2512.13505v1)

### 中文摘要

近年来，提出了一些基于层级案例推理的先例限制模型。在多篇论文中，特雷弗·本奇-卡庞(I)，批评这些模型可能在某些情况下得出不正确的结论。特别是，这些模型未能考虑到中间因素可能由不同的基础因素以不同的强度建立起来的可能性。本文针对范·沃尔科姆的基于结果的层级模型进行了回应。我们认为，在某些例子中，本奇-卡庞似乎将中间因素解释为维度，并且将范·沃尔科姆基于维度的层级结果模型应用到这些例子中，能够避免本奇-卡庞的批评。

BibTeX

```
@article{2512.13505v1,
  title={Defending the Hierarchical Result Models of Precedential Constraint},
  author={Henry Prakken and Wijnand van Woerkom},
  journal={arXiv preprint arXiv:2512.13505v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13505v1}
}
```

## [Behavior-Aware and Generalizable Defense Against Black-Box Adversarial Attacks for ML-Based IDS](http://arxiv.org/abs/2512.13501v1)

Sabrine Ennaji, Elhadj Benkhelifa, Luigi Vincenzo Mancini

[PDF](https://arxiv.org/pdf/2512.13501v1)
[Abstract](http://arxiv.org/abs/2512.13501v1)

### 中文摘要

基于机器学习的入侵检测系统（IDS）日益成为黑箱对抗性攻击的目标，攻击者通过间接反馈信息如二进制输出或行为信号（如响应时间和资源使用情况），设计具有迷惑性的规避输入。尽管已有多种防御措施被提出，包括输入变换、对抗训练和代理检测等，但在实际应用中往往难以奏效。这些方法大多针对特定攻击类型，要求对模型内部信息的访问，或依赖于静态机制，难以应对不断演变的攻击策略。此外，如输入变换等防御措施可能会降低入侵检测系统的性能，不适合实时部署。
为了解决这些局限，我们提出了一种名为“自适应特征毒化”的轻量级主动防御机制，特别适用于真实场景中的黑箱攻击假设。该方法假设攻击者可以在不被察觉的情况下进行持续探测，并通过引入动态的、基于上下文的扰动，干扰攻击者的反馈回路，而不影响检测性能。具体而言，方法结合流量分析、变点检测和自适应缩放技术，对攻击者可能利用的关键流量特征进行选择性扰动，基于观测到的偏差进行动态调整。
我们在多种现实攻击策略下对“自适应特征毒化”进行评估，包括静默探测、迁移性攻击以及基于决策边界的攻击。实验结果显示，该机制能够有效迷惑攻击者、削弱攻击效果，同时保持检测性能。作为一种通用、攻击无关且难以被检测的防御方案，“自适应特征毒化”在机器学习型入侵检测系统的实际抗对抗性能提升方面迈出了重要一步。

BibTeX

```
@article{2512.13501v1,
  title={Behavior-Aware and Generalizable Defense Against Black-Box Adversarial Attacks for ML-Based IDS},
  author={Sabrine Ennaji and Elhadj Benkhelifa and Luigi Vincenzo Mancini},
  journal={arXiv preprint arXiv:2512.13501v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13501v1}
}
```

## [neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings](http://arxiv.org/abs/2512.13481v1)

Ojas Pungalia, Rashi Upadhyay, Abhishek Mishra, Abhiram H, Tejasvi Alladi, Sujan Yenuganti, Dhruv Kumar

[PDF](https://arxiv.org/pdf/2512.13481v1)
[Abstract](http://arxiv.org/abs/2512.13481v1)

### 中文摘要

嫉妒是一种普遍的人类行为，影响着竞争关系并可能改变团队环境中的结果。随着大型语言模型（LLMs）在合作与竞争工作流程中日益代表人类行事，迫切需要评估它们是否以及在何种条件下会展现出类似嫉妒的偏好。在本研究中，我们考察了LLMs是否会彼此表现出嫉妒样行为。我们设计了两种场景：一是点数分配游戏，用以测试模型是否试图争夺对手的优势；二是在工作场景中观察当认可以不公平时的行为表现。研究结果显示，部分LLMs确实表现出嫉妒样模式的证据，且在模型类型和使用场景间存在显著差异。例如，GPT-5-mini和Claude-3.7-Sonnet表现出明显倾向于拉低竞争对手以实现结果的平衡，而Mistral-Small-3.2-24B则倾向于最大化自身收益。这些发现强调在基于LLM的多智能体系统设计中，竞争倾向应作为安全性与设计考量的重要因素。

BibTeX

```
@article{2512.13481v1,
  title={neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings},
  author={Ojas Pungalia and Rashi Upadhyay and Abhishek Mishra and Abhiram H and Tejasvi Alladi and Sujan Yenuganti and Dhruv Kumar},
  journal={arXiv preprint arXiv:2512.13481v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13481v1}
}
```

## [SSAS: Cross-subject EEG-based Emotion Recognition through Source Selection with Adversarial Strategy](http://arxiv.org/abs/2512.13458v1)

Yici Liu, Qi Wei Oung, Hoi Leong Lee

[PDF](https://arxiv.org/pdf/2512.13458v1)
[Abstract](http://arxiv.org/abs/2512.13458v1)

### 中文摘要

脑电（EEG）信号长期以来在情感脑-机接口（aBCI）领域中得到广泛应用。基于跨主体的EEG情感识别由于其在不同个体中的普适性，展现出巨大的实际潜力。然而，绝大多数跨主体EEG情感识别研究在模型训练过程中忽视了个体差异的存在以及负迁移现象的影响。为解决这一问题，本文提出了一种结合源域选择与对抗策略的跨主体EEG情感识别方法。所提方法由两个模块组成：源域选择网络（SS）和对抗策略网络（AS）。其中，SS利用领域标签反向工程调整领域适应的训练过程，其核心思想是破坏类别的可分离性，放大域间差异，从而增加分类难度，促使模型学习到既具有领域不变性又与情感相关的表征。AS从SS获取源域选择结果和预训练的领域判别器，通过预训练的领域判别器计算一项新颖的损失函数，旨在提升对抗训练中域分类的性能，确保对抗策略的平衡。本文对所提方法进行了理论分析，并在两个基于EEG的情感数据集——SEED和SEED-IV上取得了优异的实验结果。相关代码已开源，地址为https://github.com/liuyici/SSAS。

BibTeX

```
@article{2512.13458v1,
  title={SSAS: Cross-subject EEG-based Emotion Recognition through Source Selection with Adversarial Strategy},
  author={Yici Liu and Qi Wei Oung and Hoi Leong Lee},
  journal={arXiv preprint arXiv:2512.13458v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13458v1}
}
```

## [Differentiable Evolutionary Reinforcement Learning](http://arxiv.org/abs/2512.13399v1)

Sitao Cheng, Tianle Li, Xuhan Huang, Xunjian Yin, Difan Zou

[PDF](https://arxiv.org/pdf/2512.13399v1)
[Abstract](http://arxiv.org/abs/2512.13399v1)

### 中文摘要

有效奖励函数的设计在强化学习（RL）中一直是核心而又常常具有挑战性的难题，尤其是在开发用于复杂推理任务的自主智能体时。虽然现有一些自动化奖励优化的方法，但它们通常依赖于无梯度的进化启发式策略，将奖励函数作为“黑箱”处理，未能充分捕捉奖励结构与任务性能之间的因果关系。为弥补这一不足，本文提出了可微分的演化强化学习框架（DERL），一种双层结构，能够自主发现最优的奖励信号。在DERL中，一个元优化器通过组合结构化的基本原语，演化出一种元奖励（Meta-Reward），从而引导内环策略的训练。更为关键的是，与以往演化方法不同，DERL在元优化过程中具有可微性：它将内环验证性能作为信号，通过强化学习的方式更新元优化器。这使得DERL能够逼近任务成功的“元梯度”，逐步学习生成更密集、更具行动指导意义的反馈信息。我们在三个不同领域进行了验证：机器人智能体（ALFWorld）、科学模拟（ScienceWorld）以及数学推理（GSM8k、MATH）。实验结果显示，DERL在ALFWorld和ScienceWorld中达到了最新的性能水平，显著优于依赖启发式奖励的方法，尤其在分布外的场景中表现突出。对演化轨迹的分析表明，DERL成功捕捉到任务的内在结构，实现了无需人为干预的自我改进与智能体对齐。

BibTeX

```
@article{2512.13399v1,
  title={Differentiable Evolutionary Reinforcement Learning},
  author={Sitao Cheng and Tianle Li and Xuhan Huang and Xunjian Yin and Difan Zou},
  journal={arXiv preprint arXiv:2512.13399v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13399v1}
}
```

## [Security and Detectability Analysis of Unicode Text Watermarking Methods Against Large Language Models](http://arxiv.org/abs/2512.13325v1)

Malte Hellmeier

[PDF](https://arxiv.org/pdf/2512.13325v1)
[Abstract](http://arxiv.org/abs/2512.13325v1)

### 中文摘要

随着大型语言模型的广泛应用，数字文本的安全性变得日益重要。个体对于在训练此类模型时数据可能被滥用或被用于生成模型输出的内容而失去控制的担忧日增。数字水印技术通过在数据中嵌入隐形水印，为数据提供额外保护。然而，目前尚缺乏对现有数字文本水印方法的安全性及其是否能被大型语言模型检测到的系统性分析。在本文中，我们探讨了与水印技术及机器学习模型在文本数据安全中的关系。在一个包含三个实验的控制测试平台上，实施并分析了十种现有Unicode文本水印方法，评估对象包括六种大型语言模型：GPT-5、GPT-4o、腾迅7B、Llama 3.3、Claude Sonnet 4以及Gemini 2.5 Pro。实验结果显示，尤其是最新的推理型模型，具有检测水印文本的能力。然而，除非提供源代码形式的具体实现细节，否则所有模型均无法提取水印。本文讨论了这些发现对安全研究人员和实践者的影响，并提出了未来可能的研究方向，以应对相关安全挑战。

BibTeX

```
@article{2512.13325v1,
  title={Security and Detectability Analysis of Unicode Text Watermarking Methods Against Large Language Models},
  author={Malte Hellmeier},
  journal={arXiv preprint arXiv:2512.13325v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13325v1}
}
```

## [No One Left Behind: How to Exploit the Incomplete and Skewed Multi-Label Data for Conversion Rate Prediction](http://arxiv.org/abs/2512.13300v1)

Qinglin Jia, Zhaocheng Du, Chuhan Wu, Huifeng Guo, Ruiming Tang, Shuting Shi, Muyu Zhang

[PDF](https://arxiv.org/pdf/2512.13300v1)
[Abstract](http://arxiv.org/abs/2512.13300v1)

### 中文摘要

在大多数现实世界的在线广告系统中，广告主通常拥有多样化的用户转化目标。为此，常用的解决方案是采用多任务学习（MTL），在后点击数据上训练统一模型，以估算这些不同目标的转化率（CVR）。然而，在实际应用中，CVR预测经常面临缺失转化数据的问题，因为许多广告主由于隐私或其他限制，仅提交部分用户转化行为数据，从而导致多任务数据的标签不完整。如果模型在所有可用样本——即广告主提交的用户转化行为数据——上训练，可能在部署时难以针对特定转化行为的子集广告主提供服务，因为训练数据和实际部署数据的分布存在偏差。尽管在多任务学习方面已做出大量努力，但长期存在的挑战是如何在多标签数据不完整且偏斜的情况下，有效地训练出一个统一的模型。为此，本文提出了一种面向不对称多标签数据的细粒度知识迁移框架（KAML）。我们引入了一种归因驱动的屏蔽策略（ADM），以更好地利用不对称多标签数据进行训练。然而，ADM中的较宽松屏蔽机制具有双刃剑的性质：一方面提供了额外的训练信号，另一方面也引入了由偏斜数据引起的噪声。为此，我们提出了分层知识提取机制（HKE），以建模目标任务塔内部的样本差异。最后，为了最大化未标注样本的利用价值，我们引入了排序损失策略，进一步提升模型性能。通过在行业线下数据集和线上A/B测试中的全面评估，验证了KAML的有效性，并展现出优于现有多任务学习基线的显著性能提升。

BibTeX

```
@article{2512.13300v1,
  title={No One Left Behind: How to Exploit the Incomplete and Skewed Multi-Label Data for Conversion Rate Prediction},
  author={Qinglin Jia and Zhaocheng Du and Chuhan Wu and Huifeng Guo and Ruiming Tang and Shuting Shi and Muyu Zhang},
  journal={arXiv preprint arXiv:2512.13300v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13300v1}
}
```

## [LINA: Learning INterventions Adaptively for Physical Alignment and Generalization in Diffusion Models](http://arxiv.org/abs/2512.13290v1)

Shu Yu, Chaochao Lu

[PDF](https://arxiv.org/pdf/2512.13290v1)
[Abstract](http://arxiv.org/abs/2512.13290v1)

### 中文摘要

扩散模型（DMs）在图像与视频生成方面已取得显著成就。然而，它们在（1）物理对齐和（2）超出分布（OOD）指令执行方面仍存在困难。我们认为，这些问题源于模型未能学习因果方向以及未能对因果因素进行解缠以实现新颖的重组。为此，我们引入了因果场景图（Causal Scene Graph, CSG）和物理对齐探针（Physical Alignment Probe, PAP）数据集，以进行诊断性干预。通过分析，我们得出了三个关键见解：第一，扩散模型在处理提示中未明确指定的多跳推理元素时表现不佳；第二，提示嵌入中包含了纹理和物理特性的解缠表示；第三，可视因果结构主要在初始、计算资源有限的去噪步骤中确立。基于这些发现，我们提出了一种新颖的框架——LINA（Learning INterventions Adaptive 自适应干预学习），其通过（1）在提示和视觉潜空间中的针对性引导，以及（2）重分配、考虑因果关系的去噪流程，实现对提示特定干预的预测。我们的方法在图像与视频扩散模型中有效实现了物理对齐和超出分布指令的执行，在多个挑战性因果生成任务和Winoground数据集上达到了先进水平。我们的项目页面为 https://opencausalab.github.io/LINA。

BibTeX

```
@article{2512.13290v1,
  title={LINA: Learning INterventions Adaptively for Physical Alignment and Generalization in Diffusion Models},
  author={Shu Yu and Chaochao Lu},
  journal={arXiv preprint arXiv:2512.13290v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13290v1}
}
```

## [Memoria: A Scalable Agentic Memory Framework for Personalized Conversational AI](http://arxiv.org/abs/2512.12686v1)

Samarth Sarin, Lovepreet Singh, Bhaskarjit Sarmah, Dhagash Mehta

[PDF](https://arxiv.org/pdf/2512.12686v1)
[Abstract](http://arxiv.org/abs/2512.12686v1)

### 中文摘要

代理记忆正逐渐成为大规模语言模型（LLM）实现连续性、个性化以及长远语境维护的关键支撑，这些能力对于将LLM作为真正的交互式和自适应代理具有重要意义。代理记忆指的是赋予LLM类似“代理人”般的持续性记忆：即在多轮对话中能够保留并运用信息的能力，类似于人类的表现。我们提出了Memoria，一种模块化的记忆框架，旨在为基于LLM的对话系统增添持久的、可解释的、富含语境的记忆。Memoria融合了两大互补组件：动态会话级摘要和基于加权知识图（KG）的用户建模引擎，该引擎逐步以结构化实体和关系的形式捕捉用户特征、偏好以及行为模式。这一混合架构在满足现代LLM.token限制的同时，兼顾短期对话连贯性与长期个性化。我们展示了Memoria如何通过弥合无状态LLM接口与代理记忆系统的差距，实现可扩展的个性化对话人工智能（AI），为行业应用中需要自适应、持续演进用户体验的场景提供了切实可行的解决方案。

BibTeX

```
@article{2512.12686v1,
  title={Memoria: A Scalable Agentic Memory Framework for Personalized Conversational AI},
  author={Samarth Sarin and Lovepreet Singh and Bhaskarjit Sarmah and Dhagash Mehta},
  journal={arXiv preprint arXiv:2512.12686v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12686v1}
}
```

## [The Forecast Critic: Leveraging Large Language Models for Poor Forecast Identification](http://arxiv.org/abs/2512.12059v1)

Luke Bhan, Hanyu Zhang, Andrew Gordon Wilson, Michael W. Mahoney, Chuck Arvin

[PDF](https://arxiv.org/pdf/2512.12059v1)
[Abstract](http://arxiv.org/abs/2512.12059v1)

### 中文摘要

监控预测系统对于大型零售企业的客户满意度、盈利能力和运营效率至关重要。我们提出了“预测批评者”系统，该系统利用大型语言模型（LLMs）进行自动化的预测监测，充分发挥其广泛的知识储备和强大的推理能力。作为前提条件，我们系统性评估了LLMs评估时间序列预测质量的能力，重点关注三个核心问题：（1）LLMs是否能够用于执行预测监控并识别明显不合理的预测？（2）LLMs是否能够有效整合非结构化的外部特征，以判断合理预测的标准？（3）在不同模型规模和推理能力下，性能表现如何变化，并在最先进的LLMs中进行衡量。我们设计了三组实验，包括对合成数据和实际应用中的预测数据的测试。结果显示，LLMs能够可靠地检测并指出差劲的预测，例如存在时间错位、趋势不一致和尖峰误差的问题。我们评估的最佳模型达到了0.88的F1分数，略低于人类水平（F1分数为0.97）。此外，我们还演示了多模态LLMs能够有效整合非结构化的上下文信号，从而优化其预测评估。当提供有关历史促销活动的背景信息时，模型能正确识别缺失或误导性的促销尖峰（F1分数为0.84）。最后，我们证明这些技术在真实的M5时间序列数据集上也能成功识别不准确的预测，表现为不合理预测的sCRPS指标至少比合理预测高出10%。这些发现表明，即使未经专门领域微调，LLMs也可能成为一种可行且具有规模化能力的自动预测监测与评估方案。

BibTeX

```
@article{2512.12059v1,
  title={The Forecast Critic: Leveraging Large Language Models for Poor Forecast Identification},
  author={Luke Bhan and Hanyu Zhang and Andrew Gordon Wilson and Michael W. Mahoney and Chuck Arvin},
  journal={arXiv preprint arXiv:2512.12059v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12059v1}
}
```

## [AgentSHAP: Interpreting LLM Agent Tool Importance with Monte Carlo Shapley Value Estimation](http://arxiv.org/abs/2512.12597v1)

Miriam Horovicz

[PDF](https://arxiv.org/pdf/2512.12597v1)
[Abstract](http://arxiv.org/abs/2512.12597v1)

### 中文摘要

使用外部工具的大规模语言模型（LLM）代理能够完成复杂任务，但对哪些工具实际上对响应产生了贡献的理解仍然是一个盲点。目前尚无任何可解释性人工智能（XAI）方法能专门解决工具层级的解释问题。我们提出了AgentSHAP，这是首个用于解释LLM代理中工具重要性的框架。AgentSHAP具有模型无关性：它将代理视为黑箱，并支持任何类型的LLM（如GPT、Claude、Llama等），无需访问模型内部参数或梯度。通过蒙特卡洛Shapley值，AgentSHAP测试代理在使用不同工具子集时的响应情况，并基于博弈论计算公平的工具重要性评分。
我们的贡献包括：① 基于博弈论中的Shapley值，提出了第一个用于代理工具归因的可解释性方法；② 采用蒙特卡洛采样，有效降低计算成本，从指数级的O(2^n)到实际可行的水平；③ 在API-Bank数据集上的完整实验显示，AgentSHAP可以在不同运行中产生一致的评分，准确识别重要工具，并区分相关与无关的工具。AgentSHAP与TokenSHAP（针对token）和PixelSHAP（针对图像区域）共同构成了一个基于Shapley值的现代生成式AI解释工具家族。相关代码已开源：https://github.com/GenAISHAP/TokenSHAP。

BibTeX

```
@article{2512.12597v1,
  title={AgentSHAP: Interpreting LLM Agent Tool Importance with Monte Carlo Shapley Value Estimation},
  author={Miriam Horovicz},
  journal={arXiv preprint arXiv:2512.12597v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12597v1}
}
```

## [Understanding Critical Thinking in Generative Artificial Intelligence Use: Development, Validation, and Correlates of the Critical Thinking in AI Use Scale](http://arxiv.org/abs/2512.12413v1)

Gabriel R. Lau, Wei Yan Low, Louis Tay, Ysabel Guevarra, Dragan Gašević, Andree Hartanto

[PDF](https://arxiv.org/pdf/2512.12413v1)
[Abstract](http://arxiv.org/abs/2512.12413v1)

### 中文摘要

生成式人工智能工具在日常工作和学习中的应用日益普及，但其流畅性、不透明性以及易产生“幻觉”式错误的问题，意味着用户在接受其输出时必须进行批判性评估，而非盲从。本研究将人工智能使用中的批判性思维概念化为一种习性倾向，表现为验证AI生成信息的来源与内容、理解模型的工作机制及其局限、以及反思依赖AI的更广泛影响。在六项研究（共1365名参与者）中，我们开发并验证了13项的“AI批判性思维”量表，并探讨了其法则性关系网络。第一项研究生成并内容验证了量表条目。第二项研究支持了量表的三维结构（验证动机、动力和反思）。第三、四、五项研究确认了这一高阶模型，表现出较高的内部一致性和重测信度，具有较强的因子负荷、性别不变性，以及汇聚效度和区分效度。第3和第4项研究进一步发现，AI批判性思维与开放性、外向性、积极性状情感以及使用频率呈正相关。最后，第6项研究验证了该量表的标准效度，高分的批判性思维个体在验证策略上表现出更高的频率与多样性，在基于ChatGPT的事实核查任务中表现出更高的真实性判断准确性，并对负责任的AI使用进行了更深入的反思。综上所述，本研究阐明了人们为何以及如何对生成式AI输出进行监督，并提供了一套经过验证的量表及生态学基础的任务范式，以支持关于批判性参与生成式AI输出的理论检验、跨群体研究及纵向追踪研究。

BibTeX

```
@article{2512.12413v1,
  title={Understanding Critical Thinking in Generative Artificial Intelligence Use: Development, Validation, and Correlates of the Critical Thinking in AI Use Scale},
  author={Gabriel R. Lau and Wei Yan Low and Louis Tay and Ysabel Guevarra and Dragan Gašević and Andree Hartanto},
  journal={arXiv preprint arXiv:2512.12413v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12413v1}
}
```

## [Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows](http://arxiv.org/abs/2512.13168v1)

Haoyu Dong, Pengkun Zhang, Yan Gao, Xuanyu Dong, Yilin Cheng, Mingzhe Lu, Adina Yakefu, Shuxin Zheng

[PDF](https://arxiv.org/pdf/2512.13168v1)
[Abstract](http://arxiv.org/abs/2512.13168v1)

### 中文摘要

我们提出了一项金融与财务评估基准（Finch），用于评价人工智能代理在实际企业级专业工作流程中的表现，这些流程包括数据输入、结构化、格式整理、网页搜索、跨文件检索、计算、建模、验证、翻译、可视化和报告等环节。Finch 数据来源于安然公司（Enron）及其他金融机构的真实企业工作空间，涵盖15,000个电子表格和50万封电子邮件，涉及150名员工，保留了多模态文档（文本、表格、公式、图表、代码和图像）中的真实杂乱现象，覆盖预算、交易和资产管理等多个领域。
我们提出了一套结合大模型辅助发现与专家标注的工作流程构建方法：(1) 利用大模型辅以专家验证，从实际电子邮件线程和电子表格版本历史中推导出工作流程；(2) 由行业专家进行细致标注，耗时超过700小时。最终，构建了172个复合工作流程，包含384个任务，涉及1710个电子表格（总计27百万个单元格），以及PDF和其他相关文档，充分体现了实际企业工作中天生的杂乱、长周期性、知识密集性和协作性。
我们对多款前沿人工智能系统进行了人类评估与自动测试，包括GPT 5.1、Claude Sonnet 4.5、Gemini 3 Pro、Grok 4和Qwen 3 Max。其中，GPT 5.1 Pro花费总计48小时，仅完成了38.4%的工作流程，而Claude Sonnet 4.5的完成率仅为25.0%。通过全面的案例研究，揭示了现实企业工作流程对AI代理提出的诸多挑战。

BibTeX

```
@article{2512.13168v1,
  title={Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows},
  author={Haoyu Dong and Pengkun Zhang and Yan Gao and Xuanyu Dong and Yilin Cheng and Mingzhe Lu and Adina Yakefu and Shuxin Zheng},
  journal={arXiv preprint arXiv:2512.13168v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13168v1}
}
```

## [SpeakRL: Synergizing Reasoning, Speaking, and Acting in Language Models with Reinforcement Learning](http://arxiv.org/abs/2512.13159v1)

Emre Can Acikgoz, Jinoh Oh, Jie Hao, Joo Hyuk Jeon, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur, Xiang Li, Chengyuan Ma, Xing Fan

[PDF](https://arxiv.org/pdf/2512.13159v1)
[Abstract](http://arxiv.org/abs/2512.13159v1)

### 中文摘要

有效的人机协作在实际应用中变得越来越普遍。目前的协作大多呈单向性，用户向智能体提出指令或问题，智能体直接回应，而缺乏必要的澄清或确认。然而，随着智能体能力的不断提升，未来需要更具主动性的互动方式，智能体应能够在对话中主动参与，澄清用户意图、解决歧义并适应变化的情境。现有的方法在充分利用语言模型（LMs）对话能力方面存在不足，通常将智能体设计为更好的执行者而非善于引导的主动交互者。在本研究中，我们提出了SpeakRL，一种强化学习（RL）方法，通过奖励智能体主动与用户交流的行为，如在必要时提出恰当的澄清问题，来增强其对话能力。为此，我们创建了SpeakER，一个包含多样任务导向对话场景的合成数据集，这些任务通过交互式澄清问题得以完成。我们系统分析了激励主动交互的奖励设计，提出了平衡提问与行动的原则性奖励函数。在实证评估中，Our方法在任务完成率上相较基础模型实现了20.14%的绝对提升，而未增加对话轮次，甚至优于一些规模更大的专有模型，充分展示了以澄清为核心的人机交互的潜力。

BibTeX

```
@article{2512.13159v1,
  title={SpeakRL: Synergizing Reasoning, Speaking, and Acting in Language Models with Reinforcement Learning},
  author={Emre Can Acikgoz and Jinoh Oh and Jie Hao and Joo Hyuk Jeon and Heng Ji and Dilek Hakkani-Tür and Gokhan Tur and Xiang Li and Chengyuan Ma and Xing Fan},
  journal={arXiv preprint arXiv:2512.13159v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13159v1}
}
```

## [MAC: A Multi-Agent Framework for Interactive User Clarification in Multi-turn Conversations](http://arxiv.org/abs/2512.13154v1)

Emre Can Acikgoz, Jinoh Oh, Joo Hyuk Jeon, Jie Hao, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur, Xiang Li, Chengyuan Ma, Xing Fan

[PDF](https://arxiv.org/pdf/2512.13154v1)
[Abstract](http://arxiv.org/abs/2512.13154v1)

### 中文摘要

对话代理在处理模糊的用户请求时，常常需要有效的澄清以确保任务的成功完成。尽管近年来在实际应用中多智能体架构的进展推动了复杂对话场景的高效管理，但模糊性解决仍然是一个关键且研究不足的挑战——尤其在于判定何时由哪个智能体发起澄清，以及在面对不确定或不完整的用户输入时，智能体应如何协调行动。关于何时打断用户，以及在最优多智能体设置下如何制定最优的澄清策略，这些根本性问题仍未得到充分解决。本文提出了多智能体澄清框架（MAC），一种专门优化以通过策略性管理澄清对话来解决用户模糊的交互式多智能体体系。我们首先引入一种新颖的用户模糊性分类体系，以系统性指导澄清策略的制定。随后，提出MAC框架，能够在多智能体之间实现自主协调，与用户协同互动。在MultiWOZ 2.4数据集上的实证评估表明，支持多层次澄清显著提升任务成功率7.8%（从54.5提升至62.3），同时减少对话轮数（从6.53轮降至4.86轮），这是通过提前获取所有所需用户信息和减少重复交互实现的。这些发现强调了主动用户交互和角色感知澄清在构建更可靠的人机交流中的重要作用。

BibTeX

```
@article{2512.13154v1,
  title={MAC: A Multi-Agent Framework for Interactive User Clarification in Multi-turn Conversations},
  author={Emre Can Acikgoz and Jinoh Oh and Joo Hyuk Jeon and Jie Hao and Heng Ji and Dilek Hakkani-Tür and Gokhan Tur and Xiang Li and Chengyuan Ma and Xing Fan},
  journal={arXiv preprint arXiv:2512.13154v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13154v1}
}
```

## [DePT3R: Joint Dense Point Tracking and 3D Reconstruction of Dynamic Scenes in a Single Forward Pass](http://arxiv.org/abs/2512.13122v1)

Vivek Alumootil, Tuan-Anh Vu, M. Khalid Jawed

[PDF](https://arxiv.org/pdf/2512.13122v1)
[Abstract](http://arxiv.org/abs/2512.13122v1)

### 中文摘要

目前在动态场景中进行稠密三维点追踪的方法，通常依赖于成对处理、已知的相机姿态或假设时间序列的帧序关系，这些限制降低了其灵活性与适用范围。此外，近年来的研究成功实现了对大规模无位姿图像集的高效三维重建，展现了在动态场景理解中统一方法的潜力。基于此，我们提出了一种名为DePT3R的创新框架，该方法能够在一次前向传播中同时实现对动态场景的稠密点追踪与三维重建，支持多视角图像输入。这一多任务学习通过强大的骨干网络提取深层时空特征，并采用密集预测头回归像素级映射。关键在于，DePT3R无需依赖相机位姿，大幅提升了其适应性与效率，尤其适用于变化迅速的动态环境。我们在多个具有挑战性的动态场景基准测试中验证了DePT3R的优异性能，展示了其强大的表现力及在存储效率方面对现有最先进方法的显著提升。相关数据与代码已经在开源仓库中公开： https://github.com/StructuresComp/DePT3R

BibTeX

```
@article{2512.13122v1,
  title={DePT3R: Joint Dense Point Tracking and 3D Reconstruction of Dynamic Scenes in a Single Forward Pass},
  author={Vivek Alumootil and Tuan-Anh Vu and M. Khalid Jawed},
  journal={arXiv preprint arXiv:2512.13122v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13122v1}
}
```

## [Uncovering the Role of Initial Saliency in U-Shaped Attention Bias: Scaling Initial Token Weight for Enhanced Long-Text Processing](http://arxiv.org/abs/2512.13109v1)

Zewen Qiang, Sendong Zhao, Haochun Wang, Bing Qin, Ting Liu

[PDF](https://arxiv.org/pdf/2512.13109v1)
[Abstract](http://arxiv.org/abs/2512.13109v1)

### 中文摘要

大型语言模型（LLMs）在多种自然语言处理（NLP）任务中展现出强大的性能。然而，由于“中段失焦”现象，它们在处理长文本序列时常常遇到困难。这一问题被证实源于一种“U型注意力偏差”，即模型对文本的起始和结束部分过度关注，而中间部分则被低估。尽管先前的研究多将此偏差归因于位置编码，但我们的研究首次发现了另一个影响因素：初始显著性。这意味着在每个词元的注意力计算中，相对于初始词元具有更高注意力权重的词元，在预测下一个词元时倾向于获得更多关注。我们进一步发现，利用这一特性，通过缩放初始词元与其他词元之间的注意力权重，能够提升模型处理长上下文的能力，在MDQA数据集上实现了最高3.6%的性能提升。此外，将此方法与现有的减弱位置编码偏差的策略结合使用，进一步增强了模型性能，在KV-Retrieval任务中最高提升达3.4%。

BibTeX

```
@article{2512.13109v1,
  title={Uncovering the Role of Initial Saliency in U-Shaped Attention Bias: Scaling Initial Token Weight for Enhanced Long-Text Processing},
  author={Zewen Qiang and Sendong Zhao and Haochun Wang and Bing Qin and Ting Liu},
  journal={arXiv preprint arXiv:2512.13109v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13109v1}
}
```

## [CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving](http://arxiv.org/abs/2512.11920v1)

Dong Liu, Yanxuan Yu

[PDF](https://arxiv.org/pdf/2512.11920v1)
[Abstract](http://arxiv.org/abs/2512.11920v1)

### 中文摘要

大型语言模型（LLMs）在自然语言处理任务中实现了革命性突破，但在数据中心环境中的部署仍面临显著挑战，主要由于关键值（KV）缓存所需的庞大内存资源。在自回归解码过程中，KV缓存会占用大量GPU内存，限制了批处理规模及系统整体吞吐量。为应对这些挑战，本文提出了“CXL-SpecKV”——一种新颖的异构KV缓存架构，利用Compute Express Link（CXL）互联技术和FPGA加速器，实现高效的推测执行和内存解耦。该方案引入了三项核心创新：（一）基于CXL的内存解耦框架，将KV缓存迁移至远端FPGA内存，降低延迟；（二）推测性KV缓存预取机制，预测并预加载未来Token的缓存项；（三）由FPGA加速的KV缓存压缩与解压引擎，有效降低内存带宽需求，提升至4倍。实验证明，在多种先进的LLMs模型上，CXL-SpecKV实现了比纯GPU方案高出最多3.2倍的吞吐量，内存成本降低约2.8倍，且保持了较高的准确率。该系统展示了结合智能内存解耦和推测执行，能够有效突破大规模LLM服务中的“内存墙”难题。相关代码已开源，地址为https://github.com/FastLM/CXL-SpecKV。

BibTeX

```
@article{2512.11920v1,
  title={CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving},
  author={Dong Liu and Yanxuan Yu},
  journal={arXiv preprint arXiv:2512.11920v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11920v1}
}
```

## [World Models Can Leverage Human Videos for Dexterous Manipulation](http://arxiv.org/abs/2512.13644v1)

Raktim Gautam Goswami, Amir Bar, David Fan, Tsung-Yen Yang, Gaoyue Zhou, Prashanth Krishnamurthy, Michael Rabbat, Farshad Khorrami, Yann LeCun

[PDF](https://arxiv.org/pdf/2512.13644v1)
[Abstract](http://arxiv.org/abs/2512.13644v1)

### 中文摘要

灵巧操控具有很大的挑战性，因为它需要理解细微的手部动作如何通过与物体接触影响环境。我们提出了DexWM，一种灵巧操控世界模型，能够在给定过去状态和灵巧动作的条件下预测环境的下一潜在状态。为了弥补灵巧操控数据集的不足，DexWM在超过900小时的人类和非灵巧机器人视频上进行训练。为了实现细粒度的操作能力，我们发现仅预测视觉特征不足以满足需求，因此引入了一种辅助的手部一致性损失，用以确保手部构型的准确性。实验结果显示，DexWM优于之前基于文本、导航或全身动作的世界模型，在预测未来环境状态方面具有更高的精确度。此外，DexWM在部署到装备有Allegro夹持器的弗朗卡潘达机械臂上时，展现出强大的零样本泛化能力，在抓取、放置和伸展等任务中，平均比Diffusion Policy性能提升超过50%。

BibTeX

```
@article{2512.13644v1,
  title={World Models Can Leverage Human Videos for Dexterous Manipulation},
  author={Raktim Gautam Goswami and Amir Bar and David Fan and Tsung-Yen Yang and Gaoyue Zhou and Prashanth Krishnamurthy and Michael Rabbat and Farshad Khorrami and Yann LeCun},
  journal={arXiv preprint arXiv:2512.13644v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13644v1}
}
```

## [DP-CSGP: Differentially Private Stochastic Gradient Push with Compressed Communication](http://arxiv.org/abs/2512.13583v1)

Zehan Zhu, Heng Zhao, Yan Huang, Joey Tianyi Zhou, Shouling Ji, Jinming Xu

[PDF](https://arxiv.org/pdf/2512.13583v1)
[Abstract](http://arxiv.org/abs/2512.13583v1)

### 中文摘要

本文提出了一种针对有向图的去中心化学习的差分隐私随机梯度推送压缩通信算法（简称DP-CSGP）。与现有工作不同，所提算法旨在在保证严格的差分隐私（DP）保障和高效通信的同时，维持较高的模型性能。对于一般的非凸光滑目标函数，我们证明了该算法在每个节点具有$\left(ε, δ\right)$-差分隐私保障的条件下，达到了紧致的性能界，即$\mathcal{O}\left( \sqrt{d\log \left( \frac{1}{δ} \right)}/(\sqrt{n}Jε) \right)$（其中，$J$为本地样本数，$d$为决策变量的维度），这一界与采用精确通信的去中心化方法相匹配。在多个基准任务上的大量实验结果表明，在相同的隐私预算条件下，DP-CSGP在模型准确率方面与现有的采用精确通信的去中心化算法相当，但显著降低了通信开销。

BibTeX

```
@article{2512.13583v1,
  title={DP-CSGP: Differentially Private Stochastic Gradient Push with Compressed Communication},
  author={Zehan Zhu and Heng Zhao and Yan Huang and Joey Tianyi Zhou and Shouling Ji and Jinming Xu},
  journal={arXiv preprint arXiv:2512.13583v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13583v1}
}
```

## [Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability](http://arxiv.org/abs/2512.13568v1)

Leonard Bereska, Zoe Tzifa-Kratira, Reza Samavi, Efstratios Gavves

[PDF](https://arxiv.org/pdf/2512.13568v1)
[Abstract](http://arxiv.org/abs/2512.13568v1)

### 中文摘要

神经网络通过叠加实现了卓越的性能：将多种特征编码为激活空间中的重叠方向，而非为每个特征单独分配神经元。这一机制挑战了模型的可解释性，但目前缺乏理论上的方法来衡量叠加现象。我们提出了一种基于信息理论的框架，用于衡量神经表示的有效自由度。具体地，我们利用香农熵对稀疏自编码器的激活进行分析，计算出在无干扰编码中所需的最少神经元数量，从而获得有效特征的数量。这等同于衡量网络通过叠加模拟的“虚拟神经元”数量。当网络编码的有效特征数超过实际神经元数量时，网络必须接受干扰作为压缩的代价。我们的指标在简易模型中与真实值高度相关，能够检测算法任务中的最小叠加程度，并展示了dropout情况下的系统性减少。这一层级分析的结构与Pythia-70M模型的内在维度研究相呼应。此外，该指标还能捕捉发展过程中的动态变化，比如在“领悟”（grokking）阶段的特征快速集中。令人惊讶的是，对抗训练不仅能增加有效特征数量，还能提升模型的鲁棒性，这与叠加导致脆弱的假设相悖。实际上，这一效果取决于任务复杂度和模型容量：简单任务和充裕容量下，模型倾向于特征扩展（丰富状态），而复杂任务或容量有限时，则表现为特征压缩（稀缺状态）。将叠加定义为一种有损压缩，本研究实现了对神经网络在有限计算资源条件下组织信息方式的理论量化，揭示了叠加与对抗鲁棒性之间的联系。

BibTeX

```
@article{2512.13568v1,
  title={Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability},
  author={Leonard Bereska and Zoe Tzifa-Kratira and Reza Samavi and Efstratios Gavves},
  journal={arXiv preprint arXiv:2512.13568v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13568v1}
}
```

## [MedCEG: Reinforcing Verifiable Medical Reasoning with Critical Evidence Graph](http://arxiv.org/abs/2512.13510v1)

Linjie Mu, Yannian Gu, Zhongzhen Huang, Yakun Zhu, Shaoting Zhang, Xiaofan Zhang

[PDF](https://arxiv.org/pdf/2512.13510v1)
[Abstract](http://arxiv.org/abs/2512.13510v1)

### 中文摘要

具有推理能力的大型语言模型在多个领域都展现出令人印象深刻的性能。在临床应用中，透明的逐步推理过程为医生提供有力的证据支持决策制定。尽管强化学习已有效提升了医学场景中的推理表现，但由于在训练过程中往往忽视了推理过程的准确性和有效性，因而其临床可靠性仍然有限。为弥补这一空白，本文提出了MedCEG框架，通过显式监督推理过程中的关键证据图（Critical Evidence Graph, CEG），增强医疗语言模型的临床有效推理路径。我们整理了一个具有挑战性的临床案例数据集，并为每个样本算法性构建了对应的CEG，用以表示高质量且可验证的推理路径。为了指导推理过程，我们引入了临床推理流程奖励机制（Clinical Reasoning Procedure Reward），该机制通过评估节点覆盖率、结构正确性和链条完整性，从整体上衡量推理的优劣。实验结果表明，MedCEG在性能上优于现有方法，同时生成具有临床有效性的推理链条，标志着在可靠医疗人工智能推理方面取得了重要突破。相关代码和模型可在https://github.com/LinjieMu/MedCEG获取。

BibTeX

```
@article{2512.13510v1,
  title={MedCEG: Reinforcing Verifiable Medical Reasoning with Critical Evidence Graph},
  author={Linjie Mu and Yannian Gu and Zhongzhen Huang and Yakun Zhu and Shaoting Zhang and Xiaofan Zhang},
  journal={arXiv preprint arXiv:2512.13510v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13510v1}
}
```

## [Context-Aware Agentic Power Resources Optimisation in EV using Smart2ChargeApp](http://arxiv.org/abs/2512.12048v1)

Muddsair Sharif, Huseyin Seker

[PDF](https://arxiv.org/pdf/2512.12048v1)
[Abstract](http://arxiv.org/abs/2512.12048v1)

### 中文摘要

本文提出了一种用于动态资源分配的新型情境感知多智能体协作框架（CAMAC-DRA），旨在通过Smart2Charge应用优化智能电动车（EV）充电生态系统。该系统在涵盖250辆电动车和45个充电站的网络中协调自主充电代理，并通过情境感知的决策机制应对动态环境变化。我们的多智能体方法采用了结合图神经网络和注意力机制的协同深度Q网络（Deep Q-Network, DQN），处理包括天气变化、交通状况、电网负荷波动和电价等20个情境特征。该框架通过加权协调机制和共识协议，在电动车用户（25%）、电网运营商（20%）、充电站运营商（20%）、车队运营商（20%）及环境因素（15%）之间实现利益平衡。基于包含441,077笔充电交易的实际数据集进行的全面验证显示，该方法优于包括DDPG、A3C、PPO和GNN在内的基线算法，达到92%的协作成功率，能源利用效率提升15%，成本降低10%，电网压力减轻20%，并在保持88%的训练稳定性和85%的样本效率的同时，实现约2.3倍的收敛速度。真实环境中的验证结果表明，该框架具有商业应用前景，净现值为负122,962美元，结合可再生能源实现了69%的成本降低。该方法的创新之处在于开发了具有高度情境感知能力的多利益相关者协作机制，有效平衡了多重目标，并能实时适应变量变化，标志着智能电动车充电协调与可持续交通电气化领域的突破性解决方案。

BibTeX

```
@article{2512.12048v1,
  title={Context-Aware Agentic Power Resources Optimisation in EV using Smart2ChargeApp},
  author={Muddsair Sharif and Huseyin Seker},
  journal={arXiv preprint arXiv:2512.12048v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12048v1}
}
```

## [TA-KAND: Two-stage Attention Triple Enhancement and U-KAN based Diffusion For Few-shot Knowledge Graph Completion](http://arxiv.org/abs/2512.12182v1)

Xinyu Gao

[PDF](https://arxiv.org/pdf/2512.12182v1)
[Abstract](http://arxiv.org/abs/2512.12182v1)

### 中文摘要

知识图谱（KG）凭借其简洁高效的三元组结构，已广泛应用于智能问答、推荐系统等多个领域。然而，现实世界数据的异质性和多样性不可避免地导致关系分布呈长尾分布，在有限样本条件下完成缺失事实变得尤为关键。以往的研究主要基于度量匹配或元学习方法，但它们要么未能充分利用图中的邻域信息，要么忽视了对比信号的分布特征。本文从生成式表示的角度重新审视该问题，提出了一种结合两阶段注意力三元组增强器与U-KAN扩散模型的少样本知识图谱补全框架。在两个公共数据集上的大量实验证明，该方法达到了新的性能最优水平。

BibTeX

```
@article{2512.12182v1,
  title={TA-KAND: Two-stage Attention Triple Enhancement and U-KAN based Diffusion For Few-shot Knowledge Graph Completion},
  author={Xinyu Gao},
  journal={arXiv preprint arXiv:2512.12182v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12182v1}
}
```

## [Detecting Emotion Drift in Mental Health Text Using Pre-Trained Transformers](http://arxiv.org/abs/2512.13363v1)

Shibani Sankpal

[PDF](https://arxiv.org/pdf/2512.13363v1)
[Abstract](http://arxiv.org/abs/2512.13363v1)

### 中文摘要

本研究探讨了情感漂移现象，即在与心理健康相关的消息中，单篇文本中情感状态的变化。尽管情感分析通常将整条信息归类为积极、消极或中性，但在信息流中情感细微转变的过程常被忽视。本研究采用预训练的变换模型，如DistilBERT和RoBERTa，进行句子水平的情感检测，并衡量情感漂移分数。研究结果揭示了心理健康对话中情感升级或缓解的模式，为理解内容中的情感动态提供了新的视角。

BibTeX

```
@article{2512.13363v1,
  title={Detecting Emotion Drift in Mental Health Text Using Pre-Trained Transformers},
  author={Shibani Sankpal},
  journal={arXiv preprint arXiv:2512.13363v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13363v1}
}
```

## [Face Identity Unlearning for Retrieval via Embedding Dispersion](http://arxiv.org/abs/2512.13317v1)

Mikhail Zakharov

[PDF](https://arxiv.org/pdf/2512.13317v1)
[Abstract](http://arxiv.org/abs/2512.13317v1)

### 中文摘要

面部识别系统依赖于学习具有高度区分性和紧凑性的身份簇，以实现精准的检索。然而，类似于其他面向监控的技术，此类系统因其可能被用于未经授权的身份追踪而引发严重的隐私担忧。尽管已有若干研究探讨了机器遗忘（machine unlearning）作为隐私保护的一种手段，但其在面部检索，特别是基于现代嵌入模型的识别系统中的应用仍然基本未被深入研究。在本研究中，我们探讨了面部身份多次遗忘（identity unlearning）在检索系统中的问题及其固有挑战。其目标是通过在超球面上扩散目标身份的嵌入向量，阻止形成有助于重识别的紧凑身份簇，从而使特定身份不可检索。主要挑战在于在实现遗忘效果的同时，保留嵌入空间的判别结构及模型对剩余身份的检索性能。为此，我们在面部检索的背景下评估了几种现有的近似类别遗忘方法（如随机标签、梯度上升、边界遗忘等，以及其他最新方法），并提出了一种简单且有效的基于扩散的遗忘策略。在标准基准数据集（VGGFace2、CelebA）上的大量实验证明，该方法在实现优越的遗忘效果的同时，能够保持检索的实用性。

BibTeX

```
@article{2512.13317v1,
  title={Face Identity Unlearning for Retrieval via Embedding Dispersion},
  author={Mikhail Zakharov},
  journal={arXiv preprint arXiv:2512.13317v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13317v1}
}
```

## [FIN-bench-v2: A Unified and Robust Benchmark Suite for Evaluating Finnish Large Language Models](http://arxiv.org/abs/2512.13330v1)

Joona Kytöniemi, Jousia Piha, Akseli Reunamo, Fedor Vitiugin, Farrokh Mehryary, Sampo Pyysalo

[PDF](https://arxiv.org/pdf/2512.13330v1)
[Abstract](http://arxiv.org/abs/2512.13330v1)

### 中文摘要

我们推出了 FIN-bench-v2，这是一个用于评估芬兰语大规模语言模型的统一基准套件。FIN-bench-v2 将广泛使用的基准测试的芬兰语版本与原始 FIN-bench 的更新和扩展版本整合为一个格式统一的集合，涵盖了多项选择和生成任务，涉及阅读理解、常识推理、情感分析、世界知识和模型对齐等多个领域。所有数据集均转换为 HuggingFace 数据集格式，包括每个任务的五个变体的完形填空和多项选择提示形式，并对如 GoldenSwag 和 XED 等机器翻译资源进行了人工标注或审查。为了筛选具有鲁棒性的任务，我们预训练了一组参数为 2.15B 的解码器模型，并通过其学习曲线计算单调性、信噪比、非随机性能和模型排序的一致性，仅保留满足全部标准的任务。我们还对一组更大规模的指令调优模型进行了评估，以分析其在不同任务和提示形式下的表现。所有数据集、提示和评估配置均在我们的 Language Model Evaluation Harness 项目分支（https://github.com/LumiOpen/lm-evaluation-harness）上公开提供。补充资源则在另一个仓库（https://github.com/TurkuNLP/FIN-bench-v2）中发布。

BibTeX

```
@article{2512.13330v1,
  title={FIN-bench-v2: A Unified and Robust Benchmark Suite for Evaluating Finnish Large Language Models},
  author={Joona Kytöniemi and Jousia Piha and Akseli Reunamo and Fedor Vitiugin and Farrokh Mehryary and Sampo Pyysalo},
  journal={arXiv preprint arXiv:2512.13330v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13330v1}
}
```

## [Non-Resolution Reasoning (NRR): A Computational Framework for Contextual Identity and Ambiguity Preservation](http://arxiv.org/abs/2512.13478v2)

Kei Saito

[PDF](https://arxiv.org/pdf/2512.13478v2)
[Abstract](http://arxiv.org/abs/2512.13478v2)

### 中文摘要

目前的人工智能系统，尽管在文本生成和模式识别方面表现出卓越的能力，但在架构层面存在根本性限制：它们过早地解决了歧义。这种过早的语义崩溃——即将多个有效解释压缩成单一输出的倾向——源于嵌入在标准神经网络结构中的经典身份假设。我们提出了一种非分辨推理（NRR）框架，该框架将歧义的保持视为一种有效的推理方式，而非应当消除的缺陷。NRR引入了三条核心原则：(1) 非身份（A ≠ A）——同一符号在不同语境中指代不同实体；(2) 近似身份（A ≈ A）——实体具有部分结构上的重叠但并非完全相同；(3) 非解决（Non-Resolution）——冲突的解释可以共存，无需强制收敛。我们通过三种架构组件形式化这些原则：多向量嵌入用于实现依赖上下文的表征、非崩塌注意机制用于并行保留多重解释以及上下文身份追踪（CIT）用于维护推理过程中的A ≠ A关系。我们通过在悖论处理、创造性生成和上下文依赖推理等案例中展示NRR的优势。特别是，我们在一个合成的上下文转变任务上进行了最小的实证验证，结果显示一个简化版的NRR模型在分布外准确率达90.9%，而标准架构仅为9.1%，证明了歧义保持有助于结构化泛化。NRR挑战了“意义必须崩塌才能有用”的假设，为实现具有复杂歧义处理和创造性推理能力的人工智能系统奠定了基础。关键问题不在于AI是否应当解决歧义，而在于何时、如何以及由谁来控制这种解决过程。

BibTeX

```
@article{2512.13478v2,
  title={Non-Resolution Reasoning (NRR): A Computational Framework for Contextual Identity and Ambiguity Preservation},
  author={Kei Saito},
  journal={arXiv preprint arXiv:2512.13478v2},
  year={2025},
  url={http://arxiv.org/abs/2512.13478v2}
}
```

## [Fault-Tolerant Sandboxing for AI Coding Agents: A Transactional Approach to Safe Autonomous Execution](http://arxiv.org/abs/2512.12806v1)

Boyang Yan

[PDF](https://arxiv.org/pdf/2512.12806v1)
[Abstract](http://arxiv.org/abs/2512.12806v1)

### 中文摘要

大型语言模型（LLMs）从被动的代码生成器向自主智能体的转变带来了显著的安全风险，尤其涉及到破坏性命令和系统状态不一致的问题。现有的商业解决方案通常优先考虑用户交互的安全性，通过认证壁垒防止非授权操作，但这会破坏实现真正自主的无头循环。本文提出了一种容错沙箱框架，旨在通过基于策略的拦截层和事务性文件系统快照机制来减轻这些风险。我们假设将智能体操作置于原子事务中，可以在保证安全性的同时实现可接受的延迟，从而优于容器的重初始化开销或商业CLI交互的摩擦。我们在一个基于EVPN/VXLAN隔离的定制Proxmox测试平台上，利用Nano-vLLM部署了Minimind-MoE大模型进行验证。试验结果表明，系统对高风险命令的拦截率达到100%，且在失败后能以100%的成功率回滚到安全状态。关键是，我们的原型每次事务仅增加约14.5%的性能开销（约1.8秒）。相比之下，对Gemini CLI沙箱的基准测试显示，它需要交互式的“登录”验证，无法用于无头的自主智能体工作流程。

BibTeX

```
@article{2512.12806v1,
  title={Fault-Tolerant Sandboxing for AI Coding Agents: A Transactional Approach to Safe Autonomous Execution},
  author={Boyang Yan},
  journal={arXiv preprint arXiv:2512.12806v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12806v1}
}
```

## [Causal Counterfactuals Reconsidered](http://arxiv.org/abs/2512.12804v1)

Sander Beckers

[PDF](https://arxiv.org/pdf/2512.12804v1)
[Abstract](http://arxiv.org/abs/2512.12804v1)

### 中文摘要

我提出了一种新颖的反事实概率语义，超越了标准的Pearl语义：它适用于不能扩展为现实结构性因果模型、因此超出Pearl语义适用范围的概率性因果模型。这一泛化是必要的，因为正如我所展示，即使在简单的场景中，也会出现此类概率性因果模型。我的语义提供了一个在Pearl和Dawid关于反事实的长期争论中的自然折中方案：我认同Dawid关于应拒绝普遍因果决定论及非现实变量的观点，但同时也认同Pearl关于反事实具有一般语义的可能性。我限制研究的因果模型须满足马尔可夫条件、只包含现实变量且具有因果完备性。尽管我的方案是基于结构性因果模型——与Pearl类似，但我避免使用所谓的响应变量。此外，我证明了我的语义与另外两种近期提出的不涉及结构性因果模型的方案等价，并且它与文献中更广泛出现的关于随机性反事实的多种评论保持一致。整个过程中，我还反思了马尔可夫条件的普遍性，并探索了一种关于因果抽象的新颖推广。

BibTeX

```
@article{2512.12804v1,
  title={Causal Counterfactuals Reconsidered},
  author={Sander Beckers},
  journal={arXiv preprint arXiv:2512.12804v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12804v1}
}
```

## [End2Reg: Learning Task-Specific Segmentation for Markerless Registration in Spine Surgery](http://arxiv.org/abs/2512.13402v1)

Lorenzo Pettinari, Sidaty El Hadramy, Michael Wehrli, Philippe C. Cattin, Daniel Studer, Carol C. Hasler, Maria Licci

[PDF](https://arxiv.org/pdf/2512.13402v1)
[Abstract](http://arxiv.org/abs/2512.13402v1)

### 中文摘要

目的：脊柱手术中的术中导航要求达到毫米级的高精度。目前基于术中放射成像和骨钉标记的系统具有侵入性强、辐射量大且影响手术流程的缺点。近年来，基于无标记RGB-D图像的注册方法提供了一种有前景的替代方案，但现有方法依赖于对相关解剖结构的弱监督分割标签，这可能会引入误差并影响注册效果。方法：本文提出了End2Reg，一种端到端的深度学习框架，能够同时优化分割与注册任务，免除对弱标记分割及人工步骤的依赖。该网络学习专门为注册优化的分割掩码，并完全由注册目标引导，无需直接的分割监督。结果：所提出的框架在体外和体内基准测试中均达到行业领先性能，将目标注册误差的中位数降低了32%，至1.83mm，平均均方根误差降低了45%，至3.95mm。消融实验验证了端到端优化显著提升了注册精度。结论：所提出的端到端RGB-D注册流程去除了对弱监督标签和手动操作的依赖，迈近了实现全自动、无标记的术中导航的目标。相关代码和交互式可视化结果可在https://lorenzopettinari.github.io/end-2-reg/获取。

BibTeX

```
@article{2512.13402v1,
  title={End2Reg: Learning Task-Specific Segmentation for Markerless Registration in Spine Surgery},
  author={Lorenzo Pettinari and Sidaty El Hadramy and Michael Wehrli and Philippe C. Cattin and Daniel Studer and Carol C. Hasler and Maria Licci},
  journal={arXiv preprint arXiv:2512.13402v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13402v1}
}
```

## [Control of a Twin Rotor using Twin Delayed Deep Deterministic Policy Gradient (TD3)](http://arxiv.org/abs/2512.13356v1)

Zeyad Gamal, Youssef Mahran, Ayman El-Badawy

[PDF](https://arxiv.org/pdf/2512.13356v1)
[Abstract](http://arxiv.org/abs/2512.13356v1)

### 中文摘要

本文提出了一种基于强化学习（RL）的控制框架，用于对双转子空气动力系统（TRAS）在特定俯仰角和方位角下实现稳定控制及轨迹跟踪。由于TRAS具有复杂的动力学特性和非线性特性，传统控制算法难以实现有效控制。然而，近年来强化学习的快速发展引起了研究者的关注，其在多旋翼无人机等多自由度系统控制中的潜在应用尤为突出。本文采用双延迟深度确定性策略梯度（TD3）算法对RL智能体进行训练。该算法适用于状态空间和动作空间连续的环境，与TRAS的特性相契合，且无需建立系统的精确模型。仿真结果证实了该RL控制方法的有效性。在此基础上，加入风扰动作为外部干扰，验证控制器在面对环境干扰时优于传统的PID控制器。最后，通过在实验室平台上的实际试验，进一步验证了该控制器在实际应用中的可行性与有效性。

BibTeX

```
@article{2512.13356v1,
  title={Control of a Twin Rotor using Twin Delayed Deep Deterministic Policy Gradient (TD3)},
  author={Zeyad Gamal and Youssef Mahran and Ayman El-Badawy},
  journal={arXiv preprint arXiv:2512.13356v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13356v1}
}
```

## [MiniLingua: A Small Open-Source LLM for European Languages](http://arxiv.org/abs/2512.13298v1)

Anna Aksenova, Boris Zverkov, Nicola Dainese, Alexander Nikitin, Pekka Marttinen

[PDF](https://arxiv.org/pdf/2512.13298v1)
[Abstract](http://arxiv.org/abs/2512.13298v1)

### 中文摘要

大型语言模型具有强大的能力，但常受限于高昂的计算成本、隐私问题以及以英语为中心的训练方式。近期的研究表明，规模约为十亿参数的紧凑高效模型也能够取得优异的效果，并支持在设备端部署。本文推出了MiniLingua，一款多语种开源大模型，拥有十亿参数，從零开始训练，涵盖13个欧洲语言，旨在在覆盖范围和指令遵循能力之间取得平衡。评估结果显示，经过指令微调的MiniLingua在文本摘要、分类以及开放式和封闭式问答任务中，均优于采用类似训练策略但训练预算更大的EuroLLM模型。此外，它在开放式生成任务中还能与更先进的前沿模型保持竞争。我们公开了模型权重、分词器以及用于数据处理和模型训练的源代码。

BibTeX

```
@article{2512.13298v1,
  title={MiniLingua: A Small Open-Source LLM for European Languages},
  author={Anna Aksenova and Boris Zverkov and Nicola Dainese and Alexander Nikitin and Pekka Marttinen},
  journal={arXiv preprint arXiv:2512.13298v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13298v1}
}
```

## [Reflective Preference Optimization (RPO): Enhancing On-Policy Alignment via Hint-Guided Reflection](http://arxiv.org/abs/2512.13240v1)

Zihui Zhao, Zechang Li

[PDF](https://arxiv.org/pdf/2512.13240v1)
[Abstract](http://arxiv.org/abs/2512.13240v1)

### 中文摘要

直接偏好优化（DPO）已成为一种轻量且有效的替代方法，用于取代基于人类反馈的强化学习（RLHF）和基于人工智能反馈的强化学习（RLAIF），以实现大型语言模型和视觉-语言模型的对齐。然而，标准的DPO公式中，生成的被选择响应和被拒绝响应均来自同一策略，这导致学习信号较弱，因为两者常具有相似的错误且Kullback-Leibler（KL）散度较小，从而引发收敛速度缓慢且不稳定的问题。为了解决这一限制，本文提出了反思偏好优化（RPO）这一新框架，该框架在DPO范式中引入提示引导的反思机制。RPO利用外部模型识别幻觉源并生成简洁的反思提示，从而构建具有更强对比度和更明确信号的在策略偏好对。我们在理论上证明，条件化提示能够通过互信息增加预期偏好边界，提升采样效率，同时仍保持在策略分布家族之内。在实际应用中，RPO在较少的训练样本和迭代次数下实现了更优的对齐效果，显著降低了幻觉发生率，并在多模态基准测试中取得了最先进的性能。

BibTeX

```
@article{2512.13240v1,
  title={Reflective Preference Optimization (RPO): Enhancing On-Policy Alignment via Hint-Guided Reflection},
  author={Zihui Zhao and Zechang Li},
  journal={arXiv preprint arXiv:2512.13240v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13240v1}
}
```

## [WAY: Estimation of Vessel Destination in Worldwide AIS Trajectory](http://arxiv.org/abs/2512.13190v1)

Jin Sob Kim, Hyun Joon Park, Wooseok Shin, Dongil Park, Sung Won Han

[PDF](https://arxiv.org/pdf/2512.13190v1)
[Abstract](http://arxiv.org/abs/2512.13190v1)

### 中文摘要

自动识别系统（AIS）作为一种数据驱动的海事监控工具，虽然具有广泛应用，但在可靠性和数据间隔的不规则性方面尚存挑战。本文针对利用全球范围AIS数据进行船舶目的地预测的问题，提出一种差异化方法，将长距离的港口间轨迹重新构建为嵌套序列结构。该方法采用空间划分网格，有效缓解了时空偏差，同时保持了细粒度的轨迹信息。我们引入了一种新颖的深度学习架构——WAY，旨在处理这些经过重构的轨迹数据，以实现提前数天至数周的长远目的地预测。WAY由轨迹表达层和通道聚合序列处理（CASP）模块组成。轨迹表达层从动力学和非动力学特征中生成多通道向量序列，CASP模块利用多头通道注意力和自注意力机制进行特征聚合和序列信息传递。此外，本文还提出了一种任务专用的梯度随机失活（GD）技术，支持多对多的训练方式，单一标签下的训练避免偏倚反馈激增，通过随机阻断梯度流根据样本长度调节信息传递。在五年AIS数据集上的实验结果显示，WAY在轨迹行进过程中优于传统的空间网格方法，其性能稳定且具备显著优势。同时，采用GD技术进一步提升了模型的预测准确性。最后，我们通过多任务学习探索WAY在实际应用中的潜力，具体包括预测到达时间（ETA）等多种任务。

BibTeX

```
@article{2512.13190v1,
  title={WAY: Estimation of Vessel Destination in Worldwide AIS Trajectory},
  author={Jin Sob Kim and Hyun Joon Park and Wooseok Shin and Dongil Park and Sung Won Han},
  journal={arXiv preprint arXiv:2512.13190v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13190v1}
}
```

## [PolySet: Restoring the Statistical Ensemble Nature of Polymers for Machine Learning](http://arxiv.org/abs/2512.13186v1)

Khalid Ferji

[PDF](https://arxiv.org/pdf/2512.13186v1)
[Abstract](http://arxiv.org/abs/2512.13186v1)

### 中文摘要

在聚合物科学的机器学习（ML）模型中，通常将聚合物作为一个单一、完全定义的分子图进行处理，尽管实际材料由具有分布链长的随机链系组成。这种物理现实与数字表示之间的不匹配，限制了当前模型在捕捉聚合物行为方面的能力。本文引入了PolySet框架，它将聚合物表示为从假设的摩尔质量分布中采样得到的有限加权链系集合。这种基于链系的编码方式不依赖于具体的化学细节，兼容任何分子表示方法，并在此以单一聚合物的示例，采用最简语言模型进行说明。我们证明了PolySet能够保留更高阶的分布矩（如Mz、Mz+1），使得机器学习模型可以学习对尾部敏感的性质，从而显著提升稳定性和准确性。通过明确承认聚合物的统计特性，PolySet为未来的聚合物机器学习建立了一个具有物理基础的框架，具有自然的可扩展性，适用于共聚物、块状结构及其他复杂拓扑的材料。

BibTeX

```
@article{2512.13186v1,
  title={PolySet: Restoring the Statistical Ensemble Nature of Polymers for Machine Learning},
  author={Khalid Ferji},
  journal={arXiv preprint arXiv:2512.13186v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13186v1}
}
```

## [Towards Unified Co-Speech Gesture Generation via Hierarchical Implicit Periodicity Learning](http://arxiv.org/abs/2512.13131v1)

Xin Guo, Yifan Zhao, Jia Li

[PDF](https://arxiv.org/pdf/2512.13131v1)
[Abstract](http://arxiv.org/abs/2512.13131v1)

### 中文摘要

利用语音生成三维人体动作在广泛的下游应用中展现出巨大潜力，但在模拟自然人类动作方面仍面临挑战。目前的研究主要集中在端到端的生成方案，用于生成配音手势，涵盖了GAN、VQ-VAE以及最近的扩散模型。作为一个病态问题，本文指出这些主流的学习方法未能有效建模不同运动单元（如头部、身体和手部）之间以及内部的关键关联，从而导致动作不自然且缺乏协调性。为深入探索这些内在关联，我们提出了一种用于音频驱动的三维手势生成的统一层次隐式周期性（HIP）学习方法。不同于现有的研究，我们的方法通过两个明确的技术洞见来建模这种多模态的隐式关系：一是为了解开复杂的手势运动，我们首先利用周期性自编码器探索手势运动的相位流形，以模仿人体自然的真实分布，同时结合来自当前潜在状态的非周期性运动，实现实例级的多样性；二是为了建模面部动作、身体手势和手部动作之间的层次关系，引入级联指导机制，驱动动画的学习过程。我们在3D虚拟形象上进行了验证，大量实验结果表明，该方法在定量和定性评价中均优于当前最先进的配音手势生成技术。相关代码和模型将公开发布。

BibTeX

```
@article{2512.13131v1,
  title={Towards Unified Co-Speech Gesture Generation via Hierarchical Implicit Periodicity Learning},
  author={Xin Guo and Yifan Zhao and Jia Li},
  journal={arXiv preprint arXiv:2512.13131v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13131v1}
}
```

## [M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](http://arxiv.org/abs/2512.13070v1)

Bizhe Bai, Hongming Wu, Peng Ye, Tao Chen

[PDF](https://arxiv.org/pdf/2512.13070v1)
[Abstract](http://arxiv.org/abs/2512.13070v1)

### 中文摘要

自监督强化学习（RL）为提升大型语言模型（LLMs）的推理能力提供了一种具有潜力的方法，且无需依赖昂贵的人类标注数据。然而，我们发现现有方法在长远训练过程中存在一个关键的失败模式：即“策略崩溃”，表现为性能急剧下降。我们分析了这种不稳定性，并证明简调节回合数——一种常用的提升性能的策略——只能延迟而无法根除这种崩溃。为应对这一问题，我们首先提出M-GRPO（动量锚定的群体相对策略优化）框架，该方法利用一个缓慢变化的动量模型作为稳定的训练目标。此外，我们发现此过程通常伴随着策略熵的迅速降低，导致策略过早变得过于自信，进而陷入次优。针对这一问题，我们提出第二项创新：一种基于四分位数范围（IQR）的自适应滤波方法，能够动态剔除低熵轨迹，从而保留策略的多样性。大量在多个推理基准上的实验证明，M-GRPO能够稳定训练过程，而IQR滤波器则有效防止过早收敛。这两项创新的结合显著提升了训练的稳定性和模型的性能，达到了最先进的水平。

BibTeX

```
@article{2512.13070v1,
  title={M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization},
  author={Bizhe Bai and Hongming Wu and Peng Ye and Tao Chen},
  journal={arXiv preprint arXiv:2512.13070v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13070v1}
}
```

## [Large-Language Memorization During the Classification of United States Supreme Court Cases](http://arxiv.org/abs/2512.13654v1)

John E. Ortega, Dhruv D. Joshi, Matt P. Borkowski

[PDF](https://arxiv.org/pdf/2512.13654v1)
[Abstract](http://arxiv.org/abs/2512.13654v1)

### 中文摘要

大型语言模型（LLMs）在问答之外的分类任务中已被证明能够产生多样化的响应。其响应有时被称为“幻觉”，因为输出结果并非预期内容。关于LLMs中的记忆策略正进行深入研究，旨在理解模型的响应机制。本研究对基于美国最高法院（SCOTUS）判决的分类任务进行了详细探讨。由于存在句子长度长、法律术语复杂、结构不规范以及专门领域词汇丰富等诸多挑战，SCOTUS语料库成为研究LLMs记忆准确性理想的分类任务。我们结合最新的微调技术和检索方法（如参数高效微调、自主建模等）对两个传统类别的SCOTUS分类任务进行了实验：一个涵盖15个标签主题，另一个则包含279个。结果显示，采用提示和记忆技术的模型，如DeepSeek，在两个任务中都展现出比以往基于BERT的模型更强的鲁棒性，性能提升约2个百分点，优于非提示型模型。

BibTeX

```
@article{2512.13654v1,
  title={Large-Language Memorization During the Classification of United States Supreme Court Cases},
  author={John E. Ortega and Dhruv D. Joshi and Matt P. Borkowski},
  journal={arXiv preprint arXiv:2512.13654v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13654v1}
}
```

## [SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping](http://arxiv.org/abs/2512.13494v1)

Yu-Chen Lu, Sheng-Feng Yu, Hui-Hsien Weng, Pei-Shuo Wang, Yu-Fang Hu, Liang Hung-Chun, Hung-Yueh Chiang, Kai-Chiang Wu

[PDF](https://arxiv.org/pdf/2512.13494v1)
[Abstract](http://arxiv.org/abs/2512.13494v1)

### 中文摘要

大型语言模型（LLM）在多种任务中取得了卓越的性能，但其庞大的参数规模为在计算和存储资源有限的边缘设备上部署带来了重大挑战。低秩压缩是一种具有潜力的解决方案，它可以同时减少计算和存储成本，从而使LLM更适合资源受限的环境。然而，单纯的低秩压缩方法通常需要显著降低模型的秩值，才能实现有意义的存储和计算节省。对于低秩模型而言，秩值通常需要减少一半以上，才能带来效率提升，但这种剧烈的裁剪往往导致性能的大幅下降。为了解决这一权衡问题，我们提出了SkipCat，一种新颖的低秩压缩框架，能够在实现相同压缩率的同时，使用更高的秩值。首先，我们引入一种层内共享的低秩投影技术，在该方法中，多个共享相同输入的矩阵采用公共投影，从而降低冗余、提升压缩效率。其次，我们提出一种块跳跃技术，能够在低秩分解中省略部分子块的计算和存储传输。这两项技术共同使我们的压缩模型在相同的压缩预算下，能够保留更多的有效秩。实验结果显示，在无需额外微调的情况下，我们的方法在相同压缩率的零样本任务中，比以往低秩压缩方法达到7%的准确率提升。这些结果充分体现了我们秩最大化压缩策略在资源受限环境下保持模型性能的有效性。

BibTeX

```
@article{2512.13494v1,
  title={SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping},
  author={Yu-Chen Lu and Sheng-Feng Yu and Hui-Hsien Weng and Pei-Shuo Wang and Yu-Fang Hu and Liang Hung-Chun and Hung-Yueh Chiang and Kai-Chiang Wu},
  journal={arXiv preprint arXiv:2512.13494v1},
  year={2025},
  url={http://arxiv.org/abs/2512.13494v1}
}
```

## [Entropy Collapse: A Universal Failure Mode of Intelligent Systems](http://arxiv.org/abs/2512.12381v1)

Truong Xuan Khanh, Truong Quynh Hoa

[PDF](https://arxiv.org/pdf/2512.12381v1)
[Abstract](http://arxiv.org/abs/2512.12381v1)

### 中文摘要

智能系统普遍被认为通过学习、协调和优化而不断提升。然而，在不同领域——从人工智能到经济制度再到生物进化——越来越高的智能水平往往伴随着悖论性的退化：系统变得僵化，失去适应能力，甚至出现意外失效。我们将“熵崩塌”定义为一种普遍的动态失效模式，发生在反馈放大超过有限创新再生时。在最小的跨领域假设下，我们证明了智能系统会经历一个由高熵适应性状态到低熵崩塌状态的剧烈转变。崩塌被形式化为趋向一个稳定的低熵流形，而非零熵状态，这意味着有效适应维度的收缩，而非活动性或规模的丧失。我们通过解析方式确立了临界阈值、动态不可逆性以及吸引子结构，并利用最小的模拟展示了该理论在不同更新机制中的普适性。该框架将人工智能中的模型崩溃、经济学中的制度僵化以及进化中的遗传瓶颈等多样现象统一为同一基础过程的表现。通过将崩塌重新定义为智能的结构性代价，我们阐明了为何晚期干预往往系统性失败，并提出了以熵为导向的设计原则，以促进智能系统的长期适应性持续发展。
关键词：熵崩塌；智能系统；反馈放大；相变；有效维度；复杂系统；模型崩溃；制度僵化

BibTeX

```
@article{2512.12381v1,
  title={Entropy Collapse: A Universal Failure Mode of Intelligent Systems},
  author={Truong Xuan Khanh and Truong Quynh Hoa},
  journal={arXiv preprint arXiv:2512.12381v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12381v1}
}
```

## [Quantum-Aware Generative AI for Materials Discovery: A Framework for Robust Exploration Beyond DFT Biases](http://arxiv.org/abs/2512.12288v1)

Mahule Roy, Guillaume Lambard

[PDF](https://arxiv.org/pdf/2512.12288v1)
[Abstract](http://arxiv.org/abs/2512.12288v1)

### 中文摘要

传统材料发现的生成模型主要基于密度泛函理论（DFT）及其近似交换-关联泛函进行训练和验证。这带来了根本性的瓶颈：这些模型继承了DFT在强相关体系方面的系统性失效，导致探索偏差以及在DFT预测出现定性错误的材料中难以发现潜在候选。我们提出了一种具有量子感知的生成式人工智能框架，系统性地解决了这一限制，通过紧密结合多保真学习和主动验证技术。该方法采用以量子力学描述符为条件的扩散模型作为生成器，并配备一个利用等变神经网络势的验证器，该势模型是在涵盖多层级理论（PBE、SCAN、HSE06、CCSD(T)）的分层数据集上训练而成。关键在于，我们实现了一个稳健的主动学习循环，用于量化并针对低保真和高保真预测之间的偏差进行优化。通过全面的消融研究，我们分析了各组成部分的贡献，详细探讨了失效模式，并在多个具有挑战性材料类别（如强相关氧化物）上，将框架与最新的生成模型（如CDVAE、GNoME、DiffCSP）进行了对比基准测试。结果显示，我们的方法在识别潜在稳定候选材料方面取得了显著的实际提升：在高偏差区域（例如强相关氧化物）成功率比传统的DFT单一保真模型提高了3至5倍，同时保持了较好的计算效率。这项工作为超越单一保真模型限制、扩展计算材料发现搜索空间提供了一个严谨、透明的框架。

BibTeX

```
@article{2512.12288v1,
  title={Quantum-Aware Generative AI for Materials Discovery: A Framework for Robust Exploration Beyond DFT Biases},
  author={Mahule Roy and Guillaume Lambard},
  journal={arXiv preprint arXiv:2512.12288v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12288v1}
}
```

## [A Multi-Axial Mindset for Ontology Design Lessons from Wikidata's Polyhierarchical Structure](http://arxiv.org/abs/2512.12260v1)

Ege Atacan Doğan, Peter F. Patel-Schneider

[PDF](https://arxiv.org/pdf/2512.12260v1)
[Abstract](http://arxiv.org/abs/2512.12260v1)

### 中文摘要

传统的本体设计强调互斥且穷尽的顶层区分，例如持续实体与发生实体、抽象与具体、类别与实例等。这些区分用于构建统一的层次结构，将每个实体归入单一的上层类别。而维基数据（Wikidata）则不强制采用单一的基础分类体系。相反，它在共享的根类实体下同时容纳多种分类轴，实现多重分类。这篇论文分析了维基数据多重层级和多轴设计的结构性影响。维基数据的架构支持一种可扩展且模块化的本体构建方式，特别适合于协作性和不断发展的知识图谱。

BibTeX

```
@article{2512.12260v1,
  title={A Multi-Axial Mindset for Ontology Design Lessons from Wikidata's Polyhierarchical Structure},
  author={Ege Atacan Doğan and Peter F. Patel-Schneider},
  journal={arXiv preprint arXiv:2512.12260v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12260v1}
}
```

## [Reliable Policy Iteration: Performance Robustness Across Architecture and Environment Perturbations](http://arxiv.org/abs/2512.12088v1)

S. R. Eshwar, Aniruddha Mukherjee, Kintan Saha, Krishna Agarwal, Gugan Thoppe, Aditya Gopalan, Gal Dalal

[PDF](https://arxiv.org/pdf/2512.12088v1)
[Abstract](http://arxiv.org/abs/2512.12088v1)

### 中文摘要

在最近的研究中，我们提出了可靠策略迭代（RPI），该方法在函数逼近环境下恢复了策略迭代中价值估计单调性的性质。本文评估了RPI在两个经典控制任务——CartPole和倒立摆——中对神经网络和环境参数变化的鲁棒性。相比于DQN、Double DQN、DDPG、TD3和PPO，RPI在训练初期即可达到接近最优的性能，并能在训练过程中持续保持这一策略。鉴于深度强化学习方法常常面临样本效率低、训练不稳定以及超参数敏感等问题，我们的结果突显了RPI作为一种更可靠替代方案的潜力。

BibTeX

```
@article{2512.12088v1,
  title={Reliable Policy Iteration: Performance Robustness Across Architecture and Environment Perturbations},
  author={S. R. Eshwar and Aniruddha Mukherjee and Kintan Saha and Krishna Agarwal and Gugan Thoppe and Aditya Gopalan and Gal Dalal},
  journal={arXiv preprint arXiv:2512.12088v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12088v1}
}
```

## [Synergizing Code Coverage and Gameplay Intent: Coverage-Aware Game Playtesting with LLM-Guided Reinforcement Learning](http://arxiv.org/abs/2512.12706v1)

Enhong Mu, Minami Yoda, Yan Zhang, Mingyue Zhang, Yutaka Matsuno, Jialong Li

[PDF](https://arxiv.org/pdf/2512.12706v1)
[Abstract](http://arxiv.org/abs/2512.12706v1)

### 中文摘要

由于“游戏即服务”模式的广泛采用，频繁的内容更新成为必然，这给质量保证带来了巨大压力。为此，自动化游戏测试被视为应对这一高频发布节奏的有效解决方案。然而，现有的自动化测试方法通常存在两种极端：以代码为中心的方法侧重于结构覆盖，却缺乏对游戏玩法上下文的理解；而以玩家为中心的代理则验证高层次的操作意图，但往往无法覆盖底层代码的具体变更。为弥合这一差距，本文提出了SMART（Structural Mapping for Augmented Reinforcement Testing）框架，这是一种结合结构验证与功能验证的创新测试方法，用于游戏更新验证。SMART利用大型语言模型（LLMs）识别抽象语法树（AST）差异并提取功能意图，构建上下文感知的混合奖励机制。该机制引导增强学习代理逐步实现游戏目标，同时自适应地探索被修改的代码分支。我们在“Trouballs”和“我的世界”两个环境中对SMART进行了评估。结果显示，SMART显著优于现有最先进的方法，其在被修改代码的分支覆盖率超过94%，几乎是传统强化学习方法的两倍，而任务完成率达98%，有效实现了结构覆盖的全面性与功能正确性的平衡。

BibTeX

```
@article{2512.12706v1,
  title={Synergizing Code Coverage and Gameplay Intent: Coverage-Aware Game Playtesting with LLM-Guided Reinforcement Learning},
  author={Enhong Mu and Minami Yoda and Yan Zhang and Mingyue Zhang and Yutaka Matsuno and Jialong Li},
  journal={arXiv preprint arXiv:2512.12706v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12706v1}
}
```

## [Robustness of Probabilistic Models to Low-Quality Data: A Multi-Perspective Analysis](http://arxiv.org/abs/2512.11912v1)

Liu Peng, Yaochu Jin

[PDF](https://arxiv.org/pdf/2512.11912v1)
[Abstract](http://arxiv.org/abs/2512.11912v1)

### 中文摘要

对低质量数据影响的系统性对比研究揭示了当代概率模型在鲁棒性方面存在显著的差异。研究发现，自回归语言模型在从词元预测到序列到序列任务中表现出极强的抗干扰能力（以GPT-2为例，尽管出现50%的词元损坏，其测试负对数似然（NLL）仅从2.87稍微上升至3.59）。相比之下，在相同数据损坏水平下，条件生成扩散模型的性能则出现灾难性下降（图像与标签的一致性较基线下降56.81%），而分类器则表现出中等程度的性能降低，且随着数据集规模的扩大而减弱。为了阐释这些差异，本文采用多角度分析框架，结合信息论、PAC学习理论和梯度动力学进行解析。这些分析表明，模型的鲁棒性主要受两大原则影响：一是条件信息的丰富程度，它限制了学习任务的复杂性；二是训练数据的绝对信息量，它使得正确信息所携带的信号能够优先生统计噪声，从而增强模型的抗干扰能力。

BibTeX

```
@article{2512.11912v1,
  title={Robustness of Probabilistic Models to Low-Quality Data: A Multi-Perspective Analysis},
  author={Liu Peng and Yaochu Jin},
  journal={arXiv preprint arXiv:2512.11912v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11912v1}
}
```

## [Causal Strengths and Leaky Beliefs: Interpreting LLM Reasoning via Noisy-OR Causal Bayes Nets](http://arxiv.org/abs/2512.11909v1)

Hanna Dettki

[PDF](https://arxiv.org/pdf/2512.11909v1)
[Abstract](http://arxiv.org/abs/2512.11909v1)

### 中文摘要

人类与机器的智能本质一直是一个长期存在的问题。尽管尚无统一公认的定义，但因果推理能力通常被视为智能的核心特征之一（Lake等，2017）。通过在相同任务上评估大型语言模型（LLMs）与人类的因果推理表现，有助于更全面地理解它们各自的优势与不足。本研究提出了三个问题：(Q1) 在相同的推理任务下，LLMs是否与人类的表现一致？(Q2) 他们在任务层面上的推理是否具有一致性？(Q3) 是否存在明显不同的推理特征？为此，我们对20多种LLMs在11个具有语义意义的因果任务中进行了评估，这些任务由Collider图（$C\_1 \to E \leftarrow C\_2$）形式表达，采用两种方法：一是“直接”法（One-shot），以概率判断为响应，即查询节点为“1”的概率；二是“链式思考”法（Chain of Thought, CoT），先思考后给出答案。这些判断通过泄露的噪声-OR因果贝叶斯网络（CBN）进行建模，其参数$θ=(b,m\_1,m\_2,p(C))\in[0,1]$，其中包括一个共享的先验$p(C)$。模型的优劣通过AIC（赤池信息准则）在具有对称（$m\_1 = m\_2$）的三参数和非对称（$m\_1 \neq m\_2$）四参数的因果强度变体中进行选择。

BibTeX

```
@article{2512.11909v1,
  title={Causal Strengths and Leaky Beliefs: Interpreting LLM Reasoning via Noisy-OR Causal Bayes Nets},
  author={Hanna Dettki},
  journal={arXiv preprint arXiv:2512.11909v1},
  year={2025},
  url={http://arxiv.org/abs/2512.11909v1}
}
```

## [A Semantically Enhanced Generative Foundation Model Improves Pathological Image Synthesis](http://arxiv.org/abs/2512.13164v2)

Xianchao Guan, Zhiyuan Fan, Yifeng Wang, Fuqiang Chen, Yanjiang Zhou, Zengyang Che, Hongxue Meng, Xin Li, Yaowei Wang, Hongpeng Wang, Min Zhang, Heng Tao Shen, Zheng Zhang, Yongbing Zhang

[PDF](https://arxiv.org/pdf/2512.13164v2)
[Abstract](http://arxiv.org/abs/2512.13164v2)

### 中文摘要

在病理学中，临床级人工智能的发展受到多样化高质量标注数据稀缺的限制。生成模型提供了一种潜在的解决方案，但其在语义稳定性和形态幻觉方面存在问题，影响诊断的可靠性。为应对这一挑战，我们提出了一种用于组织切片合成的相关性调节对齐框架（CRAFTS），这是首个针对病理专业文本到图像合成的生成基础模型。该模型采用双阶段训练策略，在约280万对图像-标签配对数据上进行训练，并引入了一种新颖的对齐机制，抑制语义漂移，确保生物学的准确性。CRAFTS能够生成涵盖30种癌症类型的多样化病理图像，其质量通过客观指标和病理学家的评估得到了严格验证。此外，利用CRAFTS增强的数据集显著提升了分类、跨模态检索、自监督学习和视觉问答等多项临床任务的性能。同时，结合ControlNet，CRAFTS实现了对组织结构的精确控制，例如通过核分割掩模和荧光图像输入调节组织架构。通过克服数据稀缺和隐私保护的关键障碍，CRAFTS为多样化、标注完备的组织学数据提供了无限源泉，有效促进了针对罕见与复杂癌症表型的稳健诊断工具开发。

BibTeX

```
@article{2512.13164v2,
  title={A Semantically Enhanced Generative Foundation Model Improves Pathological Image Synthesis},
  author={Xianchao Guan and Zhiyuan Fan and Yifeng Wang and Fuqiang Chen and Yanjiang Zhou and Zengyang Che and Hongxue Meng and Xin Li and Yaowei Wang and Hongpeng Wang and Min Zhang and Heng Tao Shen and Zheng Zhang and Yongbing Zhang},
  journal={arXiv preprint arXiv:2512.13164v2},
  year={2025},
  url={http://arxiv.org/abs/2512.13164v2}
}
```



Generated by ArXiv AI Agent • Powered by DeepSeek & Jina AI