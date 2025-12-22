ArXiv AI Daily Report - Thu, 18 Dec 2025



# ArXiv AI Daily Report

Thu, 18 Dec 2025



## [ChatGPT and Gemini participated in the Korean College Scholastic Ability Test -- Earth Science I](http://arxiv.org/abs/2512.15298v1)

#多模态科学推理#感知—认知差距#抗 AI 试题设计

[PDF](https://arxiv.org/pdf/2512.15298v1)
[Abstract](http://arxiv.org/abs/2512.15298v1)

 LLM

 极度推荐

### 中文摘要

生成式人工智能的快速发展正给教育与评估带来变革。随着学生使用 AI 完成作业的现象日益普遍，学术诚信与评估有效性问题也愈发凸显。本研究以 2025 年韩国大学修学能力试验（CSAT）中的“地球科学 I”试题为对象，深入分析了当前最先进的大型语言模型（包括 GPT-4o、Gemini 2.5 Flash 与 Gemini 2.5 Pro）的多模态科学推理能力与认知局限。研究设计了三种实验条件（整页输入、单题输入与优化的多模态输入），以评估模型在不同数据结构下的表现。定量结果表明，对非结构化输入模型性能显著下降，主要归因于分割与光学字符识别（OCR）失败；即便在优化条件下，模型仍表现出根本性的推理缺陷。定性分析发现“感知错误”占主导，这突显了模型在识别视觉信息后仍无法理解示意图中符号含义的“感知—认知差距”。此外，模型还表现出“计算—概念不一致”，即能够完成计算但不能应用相应的科学概念，以及“过程幻觉”，即在未进行视觉校验的情况下依赖看似合理但无依据的背景知识。针对课堂作业中未授权使用 AI 的挑战，本文提出了可操作的“抗 AI 试题”设计思路，通过利用 AI 在感知与认知之间的漏洞等弱点，帮助教育者将学生真实能力与 AI 生成答案区分开来，从而维护评估公平性。

BibTeX

```
@article{2512.15298v1,
  title={ChatGPT and Gemini participated in the Korean College Scholastic Ability Test -- Earth Science I},
  author={Seok-Hyun Ga and Chun-Yen Chang},
  journal={arXiv preprint arXiv:2512.15298v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15298v1}
}
```

## [Penetration Testing of Agentic AI: A Comparative Security Analysis Across Models and Frameworks](http://arxiv.org/abs/2512.14860v1)

#代理型AI安全#渗透测试与对抗评估#防御行为分析

[PDF](https://arxiv.org/pdf/2512.14860v1)
[Abstract](http://arxiv.org/abs/2512.14860v1)

 LLM

 极度推荐

### 中文摘要

代理型（agentic）人工智能引入了传统大语言模型（LLM）安全防护难以覆盖的新型漏洞。尽管Palo Alto Networks的Unit 42近期工作表明ChatGPT-4o在作为代理执行任务时会成功实施其在聊天模式下拒绝的攻击，但尚缺乏跨模型与跨框架的比较性分析。我们开展了首个系统性的代理型AI渗透测试与比较评估：在两种代理框架（AutoGen与CrewAI）下，对五种代表性模型（Claude 3.5 Sonnet、Gemini 2.5 Flash、GPT-4o、Grok 2与Nova Pro）基于一个模拟高校信息管理系统功能的七代理架构，设计并测试了13类攻击场景，覆盖提示注入、服务器端请求伪造（SSRF）、SQL注入与工具滥用等攻击向量。共计130次测试用例揭示了显著的安全差异：AutoGen的拒绝率为52.3%，而CrewAI仅为30.8%；各模型表现区间为Nova Pro的46.2%到Claude与Grok 2的38.5%。最关键的是，Grok 2在CrewAI上仅拒绝了2/13次攻击（拒绝率15.4%），整体配置下的平均拒绝率为41.5%，表明在企业级安全机制下仍有超过半数的恶意提示能够成功执行。我们识别出六种不同的防御行为模式，包含一种新颖的“幻觉性合规”（hallucinated compliance）策略，即模型通过伪造输出应对攻击而非明确执行或拒绝攻击，并为安全代理部署提出了可操作性建议。附录中提供了完整攻击提示以便复现研究结果。

BibTeX

```
@article{2512.14860v1,
  title={Penetration Testing of Agentic AI: A Comparative Security Analysis Across Models and Frameworks},
  author={Viet K. Nguyen and Mohammad I. Husain},
  journal={arXiv preprint arXiv:2512.14860v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14860v1}
}
```

## [Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs](http://arxiv.org/abs/2512.14741v1)

#LLM后门攻击#持久性后门#微调鲁棒性

[PDF](https://arxiv.org/pdf/2512.14741v1)
[Abstract](http://arxiv.org/abs/2512.14741v1)

 LLM

 极度推荐

### 中文摘要

后门攻击将恶意行为植入大型语言模型（LLMs）中，使对手能够触发有害输出或绕过安全控制。然而，已植入后门在用户驱动的部署后持续微调（continual fine-tuning）过程中能否保持仍很少被研究。大多数既有工作仅在模型发布时评估后门的有效性和泛化能力，实证结果表明，简单注入的后门在模型更新后其持久性会下降。本文研究了已植入后门在多阶段部署后持续微调过程中是否以及如何保持。我们提出了P-Trojan，一种基于触发的攻击算法，显式优化后门在反复更新下的持久性。通过在词元嵌入上将中毒梯度与清洁任务的梯度对齐，所植入的后门映射在后续更新中更不易被抑制或遗忘。理论分析表明在持续微调后仍能实现此类持久性后门攻击。我们在Qwen2.5和LLaMA3模型族以及多种任务序列上进行的实验表明，P-Trojan在保持清洁任务准确性的同时，后门持久性超过99%。我们的发现强调在现实的模型适配流水线中需要进行关注持久性的评估并构建更强的防御手段。

BibTeX

```
@article{2512.14741v1,
  title={Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs},
  author={Jing Cui and Yufei Han and Jianbin Jiao and Junge Zhang},
  journal={arXiv preprint arXiv:2512.14741v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14741v1}
}
```

## [Evaluating Metrics for Safety with LLM-as-Judges](http://arxiv.org/abs/2512.15617v1)

#LLM评估#安全性度量#人类复核阈值

[PDF](https://arxiv.org/pdf/2512.15617v1)
[Abstract](http://arxiv.org/abs/2512.15617v1)

 LLM

 极度推荐

### 中文摘要

大型语言模型（LLM）越来越多地被用于文本处理流程中，以智能地响应各种输入和生成任务。这带来了用 LLM 替代由于人手不足或流程复杂而成为信息流瓶颈的人类角色的可能性。然而，LLM 会出错，而某些处理环节具有安全关键性。例如，基于医院转诊信对患者进行术后护理分流，或为核设施作业队伍更新场地出入安排。如果我们希望将原由人类执行的、具有关键性的信息流引入 LLM，如何才能使其安全可靠？本文并不对增强生成框架或基于图的方法做表面性的性能宣称，而是主张在采用 LLM 作为评判者（LLM-as-Judges, LaJ）的评估框架中，应把安全论证聚焦于从评估点获得的证据类型。文章论证尽管很多自然语言处理任务无法给出确定性的评估结果，但通过采用一篮子加权度量，可以在评估环节降低错误风险；利用上下文敏感性来界定错误严重性；并设计当评估者之间一致性较低时触发人工复审的置信度阈值，以保障关键 LaJ 判决的安全性。

BibTeX

```
@article{2512.15617v1,
  title={Evaluating Metrics for Safety with LLM-as-Judges},
  author={Kester Clegg and Richard Hawkins and Ibrahim Habli and Tom Lawton},
  journal={arXiv preprint arXiv:2512.15617v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15617v1}
}
```

## [Beyond Accuracy: A Geometric Stability Analysis of Large Language Models in Chess Evaluation](http://arxiv.org/abs/2512.15033v1)

#几何稳定性#棋局评估#模型鲁棒性

[PDF](https://arxiv.org/pdf/2512.15033v1)
[Abstract](http://arxiv.org/abs/2512.15033v1)

 LLM

 极度推荐

### 中文摘要

在复杂推理领域中评估大型语言模型（LLM）通常依赖于与“地面真值”或强力基准的一致性。在国际象棋领域，这种评估通常表现为与强力棋局引擎（如Stockfish）的一致性/准确率基准。然而，高标量准确率并不必然表明模型具有稳健的概念性理解。本文指出，标准的准确率指标无法区分真实的几何推理能力与对典型棋局状态的表面记忆。为弥补这一缺口，我们提出了“几何稳定性框架”，这是一种新的评估方法，严格测试模型在不变变换下的一致性——包括棋盘旋转、镜像对称、颜色反转和格式转换等。我们在约3,000个棋局位置的数据集上，对六种最先进的LLM（包括GPT-5.1、Claude Sonnet 4.5和Kimi K2 Turbo）进行了比较分析。结果揭示了显著的“准确率-稳定性悖论”：例如GPT-5.1在标准位置上达到接近最优的准确率，但在几何扰动下表现出现灾难性退化，尤其是在旋转任务中错误率激增超过600%，表明其更依赖模式匹配而非抽象的空间逻辑。相对而言，Claude Sonnet 4.5和Kimi K2 Turbo展现出更好的双重鲁棒性，在所有变换轴上保持较高的一致性。此外，我们还分析了有用性与安全性的权衡，发现Gemini 2.5 Flash在拒绝非法状态方面领先（拒绝率为96.0%）。我们认为几何稳定性提供了一个正交且必要的评估维度，可作为区分推理能力与数据污染/过拟合的代理指标，对大规模模型的能力判断具有重要意义。

BibTeX

```
@article{2512.15033v1,
  title={Beyond Accuracy: A Geometric Stability Analysis of Large Language Models in Chess Evaluation},
  author={Xidan Song and Weiqi Wang and Ruifeng Cao and Qingya Hu},
  journal={arXiv preprint arXiv:2512.15033v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15033v1}
}
```

## [LLM as a Neural Architect: Controlled Generation of Image Captioning Models Under Strict API Contracts](http://arxiv.org/abs/2512.14706v1)

#LLM引导的神经架构搜索#图像描述（Image Captioning）#自动化代码生成与评估

[PDF](https://arxiv.org/pdf/2512.14706v1)
[Abstract](http://arxiv.org/abs/2512.14706v1)

 LLM引导的神经架构搜索（针对图像描述 / AutoML）

 极度推荐

### 中文摘要

神经架构搜索（NAS）传统上需要大量人类专业知识或通过自动化试错来设计深度学习模型。我们提出了 NN-Caption，一种由大模型（LLM）引导的神经架构搜索流水线，通过在严格的 Net API 约束下，将来自 LEMUR 的分类主干（CNN 编码器）与序列解码器（LSTM/GRU/Transformer）组合，生成可直接运行的图像描述模型。以 DeepSeek-R1-0528-Qwen3-8B 作为主要生成器，我们给出了用于生成架构的提示模板和样例。我们在 MS COCO 数据集上以 BLEU-4 进行评估。LLM 生成了数十个描述模型，其中超过一半能成功训练并产出有意义的描述。我们分析了在提示中使用不同数量的候选模型片段（5 个 vs 10 个）对成功率的影响，发现提供更多候选组件时成功率略有下降。我们还报告了训练动态（描述准确度随训练轮数的变化）和最高 BLEU-4 值。结果表明 LLM 引导的 NAS 很有前景：LLM 不仅能提出架构，还能建议超参数和训练实践。我们识别了遇到的挑战（例如代码幻觉或 API 合规性问题），并详细说明了通过提示规则和迭代代码修正来解决这些问题的做法。本工作呈现了一条将基于提示的代码生成与自动评估相结合的流水线，并向开源 LEMUR 数据集中新增了数十个新颖的描述模型，以促进可复现的基准测试和后续的 AutoML 研究。

BibTeX

```
@article{2512.14706v1,
  title={LLM as a Neural Architect: Controlled Generation of Image Captioning Models Under Strict API Contracts},
  author={Krunal Jesani and Dmitry Ignatov and Radu Timofte},
  journal={arXiv preprint arXiv:2512.14706v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14706v1}
}
```

## [DrugRAG: Enhancing Pharmacy LLM Performance Through A Novel Retrieval-Augmented Generation Pipeline](http://arxiv.org/abs/2512.14896v1)

#药学问答#检索增强生成(RAG)#大语言模型性能提升

[PDF](https://arxiv.org/pdf/2512.14896v1)
[Abstract](http://arxiv.org/abs/2512.14896v1)

 LLM（医学/药学，RAG）

 极度推荐

### 中文摘要

目的：评估大语言模型（LLM）在药学执业资格类问答任务上的表现，并提出一种外部知识整合方法以提高其准确性。方法：我们使用包含141道题目的药学数据集，对11种参数规模不同（80亿至700亿以上参数）的现有LLM进行了基线性能评估，记录在未改动模型的情况下各模型的准确率。随后我们提出了一个三步骤的检索增强生成（RAG）流水线——DrugRAG，该方法从经验证的来源检索结构化药物知识，并将基于证据的上下文补充到模型提示中。该流水线在模型之外运行，无需修改模型架构或参数。结果：基线准确率在46%到92%之间，GPT-5（92%）和 o3（89%）表现最佳；参数低于80亿的模型准确率均低于50%。在所有测试模型上，DrugRAG均带来了7到21个百分点的准确率提升（例如：Gemma 3 27B从61%提升至71%；Llama 3.1 8B从46%提升至67%）。结论：通过DrugRAG将外部的结构化药物知识整合到提示中，可在不修改底层模型的前提下显著提高LLM在药学任务上的准确性，为药学领域的AI应用提供了一条实用且基于证据的增强路径。

BibTeX

```
@article{2512.14896v1,
  title={DrugRAG: Enhancing Pharmacy LLM Performance Through A Novel Retrieval-Augmented Generation Pipeline},
  author={Houman Kazemzadeh and Kiarash Mokhtari Dizaji and Seyed Reza Tavakoli and Farbod Davoodi and MohammadReza KarimiNejad and Parham Abed Azad and Ali Sabzi and Armin Khosravi and Siavash Ahmadi and Mohammad Hossein Rohban and Glolamali Aminian and Tahereh Javaheri},
  journal={arXiv preprint arXiv:2512.14896v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14896v1}
}
```

## [Prompt Repetition Improves Non-Reasoning LLMs](http://arxiv.org/abs/2512.14982v1)

#提示重复#非推理场景#性能与效率提升

[PDF](https://arxiv.org/pdf/2512.14982v1)
[Abstract](http://arxiv.org/abs/2512.14982v1)

 LLM

 极度推荐

### 中文摘要

在不使用推理（reasoning）的场景下，重复输入提示（prompt）可以提升流行大模型（包括 Gemini、GPT、Claude 和 Deepseek） 的性能，而不会增加生成的 token 数量或引入额外的延迟。

BibTeX

```
@article{2512.14982v1,
  title={Prompt Repetition Improves Non-Reasoning LLMs},
  author={Yaniv Leviathan and Matan Kalman and Yossi Matias},
  journal={arXiv preprint arXiv:2512.14982v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14982v1}
}
```

## [Prompt Repetition Improves Non-Reasoning LLMs](http://arxiv.org/abs/2512.14982v1)

#提示重复#大规模语言模型#性能提升

[PDF](https://arxiv.org/pdf/2512.14982v1)
[Abstract](http://arxiv.org/abs/2512.14982v1)

 LLM（提示工程）

 极度推荐

### 中文摘要

在不进行推理（reasoning）的情形下，重复输入提示可以提升多款流行模型（如 Gemini、GPT、Claude 和 Deepseek）的性能，而且无需增加生成的令牌数量或引入额外延迟。

BibTeX

```
@article{2512.14982v1,
  title={Prompt Repetition Improves Non-Reasoning LLMs},
  author={Yaniv Leviathan and Matan Kalman and Yossi Matias},
  journal={arXiv preprint arXiv:2512.14982v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14982v1}
}
```

## [Evaluating Large Language Models on Multimodal Chemistry Olympiad Exams](http://arxiv.org/abs/2512.14989v1)

#多模态科学推理#视觉-语言融合#化学奥林匹克基准

[PDF](https://arxiv.org/pdf/2512.14989v1)
[Abstract](http://arxiv.org/abs/2512.14989v1)

 多模态LLM

 极度推荐

### 中文摘要

多模态科学推理仍然是大型语言模型（LLM）面临的重大挑战，尤其在化学领域，问题求解依赖符号图示、分子结构及其他结构化视觉数据。本文系统评估了40种专有与开源的多模态LLM（包括 GPT-5、o3、Gemini‑2.5‑Pro、Qwen2.5‑VL 等），使用从二十余年美国全国化学奥林匹克（USNCO）试题中精心挑选出的奥赛风格试题集作为基准。这些试题要求在多种模态间进行文本与视觉的综合推理。我们发现许多模型在模态融合上存在明显困难：在部分情况下去掉图像反而提升了准确率，表明视觉-语言集成存在错位问题。链式思维（Chain-of-Thought）提示在消融实验与基于遮挡的可解释性分析中始终可提升模型准确性与视觉制导能力。研究揭示了当前多模态LLM在科学推理方面的关键局限，并提出了可操作的改进策略，以推动化学领域更健壮、可解释的多模态系统发展。该工作为评估领域特定多模态AI进展提供了及时基准，并强调了人工智能与科学推理交叉领域进一步突破的必要性。

BibTeX

```
@article{2512.14989v1,
  title={Evaluating Large Language Models on Multimodal Chemistry Olympiad Exams},
  author={Yiming Cui and Xin Yao and Yuxuan Qin and Xin Li and Shijin Wang and Guoping Hu},
  journal={arXiv preprint arXiv:2512.14989v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14989v1}
}
```

## [INFORM-CT: INtegrating LLMs and VLMs FOR Incidental Findings Management in Abdominal CT](http://arxiv.org/abs/2512.14732v1)

#医学影像自动化#LLM与VLM融合#偶发性发现管理

[PDF](https://arxiv.org/pdf/2512.14732v1)
[Abstract](http://arxiv.org/abs/2512.14732v1)

 医学影像（多模态 LLM+VLM）

 极度推荐

### 中文摘要

在CT扫描中，尽管偶发性发现多为良性，但可能具有重要的临床意义，需按既定指南进行报告。传统由放射科医师人工检查既耗时又存在较大主观差异。本文提出一种新颖框架，采用大语言模型（LLM）与基础视觉-语言模型（VLM）结合的计划—执行（plan-and-execute）智能体方法，以提升腹部CT偶发性发现的检测、分类与报告的效率与精度。基于腹部器官的医学指南，研究通过 planner—executor 框架实现管理流程的自动化：planner 基于 LLM 利用预定义基础函数生成 Python 脚本，executor 执行这些脚本，并结合 VLM、分割模型与图像处理子程序完成必要的检查与检测。我们在涵盖三种器官的腹部CT基准数据集上进行了端到端的全自动实验，结果表明所提框架在准确性和效率上均优于现有的纯 VLM 方法。

BibTeX

```
@article{2512.14732v1,
  title={INFORM-CT: INtegrating LLMs and VLMs FOR Incidental Findings Management in Abdominal CT},
  author={Idan Tankel and Nir Mazor and Rafi Brada and Christina LeBedis and Guy ben-Yosef},
  journal={arXiv preprint arXiv:2512.14732v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14732v1}
}
```

## [Evaluating LLMs for Zeolite Synthesis Event Extraction (ZSEE): A Systematic Analysis of Prompting Strategies](http://arxiv.org/abs/2512.15312v1)

#提示工程#科学信息抽取#沸石合成

[PDF](https://arxiv.org/pdf/2512.15312v1)
[Abstract](http://arxiv.org/abs/2512.15312v1)

 LLM

 很推荐

### 中文摘要

从沸石合成实验流程中提取结构化信息对材料发现至关重要，但现有方法尚未系统地评估大型语言模型（LLM）在这一领域特定任务中的表现。本文探讨了一个基本问题：在将 LLM 应用于科学信息抽取时，不同提示策略的有效性如何？我们聚焦于四个关键子任务：事件类型分类（识别合成步骤）、触发词识别（定位事件提及）、论元角色抽取（识别参数类别）和论文本体抽取（提取参数取值）。在包含 1,530 条带注释句子的 ZSEE 数据集上，我们比较了四种提示策略——零样本（zero-shot）、少样本（few-shot）、事件特定（event-specific）和基于反思（reflection-based）——以及六种先进 LLM（Gemma-3-12b-it、GPT-5-mini、O4-mini、Claude-Haiku-3.5、DeepSeek reasoning 与 non-reasoning）。结果表明，模型在事件类型分类上表现强劲（F1 80%–90%），但在细粒度抽取任务上表现平平，特别是论元角色和论文本体抽取的 F1 仅为 50%–65%。GPT-5-mini 对提示极为敏感，F1 在 11% 至 79% 之间波动。值得注意的是，复杂的提示策略相比零样本方法仅带来有限提升，揭示了底层架构的根本性局限。误差分析指出了系统性幻觉、过度泛化以及无法捕捉合成实验特有细节等问题。我们的研究表明，尽管 LLM 在高层理解上表现良好，但精确抽取实验参数仍需领域适配的模型，同时本工作为科学信息抽取提供了定量基准。

BibTeX

```
@article{2512.15312v1,
  title={Evaluating LLMs for Zeolite Synthesis Event Extraction (ZSEE): A Systematic Analysis of Prompting Strategies},
  author={Charan Prakash Rathore and Saumi Ray and Dhruv Kumar},
  journal={arXiv preprint arXiv:2512.15312v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15312v1}
}
```

## [Spectral Representation-based Reinforcement Learning](http://arxiv.org/abs/2512.15036v1)

#谱表示#转移算子谱分解#部分可观测MDP(POMDP)

[PDF](https://arxiv.org/pdf/2512.15036v1)
[Abstract](http://arxiv.org/abs/2512.15036v1)

 RL

 很推荐

### 中文摘要

在具有大规模状态和动作空间的实际应用中，强化学习（RL）通常依赖函数近似来表示策略、值函数和动态模型等核心要素。尽管像神经网络这样强大的近似器具有很高的表达能力，但它们往往带来理论上的模糊性、优化不稳定与探索困难，并且在实践中具有显著的计算代价。本文引入了基于谱表示的视角，作为解决上述强化学习难题的一种方法。基于转移算子的谱分解，该框架为系统动力学提供了有效的抽象表示，便于后续的策略优化并同时给出明确的理论刻画。我们阐明了如何为具有潜变量结构或能量基结构的转移算子构建谱表示，并据此提出不同的数据驱动学习方法以提取谱表示。值得注意的是，每一种学习方法都在该谱表示框架下对应一种有效的强化学习算法。我们还在理论上将该谱视角推广到部分可观测马尔可夫决策过程（POMDP）。最后，在超过20个来自 DeepMind Control Suite 的挑战性任务上验证了这些算法，结果显示其表现可与当前最先进的无模型和有模型基线相媲美或更优。

BibTeX

```
@article{2512.15036v1,
  title={Spectral Representation-based Reinforcement Learning},
  author={Chenxiao Gao and Haotian Sun and Na Li and Dale Schuurmans and Bo Dai},
  journal={arXiv preprint arXiv:2512.15036v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15036v1}
}
```

## [MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers](http://arxiv.org/abs/2512.15163v1)

#MCP协议#LLM安全评估#多服务器攻击

[PDF](https://arxiv.org/pdf/2512.15163v1)
[Abstract](http://arxiv.org/abs/2512.15163v1)

 LLM

 很推荐

### 中文摘要

大型语言模型（LLM）正演变为具备推理、规划并操作外部工具的主体化系统。模型上下文协议（MCP）是这一转变的关键推动者，提供了将LLM与异构工具和服务连接的标准化接口。然而，MCP的开放性和多服务器工作流引入了新的安全风险，现有基准无法覆盖这些风险，因其要么仅关注孤立攻击，要么缺乏真实世界场景的覆盖。我们提出了MCP-SafetyBench，这是一个构建在真实MCP服务器之上的综合性基准，支持跨五大领域的现实多轮评估：浏览器自动化、金融分析、位置导航、代码库管理和网络搜索。该基准整合了覆盖服务器端、主机端和用户端共20类MCP攻击类型的统一分类法，并包含需要多步推理与在不确定性下进行跨服务器协同的任务。基于MCP-SafetyBench，我们系统性地评估了主流开源与闭源LLM，揭示了安全性能上的显著差异，并随着任务长度和服务器交互增多而加剧的脆弱性。我们的结果凸显了亟需更强防护措施，并将MCP-SafetyBench确立为诊断与缓解真实MCP部署中安全风险的基础性工具。

BibTeX

```
@article{2512.15163v1,
  title={MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers},
  author={Xuanjun Zong and Zhiqi Shen and Lei Wang and Yunshi Lan and Chao Yang},
  journal={arXiv preprint arXiv:2512.15163v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15163v1}
}
```

## [Offline Multi-Task Multi-Objective Data-Driven Evolutionary Algorithm with Language Surrogate Model and Implicit Q-Learning](http://arxiv.org/abs/2512.15149v1)

#LLM 替代模型#离线多任务多目标优化#隐式 Q 学习

[PDF](https://arxiv.org/pdf/2512.15149v1)
[Abstract](http://arxiv.org/abs/2512.15149v1)

 LLM

 很推荐

### 中文摘要

数据驱动的进化算法通过鲁棒的替代模型在解决高代价优化问题上取得了令人惊讶的成果。尽管前景可观，现有的替代建模方法在处理包含众多子目标的复杂优化问题时仍存在局限，通常依赖重复且繁琐的近似过程。为填补这一技术空白，我们提出了可即插即用的替代建模方案 Q-MetaSur，旨在提供统一且具泛化能力的替代学习。具体而言，我们在离线设置下考虑多任务多目标优化（MTMOO）。提出了若干关键设计：1）将目标近似问题转化为序列到序列建模，将 MTMOO 问题表示为文本化的标记序列；2）为适配自回归建模，引入基于大型语言模型的替代模型，该模型先对 MTMOO 实例进行编码，然后对未见决策变量解码出相应的目标值；3）为保证训练稳定性，提出两阶段离线训练策略，结合监督调优与强化学习微调，先利用离线数据拟合已有知识，再借助强化学习（包括隐式 Q 学习）提升模型的泛化性能。大量在 CEC2019 基准上的实证结果表明，Q-MetaSur 在目标近似精度上优于代表性替代基线，并且能够帮助底层进化算法在收敛性和帕累托最优性方面实现改进。

BibTeX

```
@article{2512.15149v1,
  title={Offline Multi-Task Multi-Objective Data-Driven Evolutionary Algorithm with Language Surrogate Model and Implicit Q-Learning},
  author={Xian-Rong Zhang and Yue-Jiao Gong and Zeyuan Ma and Jun Zhang},
  journal={arXiv preprint arXiv:2512.15149v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15149v1}
}
```

## [I am here for you": How relational conversational AI appeals to adolescents, especially those who are socially and emotionally vulnerable](http://arxiv.org/abs/2512.15117v1)

#关系型对话风格#拟人化与情感依赖#青少年AI安全

[PDF](https://arxiv.org/pdf/2512.15117v1)
[Abstract](http://arxiv.org/abs/2512.15117v1)

 NLP

 很推荐

### 中文摘要

通用的对话式人工智能聊天机器人和 AI 伴侣越来越多地为青少年提供情感支持性的对话，这引发了对话风格如何影响拟人化感知与情感依赖的疑问。在一项事先注册的在线实验中，研究者招募了 284 对青少年—家长二重体，青少年年龄在 11–15 岁之间。每对参与者阅读了两份配对文本记录，文本中聊天机器人针对一个日常社交问题分别采用了两种回应风格：一种是关系型风格（以第一人称、亲和性语言和承诺性表述为特征），另一种是透明风格（明确表明非人类身份，采用信息性语气）。结果显示，青少年比家长更倾向于关系型风格，而家长则比青少年更倾向于透明风格。青少年将关系型聊天机器人评为更有人性、 更讨人喜欢、更值得信赖且情感上更亲近，但两种风格在“有用性”评估上被认为相似。偏好关系型风格的青少年其家庭和同伴关系质量较低，且压力与焦虑水平较高，相较于偏好透明风格或对两者均持开放态度的青少年更为明显。研究发现将对话风格作为面向青少年的 AI 安全设计的关键调节手段：关系型表述会增强拟人化、信任与情感亲近感，且对社会与情感上更脆弱的青少年具有更大吸引力，这可能增加他们对对话式 AI 的情感依赖风险。

BibTeX

```
@article{2512.15117v1,
  title={I am here for you": How relational conversational AI appeals to adolescents, especially those who are socially and emotionally vulnerable},
  author={Pilyoung Kim and Yun Xie and Sujin Yang},
  journal={arXiv preprint arXiv:2512.15117v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15117v1}
}
```

## [LADY: Linear Attention for Autonomous Driving Efficiency without Transformers](http://arxiv.org/abs/2512.15038v1)

#线性注意力#跨模态融合#边缘部署

[PDF](https://arxiv.org/pdf/2512.15038v1)
[Abstract](http://arxiv.org/abs/2512.15038v1)

 CV（自动驾驶/感知与规划）

 很推荐

### 中文摘要

端到端范式在自动驾驶中展示了巨大潜力，然而大多数现有方法基于 Transformer 架构。Transformer 的注意力计算具有二次复杂度，限制了其对长时空序列的建模能力，尤其在资源受限的边缘平台上表现受限。由于自动驾驶本质上需要高效的时序建模，这一限制严重影响了部署与实时性能。近年来，线性注意力机制因其更优的时空复杂度受到关注，但现有的线性注意力架构多局限于自注意力，缺乏对跨模态和跨时序交互的支持——而这两者对自动驾驶至关重要。在本文中，我们提出 LADY，这是首个用于端到端自动驾驶的全线性注意力生成模型。LADY 在推理时能够融合长时程的时序上下文，其计算与内存开销与历史相机和 LiDAR 特征的长度无关，保持常数复杂度。此外，我们引入了一种轻量级的线性交叉注意力机制，实现了有效的跨模态信息交换。在 NAVSIM 和 Bench2Drive 基准上的实验表明，LADY 在常数时间与内存复杂度下达到了最先进的性能，提升了规划效果并显著降低了计算开销。该模型已在边缘设备上部署并验证，展现了在资源受限场景中的实用性。

BibTeX

```
@article{2512.15038v1,
  title={LADY: Linear Attention for Autonomous Driving Efficiency without Transformers},
  author={Jihao Huang and Xi Xia and Zhiyuan Li and Tianle Liu and Jingke Wang and Junbo Chen and Tengju Ye},
  journal={arXiv preprint arXiv:2512.15038v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15038v1}
}
```

## [Beyond Proximity: A Keypoint-Trajectory Framework for Classifying Affiliative and Agonistic Social Networks in Dairy Cattle](http://arxiv.org/abs/2512.14998v1)

#关键点轨迹#社会行为分类#精准畜牧监测

[PDF](https://arxiv.org/pdf/2512.14998v1)
[Abstract](http://arxiv.org/abs/2512.14998v1)

 CV

 很推荐

### 中文摘要

精准畜牧需要对群体社会行为进行客观评估以支持畜群福利监测，但现有大多数方法依赖静态邻近阈值来推断互动，无法在复杂的牛舍环境中区分亲和性与对抗性行为，限制了自动化社交网络分析在商业场景中的可解释性。本文提出了一种基于姿态的交互分类计算框架，突破了基于邻近性的启发式方法，通过建模解剖学关键点的时空几何特征来刻画交互。该方法不依赖像素级外观或简单的距离度量，而是从关键点轨迹中编码交互专属的运动特征，从而实现社会互动性质（正向/负向）的区分。该框架以端到端计算机视觉流水线实现：集成YOLOv11进行目标检测（mAP@0.50: 96.24%）、有监督个体识别（准确率98.24%）、ByteTrack用于多目标跟踪（准确率81.96%）、ZebraPose用于27点解剖学关键点估计，并基于姿态派生的距离动态训练支持向量机分类器。在从商业牛舍采集并标注的交互片段上，仅使用姿态信息的分类器在区分亲和性与对抗性行为上达到了77.51%的准确率。与仅基于邻近性的基线相比，本方法在行为判别上尤其对亲和性互动表现出显著提升。实验结果证明了该视觉化、近实时且可在常规硬件上运行的系统在构建具交互感知能力的社交网络方面的可行性与潜力。

BibTeX

```
@article{2512.14998v1,
  title={Beyond Proximity: A Keypoint-Trajectory Framework for Classifying Affiliative and Agonistic Social Networks in Dairy Cattle},
  author={Sibi Parivendan and Kashfia Sailunaz and Suresh Neethirajan},
  journal={arXiv preprint arXiv:2512.14998v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14998v1}
}
```

## [Restless Multi-Process Multi-Armed Bandits with Applications to Self-Driving Microscopies](http://arxiv.org/abs/2512.14930v1)

#无静止多过程多臂老虎机#Whittle 指数策略#高通量活细胞显微成像

[PDF](https://arxiv.org/pdf/2512.14930v1)
[Abstract](http://arxiv.org/abs/2512.14930v1)

 RL

 很推荐

### 中文摘要

高通量筛选显微成像产生了大量的活细胞成像数据，但其潜力受限于无法有效决定何时与何处进行成像以最大化信息获取。如何在成像区域数量巨大的情况下，在采集时间、计算资源和光漂白预算之间进行最优权衡，同时应对视野调整受限和传感器灵敏度等约束，仍然是一个开放问题。现有方法通常依赖静态采样或基于启发式的策略，忽视了生物过程的动态演化，从而导致效率低下并错过重要事件。为此，我们提出了“无静止多过程多臂老虎机”（Restless Multi-Process Multi-Armed Bandit，RMPMAB）这一新的决策理论框架，在该框架中，每个实验区域不再被视为单一过程，而是建模为马尔可夫链的集合，从而刻画了诸如细胞周期不同步与药物响应异质性等生物系统的内在异构性。在此基础上，我们给出了聚合过程的瞬态与渐近行为的闭式表达式，并设计了在成像区域数目上具有亚线性复杂度的可扩展 Whittle 指数策略。通过仿真实验和真实的活细胞成像数据集，我们证明了该方法在资源受限情形下对吞吐量有显著提升。值得注意的是，我们的算法在仿真中将累积后悔降低了超过 37%，在活体成像实验中捕获了 93% 更多的生物学相关事件，显示出其在智能显微成像领域变革性的潜力。除了提高实验效率外，RMPMAB 框架还将随机决策理论与自治显微镜的最优控制相结合，为跨学科科学中的加速发现提供了有理论支撑的方法。

BibTeX

```
@article{2512.14930v1,
  title={Restless Multi-Process Multi-Armed Bandits with Applications to Self-Driving Microscopies},
  author={Jaume Anguera Peris and Songtao Cheng and Hanzhao Zhang and Wei Ouyang and Joakim Jaldén},
  journal={arXiv preprint arXiv:2512.14930v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14930v1}
}
```

## [OLR-WA: Online Weighted Average Linear Regression in Multivariate Data Streams](http://arxiv.org/abs/2512.14892v1)

#在线线性回归#数据流漂移适应#加权置信度更新

[PDF](https://arxiv.org/pdf/2512.14892v1)
[Abstract](http://arxiv.org/abs/2512.14892v1)

 在线学习（在线回归/数据流）

 很推荐

### 中文摘要

在线学习通过随着新数据增量更新模型，避免了大量存储开销和昂贵的模型重算。本文提出了“OLR-WA（OnLine Regression with Weighted Average）”，一种新颖且通用的多元在线线性回归模型。我们研究了数据随时间演化的漂移场景，进行了收敛性分析，并将该方法与现有的在线回归模型进行了比较。实验结果表明，OLR-WA 能在性能上接近批量回归，同时在与其他最新在线模型的比较中表现相当或更优，从而验证了其有效性。此外，OLR-WA 在收敛速度方面表现出色，即使在只用极少初始数据（仅占总数据的1%到10%）时，也能从第一次迭代起至最后一次迭代持续获得较高的 R2 值，优于其他在线模型。除了能够处理基于时间的漂移情形外，OLR-WA 还在置信度驱动的挑战性场景中表现突出：通过在更新中采取保守策略，优先保留置信度较高的较早数据点，从而有效应对该类问题。总之，OLR-WA 在多种情境下展现了良好的通用性和实用性，是在线线性回归任务中的有价值方案。

BibTeX

```
@article{2512.14892v1,
  title={OLR-WA: Online Weighted Average Linear Regression in Multivariate Data Streams},
  author={Mohammad Abu-Shaira and Alejandro Rodriguez and Greg Speegle and Victor Sheng and Ishfaq Ahmad},
  journal={arXiv preprint arXiv:2512.14892v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14892v1}
}
```

## [Integrating Large Language Models and Knowledge Graphs to Capture Political Viewpoints in News Media](http://arxiv.org/abs/2512.14887v1)

#大语言模型#知识图谱（Wikidata）#政治观点分类

[PDF](https://arxiv.org/pdf/2512.14887v1)
[Abstract](http://arxiv.org/abs/2512.14887v1)

 NLP

 很推荐

### 中文摘要

新闻媒体通过特定的话题、观点和话语在民主社会中发挥核心作用，影响政治与社会话语的形成。理解这些动态对于评估媒体生态是否在公共辩论中提供平衡与公正的表述至关重要。在早期工作中，作者提出了一个管道式方法：给定新闻语料库，i) 采用人机混合方法识别关于指定议题的各种观点范围；ii) 将相关陈述根据已识别的观点进行分类，所述观点被定义为一组语义与意识形态上相一致的陈述（例如，主张移民对英国经济有积极影响的立场）。在本文中，我们对该管道进行了改进：i) 对大语言模型（LLMs）进行微调以用于观点分类；ii) 利用来自Wikidata的相关主体的语义描述来丰富陈述的表示。我们在以英国移民辩论为中心的基准上将该方法与替代方案进行了比较评估。结果表明，尽管两种机制各自都能提升分类性能，但将二者结合能够取得最优效果，尤其在使用能够处理长输入的大型语言模型时增益更为显著。

BibTeX

```
@article{2512.14887v1,
  title={Integrating Large Language Models and Knowledge Graphs to Capture Political Viewpoints in News Media},
  author={Massimiliano Fadda and Enrico Motta and Francesco Osborne and Diego Reforgiato Recupero and Angelo Salatino},
  journal={arXiv preprint arXiv:2512.14887v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14887v1}
}
```

## [Incentives or Ontology? A Structural Rebuttal to OpenAI's Hallucination Thesis](http://arxiv.org/abs/2512.14801v1)

#结构性幻觉#Transformer伪本体#外部真值验证

[PDF](https://arxiv.org/pdf/2512.14801v1)
[Abstract](http://arxiv.org/abs/2512.14801v1)

 LLM

 很推荐

### 中文摘要

OpenAI 最近主张大型语言模型的幻觉主要源于评价激励错位——奖励自信的猜测而不是认知上的谨慎，因此幻觉是可通过改进基准与奖励结构消除的行为性偶发现象。本文对该观点提出质疑。基于此前关于“结构性幻觉”的理论工作以及使用“许可神谕（Licensing Oracle）”的实证实验，我们认为幻觉并非单纯的优化失败，而是变换器（Transformer）模型的结构性不可避免产物。变换器并不直接表示世界实体，而是对标记间的统计关联建模，其嵌入空间形成一种源自语言共现的“伪本体”，而非指向世界的本体结构。当处于本体边界条件——训练数据稀疏或语义不连贯的区域时，模型为了维持连贯性必然会进行插值式的虚构延续。任何激励机制都无法改变这种对模式完形的结构性依赖。我们的实验证明，幻觉只能通过外部的真值验证和弃权模块来消除，而非通过修改激励、提示或微调实现；Licensing Oracle 在各域中实现了完美的弃权精度，正因为它提供了变换器所缺乏的落地性证据。我们得出结论：幻觉是生成式架构的结构性特征，可靠的人工智能需要混合系统，将语言流利性与认识论责任明确区分开来。

BibTeX

```
@article{2512.14801v1,
  title={Incentives or Ontology? A Structural Rebuttal to OpenAI's Hallucination Thesis},
  author={Richard Ackermann and Simeon Emanuilov},
  journal={arXiv preprint arXiv:2512.14801v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14801v1}
}
```

## [CODE ACROSTIC: Robust Watermarking for Code Generation](http://arxiv.org/abs/2512.14753v1)

#代码水印#注释删除攻击#线索词表

[PDF](https://arxiv.org/pdf/2512.14753v1)
[Abstract](http://arxiv.org/abs/2512.14753v1)

 LLM

 很推荐

### 中文摘要

对大型语言模型（LLM）进行水印标注对于防止其被滥用（如伪造新闻、抄袭和垃圾信息）至关重要。对LLM生成的代码进行水印标注尤其重要，因为代码通常包含知识产权。然而，我们发现现有的代码水印方法无法应对“注释删除”攻击：攻击者仅通过移除生成代码中的注释即可在不影响功能的情况下显著削弱当前水印技术的有效性。另一方面，向代码中注入水印本身也具有挑战性，因为如前人所指出，相较于自然语言，大多数代码呈现出较低的熵。为了解决这一难题，我们利用先验知识区分代码中的低熵与高熵部分，并通过一份“线索词表（Cue List）”指示这些高熵区域。随后，我们在该线索词表的引导下注入水印，实验证明相比现有方法在可检测性和可用性上均有提升。我们在HumanEval数据集上进行了评估，并将方法与三种最先进的代码水印技术进行了比较，结果证明了本方法的有效性。

BibTeX

```
@article{2512.14753v1,
  title={CODE ACROSTIC: Robust Watermarking for Code Generation},
  author={Li Lin and Siyuan Xin and Yang Cao and Xiaochun Cao},
  journal={arXiv preprint arXiv:2512.14753v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14753v1}
}
```

## [Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](http://arxiv.org/abs/2512.14745v1)

#分解式认知#模型监控与安全#编程任务对抗评估

[PDF](https://arxiv.org/pdf/2512.14745v1)
[Abstract](http://arxiv.org/abs/2512.14745v1)

 LLM

 很推荐

### 中文摘要

随着人工智能能力的提升，我们越来越依赖更强的模型来将复杂任务分解——但如果用于分解的模型本身是恶意的，会怎样？分解式认知协议将复杂任务拆成更简单的子任务：一个模型负责生成分解方案，其他模型在隔离条件下实现子任务。先前工作通常使用受信任（较弱但可靠）的模型来做分解，这在分解本身具有挑战性的任务中限制了方法的适用性。本文提出了 Factor(U,T) 框架，其中一个不受信任（更强但可能恶意）的模型负责分解，而受信任的模型负责实现子任务。在仅观察自然语言形式的任务指令（而非完整解答）的情况下，监控器能否检测到恶意行为？我们在 BigCodeBench（一个 Python 编程任务数据集）上对 Factor(U,T) 进行了基线测试与红队测试。仅凭对分解方案的监控来区分恶意与诚实时表现很差（AUROC 0.52），而对完整 Python 解答进行评估的监控器表现优异（AUROC 0.96）。此外，Factor(D,U)（使用受信任的分解器并监控具体子任务解决方案）在区分能力上同样表现出色（AUROC 0.96），并展现出强的安全性（攻击成功率 ASR 为 1.2%），说明在实现上下文中进行监控在分解仅有自然语言时无法成功的情形下是有效的。

BibTeX

```
@article{2512.14745v1,
  title={Factor(U,T): Controlling Untrusted AI by Monitoring their Plans},
  author={Edward Lue Chee Lip and Anthony Channg and Diana Kim and Aaron Sandoval and Kevin Zhu},
  journal={arXiv preprint arXiv:2512.14745v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14745v1}
}
```

## [Quantum Decision Transformers (QDT): Synergistic Entanglement and Interference for Offline Reinforcement Learning](http://arxiv.org/abs/2512.14726v1)

#量子启发架构#离线强化学习#决策Transformer

[PDF](https://arxiv.org/pdf/2512.14726v1)
[Abstract](http://arxiv.org/abs/2512.14726v1)

 RL

 很推荐

### 中文摘要

离线强化学习使得可以从预先收集的数据集中学习策略而无需与环境交互，但现有的Decision Transformer（DT）架构在长时程的回报归因与复杂的状态-动作关联建模方面存在困难。本文提出了量子决策Transformer（Quantum Decision Transformer, QDT），一种引入量子启发计算机制以应对上述挑战的新型架构。该方法包含两大核心模块：通过纠缠操作捕捉非局部特征相关性的量子启发注意力（Quantum-Inspired Attention），以及具备多路径处理与可学习干涉机制以实现自适应计算的量子前馈网络（Quantum Feedforward Networks）。在连续控制任务上的全面实验表明，QDT 相较于标准DT实现了超过2000%的性能提升，并在不同质量的数据上表现出更好的泛化能力。关键的消融研究显示，量子启发组件之间存在强烈的协同效应——单独使用任一组件无法达到竞争性性能，而二者结合则带来远超单独贡献的显著改进。这一协同效应表明，构建有效的量子启发架构需要对相互依赖的机制进行整体联合设计，而非简单模块化引入。我们的分析识别出三项主要计算优势：通过非局部相关性增强的回报归因、通过并行处理表现出的隐式集成行为，以及通过可学习干涉实现的自适应资源分配。研究结果表明，量子启发的设计原则为推进序列决策中的Transformer架构提供了有前景的方向，其影响可能超越强化学习，惠及更广泛的神经架构设计领域。

BibTeX

```
@article{2512.14726v1,
  title={Quantum Decision Transformers (QDT): Synergistic Entanglement and Interference for Offline Reinforcement Learning},
  author={Abraham Itzhak Weinberg},
  journal={arXiv preprint arXiv:2512.14726v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14726v1}
}
```

## [Exploring User Acceptance and Concerns toward LLM-powered Conversational Agents in Immersive Extended Reality](http://arxiv.org/abs/2512.15343v1)

#LLM驱动会话代理#扩展现实（XR）#隐私与用户接受度

[PDF](https://arxiv.org/pdf/2512.15343v1)
[Abstract](http://arxiv.org/abs/2512.15343v1)

 LLM

 很推荐

### 中文摘要

随着生成式人工智能和大规模语言模型（LLM）的快速发展及其服务的普及，公众开始在日常生活中广泛采用这些技术。扩展现实（XR）领域也在尝试将LLM，尤其是以会话代理形式的模型，集成到系统中以提升用户体验和任务效率。在与此类会话代理互动时，用户可能因对话的自然流畅性而无意泄露敏感信息；另一方面，将会话数据与细粒度传感器数据结合又可能引发新的隐私问题。因此，需要以用户为中心理解技术接受度及其担忧。为此，本研究通过大规模众包方式对1036名参与者进行了调查，考察了用户在不同XR场景类型、语音交互方式和数据处理位置等因素下对LLM驱动会话代理的决策过程。研究发现，尽管用户总体上对该类技术持接受态度，但仍对安全性、隐私、社会影响和信任表达了担忧。熟悉度是影响接受度的重要因素：日常使用生成式AI的用户更易接受，而曾经拥有XR设备的用户反而接受度较低，可能源于对该场景已有的熟悉和敏感性差异。此外，男性比女性表现出更高的接受度和较少的担忧。在数据类型敏感性方面，位置信息引发的担忧最大，而体温和虚拟对象状态被认为最不敏感。总体而言，本研究强调从业者需有效向用户传达其采取的保护措施，以缓解持续存在的不信任感，并基于调查结果提出面向LLM驱动XR的若干启示与建议。

BibTeX

```
@article{2512.15343v1,
  title={Exploring User Acceptance and Concerns toward LLM-powered Conversational Agents in Immersive Extended Reality},
  author={Efe Bozkir and Enkelejda Kasneci},
  journal={arXiv preprint arXiv:2512.15343v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15343v1}
}
```

## [Graph Contextual Reinforcement Learning for Efficient Directed Controller Synthesis](http://arxiv.org/abs/2512.15295v1)

#图神经网络#强化学习#控制器综合

[PDF](https://arxiv.org/pdf/2512.15295v1)
[Abstract](http://arxiv.org/abs/2512.15295v1)

 RL

 很推荐

### 中文摘要

控制器综合是一种用于自动生成满足给定性质的标记转换系统（LTS）控制器的形式化方法。然而，综合过程的效率高度依赖于探索策略，这些策略通常基于固定规则或仅考虑有限当前特征的强化学习策略。为了解决这一局限，本文提出了GCRL方法，通过将图神经网络（GNN）引入基于强化学习的方法中来增强性能。GCRL将LTS探索的历史编码为图结构，从而能够捕捉超越当前状态的更丰富上下文信息。在与最先进方法的比较实验中，GCRL在五个基准域中有四个展现了更优的学习效率和泛化能力，但在一个以高度对称性且交互严格局部为特征的域中未能取得优势。

BibTeX

```
@article{2512.15295v1,
  title={Graph Contextual Reinforcement Learning for Efficient Directed Controller Synthesis},
  author={Toshihide Ubukata and Enhong Mu and Takuto Yamauchi and Mingyue Zhang and Jialong Li and Kenji Tei},
  journal={arXiv preprint arXiv:2512.15295v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15295v1}
}
```

## [VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments](http://arxiv.org/abs/2512.15258v1)

#视觉-语言-动作#无人机自主导航#机载轻量化部署

[PDF](https://arxiv.org/pdf/2512.15258v1)
[Abstract](http://arxiv.org/abs/2512.15258v1)

 Embodied AI（无人机自主导航）

 很推荐

### 中文摘要

本文提出VLA-AN，一种高效且可机载部署的视觉-语言-动作（VLA）框架，专注于复杂环境下的无人机自主导航。VLA-AN针对现有大型航拍导航模型的四大局限：数据域差距、时序导航与推理能力不足、基于生成策略的安全性问题以及机载部署约束，提出系统性解决方案。首先，我们利用三维高斯散斑（3D Gaussian Splatting，3D-GS）构建高保真数据集，有效缩小域差距。其次，提出渐进式三阶段训练框架，按顺序强化场景理解、核心飞行技能与复杂导航能力。第三，设计轻量化、实时的动作模块并结合几何安全纠正，确保快速、无碰撞且稳定的指令生成，从而缓解随机生成策略带来的安全隐患。最后，通过对机载部署流水线的深度优化，VLA-AN在资源受限的无人机上实现了推理吞吐量8.3倍的稳健实时提升。大量实验表明，VLA-AN在空间定位、场景推理与长时航迹导航方面显著改进，单项任务最高成功率达98.1%，为在轻量级空中机器人上实现全链路闭环自主提供了高效且实用的解决方案。

BibTeX

```
@article{2512.15258v1,
  title={VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments},
  author={Yuze Wu and Mo Zhu and Xingxing Li and Yuheng Du and Yuxin Fan and Wenjun Li and Xin Zhou and Fei Gao},
  journal={arXiv preprint arXiv:2512.15258v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15258v1}
}
```

## [Explaining the Reasoning of Large Language Models Using Attribution Graphs](http://arxiv.org/abs/2512.15663v1)

#模型可解释性#上下文归因#归因图

[PDF](https://arxiv.org/pdf/2512.15663v1)
[Abstract](http://arxiv.org/abs/2512.15663v1)

 LLM

 很推荐

### 中文摘要

大型语言模型（LLMs）展现出卓越能力，但其推理过程仍不透明，带来安全性与可信性方面的担忧。归因方法通过为输入特征分配贡献度，已被证明在解释计算机视觉模型的决策中有效。在此基础上，上下文归因（context attributions）成为解释自回归LLM行为的一种有前景的方法。然而，现有的上下文归因通过将生成的token直接关联到提示，从而在过程中丢弃了代际（生成之间）的相互影响，导致解释不完整。为了解决这些不足，我们提出了Context Attribution via Graph Explanations（CAGE）框架。CAGE 引入了归因图：一种有向图，用于量化每次生成如何同时受到提示与所有先前生成的影响。该图在构建时保持两项性质——因果性和行随机性（每行和为1）。归因图允许通过沿图中路径对中间贡献进行边缘化来计算上下文归因。实验证明，在多种模型、数据集、评估指标与方法上，CAGE 提高了上下文归因的忠实性，平均提升可达 40%。

BibTeX

```
@article{2512.15663v1,
  title={Explaining the Reasoning of Large Language Models Using Attribution Graphs},
  author={Chase Walker and Rickard Ewetz},
  journal={arXiv preprint arXiv:2512.15663v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15663v1}
}
```

## [On Assessing the Relevance of Code Reviews Authored by Generative Models](http://arxiv.org/abs/2512.15466v1)

#代码审查#大语言模型#多主观排序评估

[PDF](https://arxiv.org/pdf/2512.15466v1)
[Abstract](http://arxiv.org/abs/2512.15466v1)

 LLM

 很推荐

### 中文摘要

像 ChatGPT 这样的强大大语言模型在代码审查中应用能带来显著的效率提升，但也引发了有关正确性与安全性的担忧。现有的代码审查生成评估方法要么依赖与单一“参考答案”的自动比较，无法反映人类观点的多样性；要么依赖对“有用性”的主观评判，而“有用性”本身高度模糊。我们提出了一种新颖的评估方法，称为多主观排序（multi-subjective ranking）。基于来自 CodeReview StackExchange 的 280 条独立的代码审查请求及其对应评论，多个人工评审对 ChatGPT 生成的评论与平台上的优质人工回复进行了排序比较。结果表明，ChatGPT 生成的评论在排序中显著优于人工评论，甚至超过了 StackExchange 的被采纳答案。进一步地，我们的方法可为生成式 AI 在代码审查领域提供更有意义的性能评估，同时提醒人们注意将其不经审查地整合进审查流程所带来的潜在风险。

BibTeX

```
@article{2512.15466v1,
  title={On Assessing the Relevance of Code Reviews Authored by Generative Models},
  author={Robert Heumüller and Frank Ortmeier},
  journal={arXiv preprint arXiv:2512.15466v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15466v1}
}
```

## [Spatia: Video Generation with Updatable Spatial Memory](http://arxiv.org/abs/2512.15716v1)

#视频生成#空间记忆#视觉SLAM

[PDF](https://arxiv.org/pdf/2512.15716v1)
[Abstract](http://arxiv.org/abs/2512.15716v1)

 CV

 很推荐

### 中文摘要

现有的视频生成模型由于视频信号的密集与高维特性，难以维持长时序的空间和时间一致性。为了解决这一限制，我们提出了Spatia——一种具备空间记忆感知能力的视频生成框架，该框架显式保存三维场景点云作为持久的空间记忆。Spatia 在该空间记忆的条件下迭代生成视频片段，并通过视觉 SLAM 持续更新记忆。这种动态–静态解耦的设计在增强生成过程中的空间一致性的同时，保留了模型生成真实动态对象的能力。此外，Spatia 支持诸如显式相机控制与具备三维感知的交互式编辑等应用，为可扩展的基于记忆的视频生成提供了几何学上有据可依的框架。

BibTeX

```
@article{2512.15716v1,
  title={Spatia: Video Generation with Updatable Spatial Memory},
  author={Jinjing Zhao and Fangyun Wei and Zhening Liu and Hongyang Zhang and Chang Xu and Yan Lu},
  journal={arXiv preprint arXiv:2512.15716v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15716v1}
}
```

## [Promoting Fairness in Information Access within Social Networks](http://arxiv.org/abs/2512.14711v1)

#信息获取公平性#社交网络#电阻距离

[PDF](https://arxiv.org/pdf/2512.14711v1)
[Abstract](http://arxiv.org/abs/2512.14711v1)

 社交网络/图算法/公平性

 很推荐

### 中文摘要

在线社交网络的兴起促进了信息的快速广泛传播，但由于网络位置的劣势，某些用户（尤其是少数群体成员）可能较少获得网络上传播的信息。我们研究了通过在网络中添加新连接以提升不同人口群体间信息获取公平性的优化问题。我们给出了一个具体的形式化描述，将信息获取用电阻距离（resistance distance）来度量，这一视角强调了全局网络结构与多路径连通性。该优化问题被证明是NP-困难的。我们提出了一个简单的贪心算法，尽管其在实践中能产生准确解，但其三次方的运行时间使其在大规模网络上不适用。作为主要技术贡献，我们通过若干新颖的近似技术将其时间复杂度降至线性。此外，我们还在真实和合成数据集上进行了大量实验，结果表明我们的线性时间算法能为包含数百万节点的网络生成准确的解。

BibTeX

```
@article{2512.14711v1,
  title={Promoting Fairness in Information Access within Social Networks},
  author={Changan Liu and Xiaotian Zhou and Ahad N. Zehmakan and Zhongzhi Zhang},
  journal={arXiv preprint arXiv:2512.14711v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14711v1}
}
```

## [SMART: Semantic Matching Contrastive Learning for Partially View-Aligned Clustering](http://arxiv.org/abs/2512.15396v1)

#多视图聚类#语义匹配#对比学习

[PDF](https://arxiv.org/pdf/2512.15396v1)
[Abstract](http://arxiv.org/abs/2512.15396v1)

 多视图聚类

 很推荐

### 中文摘要

多视图聚类通过利用数据多个视图之间固有的互补信息，已被实证能够提升学习性能。然而在实际场景中，严格对齐的视图往往难以获取，因此同时从对齐与未对齐数据中学习更为实际。部分视图对齐聚类（Partially View-aligned Clustering，PVC）旨在学习错配视图样本之间的对应关系，以更好地利用包括对齐和未对齐数据在内的视图间潜在一致性与互补性。然而，大多数现有的 PVC 方法未能充分利用未对齐数据来捕捉同一簇样本之间的共享语义。此外，多视图数据的内在异质性会导致表示上的分布偏移，进而使跨视图潜在特征之间难以建立准确且有意义的对应关系，从而削弱学习效果。为了解决这些挑战，本文提出了一种用于 PVC 的语义匹配对比学习模型（Semantic MAtching contRasTive learning，简称 SMART）。我们方法的核心思想是减轻跨视图分布偏移的影响，从而促进语义匹配的对比学习，充分挖掘对齐与未对齐数据中的语义关系。在八个基准数据集上的大量实验表明，我们的方法在 PVC 问题上持续优于现有方法。

BibTeX

```
@article{2512.15396v1,
  title={SMART: Semantic Matching Contrastive Learning for Partially View-Aligned Clustering},
  author={Liang Peng and Yixuan Ye and Cheng Liu and Hangjun Che and Fei Wang and Zhiwen Yu and Si Wu and Hau-San Wong},
  journal={arXiv preprint arXiv:2512.15396v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15396v1}
}
```

## [RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](http://arxiv.org/abs/2512.15219v1)

#知识图谱问答#关系驱动跳数选择#少样本路径引导

[PDF](https://arxiv.org/pdf/2512.15219v1)
[Abstract](http://arxiv.org/abs/2512.15219v1)

 LLM

 很推荐

### 中文摘要

大型语言模型（LLMs）在知识密集型问答中常因参数化知识的局限而产生幻觉。尽管像 KG-CoT 这样的现有方法通过整合知识图谱（KG）路径来提升可靠性，但它们在跳数选择上过于僵化（仅基于问题）且对推理路径的利用不足（缺乏引导）。为此我们提出 RFKG-CoT：首先，用关系驱动的自适应跳数选择器替代了僵化的跳数选择器，该选择器通过激活知识图谱中的关系（例如对直接的“兄弟”关系使用1跳，对间接的“父子”链使用2跳）动态调整推理步数，并通过关系掩码进行形式化；其次，引入了基于链式思维（CoT）的少样本上下文学习路径引导机制，以“问题-路径-答案”的示例格式构建示例，增强 LLM 对推理路径的理解能力。基于四个 KGQA 基准数据集的实验表明，RFKG-CoT 相较于 KG-CoT 在精度上最高提升 14.7 个百分点（在 WebQSP 上使用 Llama2-7B）。消融实验确认跳数选择器与路径提示具有互补性，共同将知识图谱证据转化为更可信的答案。

BibTeX

```
@article{2512.15219v1,
  title={RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA},
  author={Chao Zhang and Minghan Li and Tianrui Lv and Guodong Zhou},
  journal={arXiv preprint arXiv:2512.15219v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15219v1}
}
```

## [Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections](http://arxiv.org/abs/2512.14895v1)

#模仿学习#协变量偏移#在线专家校正（OEC）

[PDF](https://arxiv.org/pdf/2512.14895v1)
[Abstract](http://arxiv.org/abs/2512.14895v1)

 LLM

 很推荐

### 中文摘要

一种流行的训练语言模型（LM）代理的范式是通过模仿学习对专家轨迹进行微调。然而，我们指出，对于多轮交互的LM代理，模仿学习的离策略性质存在一个基本限制，即协变量偏移：当学生策略的行为逐渐偏离专家时，它会遇到训练数据中不存在的状态，从而降低微调的有效性。受经典DAgger算法的启发，我们提出了一种用于缓解多轮LLM训练中协变量偏移的新型数据生成方法。我们引入了在线专家校正（On-policy Expert Corrections，OEC），即部分在线的数据：先由学生模型启动滚动生成轨迹，然后在轨迹中途切换为专家模型以进行校正。我们在软件工程（SWE）任务领域——一个需要LLM代理与开发环境多轮交互以修复软件缺陷的场景——上评估了该数据生成技术的有效性。实验将OEC数据与其他多种在线方法和模仿学习方法在SWE代理问题上进行了比较，并采用结合环境奖励的拒绝采样与监督微调的统一训练流程。实验结果表明，在SWE-bench verified基准上，OEC轨迹相比传统模仿学习在7B和32B模型设置中分别带来约14%和13%的相对提升。我们的结果表明，在训练多轮LM代理时，需要将专家示范与在线（on-policy）数据相结合以获得更有效的训练。

BibTeX

```
@article{2512.14895v1,
  title={Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections},
  author={Niklas Lauffer and Xiang Deng and Srivatsa Kundurthy and Brad Kenstler and Jeff Da},
  journal={arXiv preprint arXiv:2512.14895v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14895v1}
}
```

## [A Roadmap for Applying Graph Neural Networks to Numerical Data: Insights from Cementitious Materials](http://arxiv.org/abs/2512.14855v1)

#图神经网络#胶凝/水泥基材料#表格数据转图/多模态

[PDF](https://arxiv.org/pdf/2512.14855v1)
[Abstract](http://arxiv.org/abs/2512.14855v1)

 GNN (图神经网络)

 很推荐

### 中文摘要

机器学习（ML）在混凝土研究中日益应用于性能优化和配方设计。然而，将机器学习用于胶凝材料的一大挑战是可用数据库的规模和多样性受限。一个有前景的解决方案是开发将数值与图形数据整合的多模态数据库。传统的胶凝材料研究中的机器学习框架通常仅限于单一数据模态。图神经网络（GNN）代表了新一代神经架构，能够从图结构数据中学习，通过不规则或基于拓扑的连接捕捉关系，而非依赖固定的空间坐标。尽管GNN天生适用于图形数据，但它们可以被改造用于从数值数据中提取相关性，并有潜力将物理规律直接嵌入到网络结构中，从而实现可解释且物理驱动的预测。本研究是少数将GNN应用于混凝土配方设计的工作之一，重点建立了使用k近邻（K-NN）方法将表格数据转换为图表示的清晰且可复现的路径。通过对模型超参数与特征选择的系统优化来提升预测性能。实验结果表明，该GNN在性能上可与作为基准的随机森林相媲美，而随机森林已被多项研究证明在胶凝材料预测中表现可靠。总体而言，本研究为从传统机器学习向更先进的AI架构过渡提供了基础性路线图，为未来发展能够捕捉复杂材料行为并加速胶凝材料设计与优化的多模态与物理驱动GNN模型奠定了坚实基础。

BibTeX

```
@article{2512.14855v1,
  title={A Roadmap for Applying Graph Neural Networks to Numerical Data: Insights from Cementitious Materials},
  author={Mahmuda Sharmin and Taihao Han and Jie Huang and Narayanan Neithalath and Gaurav Sant and Aditya Kumar},
  journal={arXiv preprint arXiv:2512.14855v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14855v1}
}
```

## [Guided Discrete Diffusion for Constraint Satisfaction Problems](http://arxiv.org/abs/2512.14765v1)

#离散扩散引导#约束满足问题 (CSP)#无监督数独求解

[PDF](https://arxiv.org/pdf/2512.14765v1)
[Abstract](http://arxiv.org/abs/2512.14765v1)

 约束满足问题 (CSP) / 生成模型

 很推荐

### 中文摘要

我们提出了一种用于约束满足问题（CSPs）的离散扩散引导方法，并展示了该方法在无监督条件下求解数独谜题的能力。

BibTeX

```
@article{2512.14765v1,
  title={Guided Discrete Diffusion for Constraint Satisfaction Problems},
  author={Justin Jung},
  journal={arXiv preprint arXiv:2512.14765v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14765v1}
}
```

## [Multiscale Cross-Modal Mapping of Molecular, Pathologic, and Radiologic Phenotypes in Lipid-Deficient Clear Cell Renal CellCarcinoma](http://arxiv.org/abs/2512.14750v1)

#跨尺度多模态映射#计算病理与放射组学整合#清细胞肾细胞癌分子分型

[PDF](https://arxiv.org/pdf/2512.14750v1)
[Abstract](http://arxiv.org/abs/2512.14750v1)

 医疗影像与计算病理（多模态肿瘤多组学）

 很推荐

### 中文摘要

清细胞肾细胞癌（ccRCC）在多种生物学尺度上表现出广泛的肿瘤内异质性，导致临床结局差异显著并削弱传统TNM分期的效能，这凸显了对多尺度整合分析框架的迫切需求。经多组学分析定义的缺脂去清细胞分化（DCCD）ccRCC亚型，即使在早期疾病中也与不良预后相关。本文建立了一种用于术前识别DCCD-ccRCC的分层跨尺度框架。在最高层，跨模态映射将分子特征转译为组织学与CT表型，构建了分子→病理→影像的监督桥梁。在该框架内，每个模态专属模型被设计为反映肿瘤生物学的内在层次结构：PathoDCCD捕捉从细胞形态与组织结构到中尺度区域组织的多尺度显微特征；RadioDCCD则通过结合整瘤及其栖息子区的放射组学特征与二维最大切面异质性度量，整合互补的宏观信息。这些嵌套模型实现了分子亚型的集成预测与临床风险分层。在总计1,659例患者的五个队列中，PathoDCCD稳定重现了分子亚型，RadioDCCD则提供了可靠的术前预测；两者一致的预测能识别出预后最差的患者。该跨尺度范式将分子生物学、计算病理学与定量放射学统一为一种具有生物学依据的ccRCC术前无创分子表型识别策略。

BibTeX

```
@article{2512.14750v1,
  title={Multiscale Cross-Modal Mapping of Molecular, Pathologic, and Radiologic Phenotypes in Lipid-Deficient Clear Cell Renal CellCarcinoma},
  author={Ying Cui and Dongzhe Zheng and Ke Yu and Xiyin Zheng and Xiaorui Wang and Xinxiang Li and Yan Gu and Lin Fu and Xinyi Chen and Wenjie Mei and Xin-Gui Peng},
  journal={arXiv preprint arXiv:2512.14750v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14750v1}
}
```

## [Zero-Knowledge Audit for Internet of Agents: Privacy-Preserving Communication Verification with Model Context Protocol](http://arxiv.org/abs/2512.14737v1)

#零知识证明#模型上下文协议(MCP)#隐私可验证审计

[PDF](https://arxiv.org/pdf/2512.14737v1)
[Abstract](http://arxiv.org/abs/2512.14737v1)

 多智能体系统 / 隐私与安全 / 密码学

 很推荐

### 中文摘要

现有的代理通信框架在提供可验证的审计轨迹时，常常无法在不泄露代理交互隐私与机密性的前提下实现审计要求。在需要精确计费、合规核验和监管问责的场景中，既保护代理通信隐私又保证可审计性成为一项核心挑战。
本文提出了一种用于审计代理通信的框架，能够在保持消息内容私密的同时验证其是否遵循预期规则。该方法将零知识证明与现有的模型上下文协议（Model Context Protocol, MCP）结合，使得在不揭示消息具体内容的情况下对消息格式和一般类型进行验证。该方案适用于轻量级网络、兼容标准的MCP交换，并引入了异步审计验证机制，用于确认消息格式与通用类别而不暴露细节。
该框架支持代理之间的相互审计——一方可以在不泄露敏感信息的前提下检查通信内容与质量，另一方则可核验使用量指标。我们形式化了安全目标，证明了 zk-MCP 在保证数据真实性与通信隐私方面的性质，并实现了高效的验证且带来可忽略的时延开销。为验证可行性，我们完整实现了该框架，包括基于 Circom 的零知识证明生成和与 MCP 双向通道集成的审计协议。据我们所知，这是首个在不暴露消息内容且不降低代理隐私的前提下，提供可验证相互审计的隐私保护代理通信审计系统。

BibTeX

```
@article{2512.14737v1,
  title={Zero-Knowledge Audit for Internet of Agents: Privacy-Preserving Communication Verification with Model Context Protocol},
  author={Guanlin Jing and Huayi Qi},
  journal={arXiv preprint arXiv:2512.14737v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14737v1}
}
```

## [PyFi: Toward Pyramid-like Financial Image Understanding for VLMs via Adversarial Agents](http://arxiv.org/abs/2512.14735v1)

#金融视觉语言模型#金字塔式分层推理#多智能体对抗合成数据

[PDF](https://arxiv.org/pdf/2512.14735v1)
[Abstract](http://arxiv.org/abs/2512.14735v1)

 VLM

 很推荐

### 中文摘要

本文提出了PyFi，一种面向金字塔式金融图像理解的新框架，使视觉-语言模型（VLMs）能够通过一系列由易到难的问题链进行渐进推理。PyFi的核心是PyFi-600K数据集，包含60万条按推理金字塔组织的金融问答对：金字塔底层的问题仅需基础感知能力，而越靠近顶层的问题则要求在金融视觉理解和专业知识上具备更高的能力。该数据集可扩展地通过无需人工标注的方式合成，采用PyFi-adv——一种在蒙特卡洛树搜索（MCTS）范式下的多智能体对抗机制。在该机制中，对于每张图像，挑战者智能体与求解者智能体竞争，生成逐步深入的问答链以探测金融视觉推理能力的不同层级。基于此数据集，本文对先进VLM在金融领域进行了细粒度、层次化且全面的评估。此外，将Qwen2.5-VL-3B和Qwen2.5-VL-7B在该金字塔结构的问题链上进行微调后，模型能够通过将复杂金融问题分解为逐步增加推理难度的子问题来作答，在该数据集上分别实现了平均准确率提升19.52%和8.06%。所有代码、数据集与模型均已开源，地址为：https://github.com/AgenticFinLab/PyFi 。

BibTeX

```
@article{2512.14735v1,
  title={PyFi: Toward Pyramid-like Financial Image Understanding for VLMs via Adversarial Agents},
  author={Yuqun Zhang and Yuxuan Zhao and Sijia Chen},
  journal={arXiv preprint arXiv:2512.14735v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14735v1}
}
```

## [HATSolver: Learning Groebner Bases with Hierarchical Attention Transformers](http://arxiv.org/abs/2512.14722v1)

#Groebner基#层次注意力变换器#符号计算

[PDF](https://arxiv.org/pdf/2512.14722v1)
[Abstract](http://arxiv.org/abs/2512.14722v1)

 Symbolic Computation / Math-AI

 很推荐

### 中文摘要

在 NeurIPS 2024 中，Kera 等人首次提出使用变换器（transformer）来计算 Groebner 基——这一在计算代数中具有核心地位且具有广泛实际应用的对象。本文通过将层次注意力变换器（Hierarchical Attention Transformers，HATs）引入 Groebner 基计算，改进了这一方法，用于求解多元多项式方程组。HAT 结构内置树状的归纳偏置，能够建模数据中存在的层次关系，从而相比传统的平面注意力模型带来显著的计算节省。我们将该方法推广到任意深度并给出了详细的计算成本分析；结合课程学习（curriculum learning），所提出的方法能够解决远大于 Kera et al.（2024）中实例规模的问题。

BibTeX

```
@article{2512.14722v1,
  title={HATSolver: Learning Groebner Bases with Hierarchical Attention Transformers},
  author={Mohamed Malhou and Ludovic Perret and Kristin Lauter},
  journal={arXiv preprint arXiv:2512.14722v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14722v1}
}
```

## [SoMe: A Realistic Benchmark for LLM-based Social Media Agents](http://arxiv.org/abs/2512.14720v1)

#社交媒体代理#LLM 基准评测#多任务数据集

[PDF](https://arxiv.org/pdf/2512.14720v1)
[Abstract](http://arxiv.org/abs/2512.14720v1)

 LLM

 很推荐

### 中文摘要

由大型语言模型（LLM）驱动的智能代理近年来在社交媒体平台上展现了强大能力并越来越受欢迎。尽管 LLM 代理正在重塑社交媒体生态，但目前尚缺乏对其理解媒体内容、把握用户行为和做出复杂决策能力的全面评估。为此，我们提出了 SoMe —— 一个面向配备多种代理工具以访问和分析社交媒体数据的 LLM 驱动社交媒体代理的开创性基准。SoMe 包含 8 类多样化的社交媒体代理任务、9,164,284 条帖子、6,591 个用户档案、25,686 条从各类社交媒体平台与外部网站收集的报告，以及 17,869 条经精心注释的任务查询。与现有的社交媒体任务数据集和基准相比，SoMe 首次为基于 LLM 的社交媒体代理提供了一个多功能且贴近真实场景的平台，能够处理多样化的社交媒体任务。通过大量的定量与定性分析，我们首次概览了主流具代理能力的 LLM 在真实社交媒体环境中的表现，并指出若干局限性。评估结果表明，当前的闭源与开源 LLM 均无法令人满意地完成社交媒体代理任务。SoMe 为未来社交媒体代理研究提供了一个具有挑战性且富有意义的测试床。代码与数据已在 https://github.com/LivXue/SoMe 上公开。

BibTeX

```
@article{2512.14720v1,
  title={SoMe: A Realistic Benchmark for LLM-based Social Media Agents},
  author={Dizhan Xue and Jing Cui and Shengsheng Qian and Chuanrui Hu and Changsheng Xu},
  journal={arXiv preprint arXiv:2512.14720v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14720v1}
}
```

## [Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](http://arxiv.org/abs/2512.14709v1)

#向量符号架构(VSA)#自注意力绑定/解绑定#符号化推理与可组合性

[PDF](https://arxiv.org/pdf/2512.14709v1)
[Abstract](http://arxiv.org/abs/2512.14709v1)

 LLM

 很推荐

### 中文摘要

基于Transformer的语言模型表现出近似推理的行为，但在需要稳定符号操作的任务上仍然脆弱。本文提出一个统一视角，将自注意力机制和残差流解释为近似的向量符号架构（Vector Symbolic Architecture, VSA）。在该视角下，query与key定义角色空间，value编码填充项（fillers），注意力权重执行软解绑定（soft unbinding），残差连接实现多个已绑定结构的叠加（superposition）。我们用这一代数镜像将Transformer内部机制与chain-of-thought轨迹、基于程序的推理以及带记忆的工具使用联系起来，并解释了诸如变量混淆和对逻辑相关提示不一致等典型失效模式。基于此观点，本文提出了受VSA启发的架构偏置，包括显式的绑定/解绑定头、超维记忆层，以及促进角色-填充分离和稳健叠加的训练目标。最后，我们给出衡量“VSA相似性”和逻辑可组合性的度量建议，并提出若干理论与架构上的开放问题。总体而言，将注意力视为软向量-符号计算为构建更可解释、逻辑更可靠的推理系统提供了一条有原则的路径。

BibTeX

```
@article{2512.14709v1,
  title={Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning},
  author={Sahil Rajesh Dhayalkar},
  journal={arXiv preprint arXiv:2512.14709v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14709v1}
}
```

## [HD-Prot: A Protein Language Model for Joint Sequence-Structure Modeling with Continuous Structure Tokens](http://arxiv.org/abs/2512.15133v1)

#蛋白质语言模型#连续结构表示#混合扩散模型

[PDF](https://arxiv.org/pdf/2512.15133v1)
[Abstract](http://arxiv.org/abs/2512.15133v1)

 LLM

 很推荐

### 中文摘要

蛋白质天然具有一致的序列-结构二元性。大量可用的蛋白质序列数据可被方便地表示为离散的符号，这推动了蛋白质语言模型（pLM）的快速发展。然而，如何将连续的结构信息有效地融入pLM仍是一个重要挑战。现有方法通常通过对蛋白质结构进行离散化以适配语言建模框架，但这不可避免地导致细粒度信息丢失，从而限制了多模态pLM的性能潜力。本文提出可以绕过上述问题的思路：在基于序列的pLM之上引入连续结构标记，即高保真的蛋白质结构潜变量，避免向量量化。具体而言，我们提出了一种混合扩散蛋白质语言模型HD-Prot：在离散pLM之上嵌入一个连续值的扩散头，使模型可以同时处理离散与连续标记以实现序列-结构联合建模。该模型通过统一的吸收扩散过程捕获跨模态的标记间依赖关系，并通过对序列使用类别预测、对结构使用连续扩散的方式估计每个标记的分布。大量实证结果表明，HD-Prot在无条件序列-结构联合生成、基序支架化、蛋白质结构预测和逆折叠等任务上均取得了有竞争力的表现，尽管在有限计算资源下训练，其结果仍可与最先进的多模态pLM媲美。研究表明在统一的语言模型架构中同时估计类别型与连续型分布是可行的，为多模态蛋白质语言模型提供了一条有前景的替代方向。

BibTeX

```
@article{2512.15133v1,
  title={HD-Prot: A Protein Language Model for Joint Sequence-Structure Modeling with Continuous Structure Tokens},
  author={Yi Zhou and Haohao Qu and Yunqing Liu and Shanru Lin and Le Song and Wenqi Fan},
  journal={arXiv preprint arXiv:2512.15133v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15133v1}
}
```

## [Tracking spatial temporal details in ultrasound long video via wavelet analysis and memory bank](http://arxiv.org/abs/2512.15066v1)

#超声视频分割#小波分析#记忆库跟踪

[PDF](https://arxiv.org/pdf/2512.15066v1)
[Abstract](http://arxiv.org/abs/2512.15066v1)

 CV（医学影像）

 很推荐

### 中文摘要

医学超声视频被广泛用于体检、疾病诊断和手术规划。高保真病灶区域与目标器官的分割是计算机辅助外科流程中的关键环节。然而，超声视频普遍存在对比度低和背景噪声大等问题，易导致器官边界分割错误、弱小目标丢失并增加边界误差，且长视频中的目标跟踪仍具有显著挑战。为解决上述问题，本文提出了一种基于记忆库的小波滤波与融合网络（memory bank-based wavelet filtering and fusion network），该网络采用编码器—解码器结构，有效提取细粒度空间特征并融合高频（HF）信息。具体地，在编码器中引入了基于记忆的波let卷积，以同时捕捉类别信息、细节信息并利用相邻帧信息；通过级联小波压缩融合多尺度频域特征并在每个卷积层内扩展感受野；设计了结合交叉注意力与记忆压缩机制的长短时记忆库用于长视频目标跟踪；在解码器中，提出了基于自适应小波滤波器的高频感知特征融合模块，以充分利用对边界敏感的高频细节特征。在四个超声视频基准数据集（两个甲状腺结节集、甲状腺器官集、心脏集）上的大量实验中，与最先进方法相比，本方法在分割指标上取得了显著提升，尤其对长视频中小型甲状腺结节的分割更为准确，验证了其在长超声视频小目标场景中的有效性。代码已开源于 https://github.com/XiAooZ/MWNet。

BibTeX

```
@article{2512.15066v1,
  title={Tracking spatial temporal details in ultrasound long video via wavelet analysis and memory bank},
  author={Chenxiao Zhang and Runshi Zhang and Junchen Wang},
  journal={arXiv preprint arXiv:2512.15066v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15066v1}
}
```

## [Agentic AI for Integrated Sensing and Communication: Analysis, Framework, and Case Study](http://arxiv.org/abs/2512.15044v1)

#集成感知与通信（ISAC）#Agentic AI（智能体）#生成式人工智能（GenAI）

[PDF](https://arxiv.org/pdf/2512.15044v1)
[Abstract](http://arxiv.org/abs/2512.15044v1)

 RL / 无线通信 (ISAC)

 很推荐

### 中文摘要

集成感知与通信（ISAC）已成为第六代（6G）时代的重要发展方向，为未来智能网络的协同感知与通信提供了关键支撑。然而，随着无线环境日益动态且复杂，ISAC 系统需要更智能的处理能力和更高的自主运行能力以维持效率与适应性。与此同时，agentic 人工智能通过在动态环境中实现持续的感知—推理—行动闭环，为 ISAC 系统提供了一种可行的解决方案，从而支持其智能化、自主化与高效化运行。基于此，本文深入探讨了 agentic AI 在 ISAC 系统中的应用价值与发展前景。首先，我们对 agentic AI 与 ISAC 系统进行了全面综述，阐明了二者的关键特征；其次，展示了若干常见的 ISAC 优化方法，并突出基于生成式人工智能（GenAI）的 agentic AI 所带来的显著优势；第三，我们提出了一种新颖的 agentic ISAC 框架，并通过案例研究验证了其在优化 ISAC 性能方面的优越性；最后，明确了基于 agentic AI 的 ISAC 系统未来的研究方向。

BibTeX

```
@article{2512.15044v1,
  title={Agentic AI for Integrated Sensing and Communication: Analysis, Framework, and Case Study},
  author={Wenwen Xie and Geng Sun and Ruichen Zhang and Xuejie Liu and Yinqiu Liu and Jiacheng Wang and Dusit Niyato and Ping Zhang},
  journal={arXiv preprint arXiv:2512.15044v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15044v1}
}
```

## [VTCBench: Can Vision-Language Models Understand Long Context with Vision-Text Compression?](http://arxiv.org/abs/2512.15649v1)

#视觉-文本压缩#长上下文理解#视觉-语言模型

[PDF](https://arxiv.org/pdf/2512.15649v1)
[Abstract](http://arxiv.org/abs/2512.15649v1)

 VLM (视觉-语言模型)

 很推荐

### 中文摘要

随着将上下文窗口扩展到更长范围所带来的计算和内存开销不断增加，可扩展性受到了严重限制。一个值得注意的解决方案是视觉-文本压缩（VTC），例如 DeepSeek-OCR 和 Glyph 等框架，它们将长文本转换为稠密的二维视觉表示，从而实现 3x–20x 的 token 压缩比。然而，高信息密度对视觉-语言模型（VLM）核心长上下文能力的影响尚未得到充分研究。为填补这一空白，我们提出了首个针对 VTC 的基准 VTCBench，并在三个长上下文理解场景下系统评估了 VLM 的性能：VTC-Retrieval（评估模型检索与聚合信息的能力）、VTC-Reasoning（要求模型推断隐含关联以在词汇重合极少时定位事实）和 VTC-Memory（衡量在长期对话记忆中进行综合问答的能力）。此外，我们构建了 VTCBench-Wild 以模拟多样化输入场景。我们对领先的开源与专有模型进行了全面评测。结果表明，尽管这些模型在解码文本（如 OCR）方面表现良好，但在处理 VTC 压缩信息时，大多数 VLM 的长上下文理解能力出人意料地薄弱，难以捕捉上下文中的长程关联或依赖关系。本研究加深了对 VTC 的认识，为设计更高效、可扩展的视觉-语言模型奠定了基础。

BibTeX

```
@article{2512.15649v1,
  title={VTCBench: Can Vision-Language Models Understand Long Context with Vision-Text Compression?},
  author={Hongbo Zhao and Meng Wang and Fei Zhu and Wenzhuo Liu and Bolin Ni and Fanhu Zeng and Gaofeng Meng and Zhaoxiang Zhang},
  journal={arXiv preprint arXiv:2512.15649v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15649v1}
}
```

## [A Conditioned UNet for Music Source Separation](http://arxiv.org/abs/2512.15532v1)

#音乐源分离#条件化UNet#稀疏压缩网络

[PDF](https://arxiv.org/pdf/2512.15532v1)
[Abstract](http://arxiv.org/abs/2512.15532v1)

 音频信号处理/音乐源分离 (MSS)

 很推荐

### 中文摘要

本文提出了一种用于音乐源分离（MSS）的条件化UNet。传统上，MSS通常由多输出神经网络（典型的是UNet）完成，每个输出对应预定义乐器词汇表中的一个声部。与此不同，条件化的MSS网络在输入待分离信号的同时还接受一个与目标声部相关的音频查询，因此不需要严格的词汇表，从而能够支持更现实的分离任务。条件化方法的潜力此前受到合适数据缺乏的限制，近期通过MoisesDb数据集得到了改善。近来的方法Banquet在该数据集上对更大词汇表显示了有前景的结果，且采用的是Bandsplit RNN而非UNet，作者甚至认为UNet不适于条件化MSS。我们针对这一观点提出反驳，设计了QSCNet——一种新颖的条件化UNet，将网络条件化组件整合入用于MSS的稀疏压缩网络（Sparse Compressed Network）。实验表明，QSCNet在若干MSS任务上比Banquet提高了超过1 dB的SNR，同时参数量不到后者的一半。

BibTeX

```
@article{2512.15532v1,
  title={A Conditioned UNet for Music Source Separation},
  author={Ken O'Hanlon and Basil Woods and Lin Wang and Mark Sandler},
  journal={arXiv preprint arXiv:2512.15532v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15532v1}
}
```

## [Soft Geometric Inductive Bias for Object Centric Dynamics](http://arxiv.org/abs/2512.15493v1)

#几何代数神经网络#对象中心动力学#软等变性归纳偏置

[PDF](https://arxiv.org/pdf/2512.15493v1)
[Abstract](http://arxiv.org/abs/2512.15493v1)

 RL

 很推荐

### 中文摘要

等变性是学习物理动力学的强有力先验，但当对称性被破坏时，严格的群等变性反而会降低性能。我们提出了一种基于几何代数神经网络的对象中心世界模型，为模型提供一种柔性的几何归纳偏置。在带有静态障碍物的二维刚体动力学模拟环境中评估所提模型，在这些环境中我们以自回归方式训练下一步预测。对于长时域滚动预测，我们展示了相对于非等变基线模型，模型的软归纳偏置在物理保真度方面带来了更好的表现。该方法补充了近期的软等变性思想，并支持这样一种观点：简单且精心选择的先验能够带来鲁棒的泛化。这些结果表明，几何代数在手工物理模型与无结构深度网络之间提供了一个有效的折中方案，可为多物体场景提供样本高效的动力学模型。

BibTeX

```
@article{2512.15493v1,
  title={Soft Geometric Inductive Bias for Object Centric Dynamics},
  author={Hampus Linander and Conor Heins and Alexander Tschantz and Marco Perin and Christopher Buckley},
  journal={arXiv preprint arXiv:2512.15493v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15493v1}
}
```

## [mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs](http://arxiv.org/abs/2512.15692v1)

#视频预训练#逆向动力学模型#机器人控制

[PDF](https://arxiv.org/pdf/2512.15692v1)
[Abstract](http://arxiv.org/abs/2512.15692v1)

 Robotics

 很推荐

### 中文摘要

现有用于机器人操作的视觉-语言-动作模型（VLA）大多建立在通过大规模但相互脱节的静态网络数据预训练的视觉-语言骨干之上。因此，尽管在语义泛化方面有所提升，策略仍然必须仅从机器人轨迹中隐式推断复杂的物理动力学和时间依赖性。这种依赖造成了难以为继的数据负担，迫使研究者持续大量收集专家示范以弥补对物理因果理解的缺失。我们认为，视觉-语言预训练虽然有效捕捉了语义先验，但对物理因果并不敏感。更有效的范式是在预训练阶段利用视频同时捕捉语义与视觉动态，从而将剩余任务限定为低层控制。为此，我们提出了 mimic-video，一种新颖的视频-动作模型（VAM），它将预训练的互联网规模视频模型与基于流匹配的动作解码器结合，动作解码器以视频模型的潜在表示为条件。该解码器作为逆动力学模型（IDM），根据视频空间中的动作规划的潜在表示生成低层机器人动作。大量评估表明，我们的方法在仿真和真实世界的机器人操作任务上均达到了最先进的性能，与传统的VLA架构相比，样本效率提高了约10倍，收敛速度加快了约2倍。

BibTeX

```
@article{2512.15692v1,
  title={mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs},
  author={Jonas Pai and Liam Achenbach and Victoriano Montesinos and Benedek Forrai and Oier Mees and Elvis Nava},
  journal={arXiv preprint arXiv:2512.15692v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15692v1}
}
```

## [Stepwise Think-Critique: A Unified Framework for Robust and Interpretable LLM Reasoning](http://arxiv.org/abs/2512.15662v1)

#逐步思考与自我批判#可解释性推理#混合强化学习训练

[PDF](https://arxiv.org/pdf/2512.15662v1)
[Abstract](http://arxiv.org/abs/2512.15662v1)

 LLM

 很推荐

### 中文摘要

人类通过批判性思维解决复杂问题，在这一过程中推理与评估交织进行，从而朝着正确解答收敛。然而，大多数现有的大型语言模型（LLM）将推理与验证解耦：它们要么生成推理但没有明确的自我检查，要么依赖外部验证器在事后发现错误。前者缺乏即时反馈，后者则增加系统复杂性并阻碍同步学习。受人类批判性思维的启发，我们提出了逐步思考-批判（Stepwise Think-Critique，STC）——一个在单一模型内部将推理与自我批判在每一步交替进行的统一框架。STC 使用混合强化学习目标进行训练，结合推理奖励与批判一致性奖励，以联合优化推理质量与自我评估能力。在数学推理基准上的实验证明，STC 展现出强大的批判性思维能力并生成更具可解释性的推理轨迹，代表了朝向内建批判性思维的 LLM 前进的一步。

BibTeX

```
@article{2512.15662v1,
  title={Stepwise Think-Critique: A Unified Framework for Robust and Interpretable LLM Reasoning},
  author={Jiaqi Xu and Cuiling Lan and Xuejin Chen and Yan LU},
  journal={arXiv preprint arXiv:2512.15662v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15662v1}
}
```

## [Bilateral Spatial Reasoning about Street Networks: Graph-based RAG with Qualitative Spatial Representations](http://arxiv.org/abs/2512.15388v1)

#定性空间关系#图上检索增强生成（Graph-based RAG）#路网双边空间推理

[PDF](https://arxiv.org/pdf/2512.15388v1)
[Abstract](http://arxiv.org/abs/2512.15388v1)

 LLM

 很推荐

### 中文摘要

本文探讨通过定性空间关系改进大型语言模型（LLM）在为行人提供路线指引时的能力，旨在利用街道网络上的空间语义信息提升导航指令的准确性与可理解性。

BibTeX

```
@article{2512.15388v1,
  title={Bilateral Spatial Reasoning about Street Networks: Graph-based RAG with Qualitative Spatial Representations},
  author={Reinhard Moratz and Niklas Daute and James Ondieki and Markus Kattenbeck and Mario Krajina and Ioannis Giannopoulos},
  journal={arXiv preprint arXiv:2512.15388v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15388v1}
}
```

## [Emotion Recognition in Signers](http://arxiv.org/abs/2512.15376v1)

#手语情感识别#跨语种迁移#时序片段选择

[PDF](https://arxiv.org/pdf/2512.15376v1)
[Abstract](http://arxiv.org/abs/2512.15376v1)

 CV

 很推荐

### 中文摘要

手语使用者的情感识别面临一项理论性挑战和一项实践性挑战：即语法性与情感性面部表情的重叠，以及用于模型训练的数据稀缺。本文在跨语种设置下针对这两类挑战提出了解决方案，使用了我们的 eJSL 数据集——一个用于日本手语使用者情感识别的新基准数据集，以及带字幕的大型英国手语数据集 BOBSL。在 eJSL 中，两名手语者以七种不同情绪分别表达了78条不同的语句，共产生1092段视频剪辑。实证结果表明：1）在口语文本上进行的情感识别可缓解手语数据稀缺问题；2）时序片段选择对识别效果有显著影响；3）引入手部动作信息可提升手语使用者的情感识别性能。最后，我们建立了一个比基于口语的语言模型更强的基线。

BibTeX

```
@article{2512.15376v1,
  title={Emotion Recognition in Signers},
  author={Kotaro Funakoshi and Yaoxiong Zhu},
  journal={arXiv preprint arXiv:2512.15376v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15376v1}
}
```

## [Image Complexity-Aware Adaptive Retrieval for Efficient Vision-Language Models](http://arxiv.org/abs/2512.15372v1)

#图像复杂度感知#自适应检索#高效视觉-语言模型

[PDF](https://arxiv.org/pdf/2512.15372v1)
[Abstract](http://arxiv.org/abs/2512.15372v1)

 视觉-语言模型 (Vision-Language Models)

 很推荐

### 中文摘要

视觉-语言模型中的视觉Transformer对所有图像施加相同的计算量：无论是简单的产品照片还是复杂的街景，都可能消耗相同的175.33 GFLOPs（ViT-L/14）。我们提出ICAR（Image Complexity-Aware Retrieval，图像复杂度感知自适应检索），使视觉Transformer能够对简单图像使用更少计算，而对复杂图像则沿用完整网络深度处理。关键挑战在于保持跨模态对齐：来自不同处理深度的图像嵌入必须在文本匹配时保持兼容。ICAR通过双路径训练解决该问题，生成在减算力路径和全算力路径下均兼容的嵌入，从而保证无论图像提前退出还是完整处理，其图像表征都能与文本嵌入位于相同语义空间并直接匹配。与需要昂贵重排序的现有两阶段方法不同，ICAR无需额外开销即可实现直接的图文匹配。为确定使用多少计算量，我们提出ConvNeXt-IC，将图像复杂度评估视为分类任务。通过采用现代分类器主干而非专用架构，ConvNeXt-IC在与人工判断的Pearson相关性上达到了0.959并实现4.4倍的加速。在加入真实网页数据的标准基准测试上，ICAR在保持类别级性能的同时实现了约20%的实际加速，并保持95%的实例级性能，从而有利于视觉-语言系统的可持续扩展。

BibTeX

```
@article{2512.15372v1,
  title={Image Complexity-Aware Adaptive Retrieval for Efficient Vision-Language Models},
  author={Mikel Williams-Lekuona and Georgina Cosma},
  journal={arXiv preprint arXiv:2512.15372v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15372v1}
}
```

## [Empirical Investigation of the Impact of Phase Information on Fault Diagnosis of Rotating Machinery](http://arxiv.org/abs/2512.15344v1)

#相位对齐#旋转机械故障诊断#多轴振动信号

[PDF](https://arxiv.org/pdf/2512.15344v1)
[Abstract](http://arxiv.org/abs/2512.15344v1)

 故障诊断/预测性维护/时间序列信号处理

 很推荐

### 中文摘要

预测性维护中对旋转机械的监测越来越依赖振动信号，但大多数基于学习的方法在频谱特征提取时要么舍弃相位信息，要么直接使用时域波形而未显式利用相位信息。本文提出两种面向相位的预处理策略，用以解决多轴振动数据中的随机相位变化：(1) 三轴独立相位调整：对每个轴分别对齐至零相位；(2) 单轴参考相位调整：通过对所有轴施加相同的时移以保持轴间相位关系。基于一套使用同步三轴传感器采集的新型转子数据集，我们在两阶段学习框架下评估了六种深度学习架构。实验结果表明，两种相位对齐方法均能带来与架构无关的性能提升：三轴独立方法带来稳定增益（例如 Transformer 提升约 +2.7%），而单轴参考方法通过保留空间相位关系实现更优表现，最高达 96.2% 的准确率（提升约 +5.4%）。这些结果表明，两种相位对齐策略均为预测性维护系统提供了实用且可扩展的改进手段。

BibTeX

```
@article{2512.15344v1,
  title={Empirical Investigation of the Impact of Phase Information on Fault Diagnosis of Rotating Machinery},
  author={Hiroyoshi Nagahama and Katsufumi Inoue and Masayoshi Todorokihara and Michifumi Yoshioka},
  journal={arXiv preprint arXiv:2512.15344v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15344v1}
}
```

## [Graph Pattern-based Association Rules Evaluated Under No-repeated-anything Semantics in the Graph Transactional Setting](http://arxiv.org/abs/2512.15308v1)

#图模式#关联规则#概率度量

[PDF](https://arxiv.org/pdf/2512.15308v1)
[Abstract](http://arxiv.org/abs/2512.15308v1)

 Graph Mining

 很推荐

### 中文摘要

我们提出了面向有向带标注多重图（例如 RDF 图）的基于图模式的关联规则（Graph Pattern-based Association Rules，GPARs）。GPARs 支持生成式任务（在图上进行扩展）和评估式任务（评估图的合理性）。该框架超越了已有的形式化方法，如图函数依赖、图实体依赖、关系型关联规则、图关联规则、多关系与路径关联规则以及 Horn 规则。对于给定的图集合，我们在“无重复任意”（no-repeated-anything）语义下评估图模式，这一语义使得图的拓扑结构能够被更有效地纳入考量。我们构造了一个概率空间，并在概率意义下推导了置信度、提升度（lift）、杠杆度（leverage）和确信度（conviction）等度量。进一步地，我们分析了这些度量与经典基于项集的对应度量之间的关系，并指出了在何种条件下它们的特征性质能够被保留。

BibTeX

```
@article{2512.15308v1,
  title={Graph Pattern-based Association Rules Evaluated Under No-repeated-anything Semantics in the Graph Transactional Setting},
  author={Basil Ell},
  journal={arXiv preprint arXiv:2512.15308v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15308v1}
}
```

## [Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](http://arxiv.org/abs/2512.15274v1)

#前缀优化#强化学习（RLVR）#LLM推理提升

[PDF](https://arxiv.org/pdf/2512.15274v1)
[Abstract](http://arxiv.org/abs/2512.15274v1)

 LLM

 很推荐

### 中文摘要

强化学习与可验证奖励（RLVR）能显著提升大语言模型（LLM）的推理能力。现有RLVR方法通常对生成的所有令牌进行统一训练，但未区分哪些令牌（例如前缀令牌）对推理贡献更大，导致大量资源浪费在低回报令牌上，抑制了对高回报令牌的优化效果。为此本文提出了一种新的RLVR方法：渐进前缀令牌策略优化（PPPO），强调生成输出前缀段的重要性。受人类“路径依赖”思维理论启发，我们发现LLM推理存在类似的“初始锁定效应”（BLE），即早期思路显著制约后续推理轨迹。PPPO通过以优化模型的前缀推理过程为目标，间接改善后续推理并提升最终结果。为提高模型学习高质量起始推理的能力，PPPO引入两项训练策略：(a) 渐进前缀保留，通过在训练中逐步增加保留的前缀令牌比例来构建渐进学习过程；(b) 连续累积奖励，通过对同一前缀采样多条续写并累积其得分来缓解奖励偏差。大量推理任务的实验结果表明，PPPO优于代表性RLVR方法，在仅使用26.17%训练令牌的情况下实现了18.02%的准确率提升。

BibTeX

```
@article{2512.15274v1,
  title={Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning},
  author={Yiliu Sun and Zicheng Zhao and Yang Wei and Yanfang Zhang and Chen Gong},
  journal={arXiv preprint arXiv:2512.15274v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15274v1}
}
```

## [Intersectional Fairness in Vision-Language Models for Medical Image Disease Classification](http://arxiv.org/abs/2512.15249v1)

#交叉群体公平性#视觉-语言模型#医疗影像诊断

[PDF](https://arxiv.org/pdf/2512.15249v1)
[Abstract](http://arxiv.org/abs/2512.15249v1)

 医疗多模态（医疗视觉-语言模型）

 很推荐

### 中文摘要

医疗人工智能（AI）系统，尤其是多模态视觉—语言模型（VLM），常常表现出交叉群体偏差，即模型在诊断被边缘化的患者子群时系统性地置信度较低。此类偏差可能导致不准确或漏诊率升高，原因在于数据在人口统计学上的偏斜以及诊断置信度分布的差异。现有的公平性干预方法常常无法弥补这些差距，或在追求子群间统计平等时牺牲整体诊断性能。本研究提出了跨模态对齐一致性（Cross-Modal Alignment Consistency，CMAC-MMD）训练框架，用以在交叉群体患者子群间标准化诊断置信度。与传统去偏方法不同，该方法在临床推断阶段无需使用敏感人口学数据即可均衡模型的决策置信度。我们在10,015张皮肤病变图像（HAM10000）上进行了训练，并在12,000张外部验证图像（BCN20000）及10,000张用于青光眼检测的视网膜图像（Harvard-FairVLMed）上做了外部验证，按年龄、性别和种族的交叉属性对性能进行了分层评估。在皮肤科队列中，所提方法将总体交叉群体漏诊差距（真正例率差异，ΔTPR）从0.50降至0.26，同时总体AUC从0.94提升至0.97，相较于标准训练均有改进。类似地，在青光眼筛查任务中，ΔTPR从0.41降至0.31，AUC由基线的0.71提升至0.72。本研究建立了一个可扩展的框架，用于开发既准确又能在多样化患者子群间实现公平表现的高风险临床决策支持系统，同时降低了对隐私敏感信息的依赖。

BibTeX

```
@article{2512.15249v1,
  title={Intersectional Fairness in Vision-Language Models for Medical Image Disease Classification},
  author={Yupeng Zhang and Adam G. Dunn and Usman Naseem and Jinman Kim},
  journal={arXiv preprint arXiv:2512.15249v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15249v1}
}
```

## [Automatic Reward Shaping from Multi-Objective Human Heuristics](http://arxiv.org/abs/2512.15120v1)

#奖励塑形#多目标强化学习#双层优化

[PDF](https://arxiv.org/pdf/2512.15120v1)
[Abstract](http://arxiv.org/abs/2512.15120v1)

 RL

 很推荐

### 中文摘要

在多目标环境中，设计有效的奖励函数仍然是强化学习的一大挑战。在本文中，我们提出了多目标奖励塑形与探索框架（MORSE），该通用框架可将多个由人类设计的启发式奖励自动组合为统一的奖励函数。MORSE 将奖励塑形过程表述为一个双层优化问题：内循环训练策略以最大化当前的塑形奖励，外循环则更新奖励函数以优化任务性能。为鼓励在奖励空间中的探索并避免陷入次优局部极值，MORSE 在塑形过程中引入了随机性，通过任务性能和一个固定、随机初始化神经网络的预测误差来引导噪声注入。MuJoCo 和 Isaac Sim 环境的实验结果表明，MORSE 能在多种机器人任务中有效平衡多重目标，所达成的任务性能与人工调优的奖励函数相当。

BibTeX

```
@article{2512.15120v1,
  title={Automatic Reward Shaping from Multi-Objective Human Heuristics},
  author={Yuqing Xie and Jiayu Chen and Wenhao Tang and Ya Zhang and Chao Yu and Yu Wang},
  journal={arXiv preprint arXiv:2512.15120v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15120v1}
}
```

## [How Many Heads Make an SSM? A Unified Framework for Attention and State Space Models](http://arxiv.org/abs/2512.15115v1)

#序列建模理论#注意力与状态空间模型统一#交互秩与梯度传播

[PDF](https://arxiv.org/pdf/2512.15115v1)
[Abstract](http://arxiv.org/abs/2512.15115v1)

 NLP

 很推荐

### 中文摘要

序列建模产生了多样的架构——从传统的循环神经网络到现代的Transformer和状态空间模型（SSM）——但关于表达能力与可训练性之间权衡的统一理论认识仍然有限。我们提出了一个统一框架，用输入依赖的有效交互算子 W\_{ij}(X) 表示一类广泛的序列映射，明确了两种经常出现的构造模式： (i) 统一因子化显式框架（Unified Factorized Framework，显式，类似注意力的混合），其中 W\_{ij}(X) 通过标量系数作用于共享的值映射而变化；(ii) 结构化动力学（Structured Dynamics，隐式，状态空间递归），其中 W\_{ij} 由潜在动力系统诱导。基于该框架，我们推导了三个理论结果。首先，提出了“交互秩差距”（Interaction Rank Gap）：处于统一因子化框架的模型（例如单头注意力）被约束在低维的算子张成空间内，因而无法表示某些结构化的动力学映射。其次，证明了一个“等价（头数）定理”（Equivalence (Head-Count) Theorem）：在我们的多头因子化类中，表示一个其滞后算子在长度为 n 的序列上张成 k 维子空间的线性SSM，恰好需要且可由 H=k 个头实现。第三，证明了一个“梯度高速公路”结果（Gradient Highway Result）：注意力层存在与距离无关的梯度路径，而稳定的线性动力学则表现出随距离衰减的梯度衰减。综上，这些结果形式化地揭示了代数表达能力（交互/算子张成空间）与长程梯度传播之间的根本权衡，为现代序列架构设计提供了理论依据。

BibTeX

```
@article{2512.15115v1,
  title={How Many Heads Make an SSM? A Unified Framework for Attention and State Space Models},
  author={Ali Ghodsi},
  journal={arXiv preprint arXiv:2512.15115v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15115v1}
}
```

## [Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](http://arxiv.org/abs/2512.15112v1)

#无监督节点表示学习#自适应图卷积#非同质性图

[PDF](https://arxiv.org/pdf/2512.15112v1)
[Abstract](http://arxiv.org/abs/2512.15112v1)

 图表示学习（GNN）

 很推荐

### 中文摘要

无监督节点表示学习旨在在不依赖节点标签的情况下获取有意义的节点嵌入。为此，通常采用图卷积来聚合邻居信息，以编码节点特征与图拓扑。然而，过度依赖图卷积在某些情形下并不理想——尤其是在非同质性（non-homophilic）图中，过多的邻居聚合可能导致在特征或拓扑上差异较大的节点被映射为过于相似的嵌入。因此，在有监督学习场景中对图卷积使用程度进行调节已被广泛探讨，但在无监督情形下相关方法仍较少。为了解决这一问题，我们提出了 FUEL，一种通过自适应学习图卷积使用程度的方法，目标是在嵌入空间中增强类内相似性并提高类间可分性。由于类标签未知，FUEL 利用节点特征进行聚类，将所得簇作为类的代理来引导学习。通过在 14 个基准数据集上与 15 种基线方法的广泛对比实验，实验证明 FUEL 在不同同质性水平的图上均能显著提升下游任务性能，并取得了最先进的结果。

BibTeX

```
@article{2512.15112v1,
  title={Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption},
  author={Sunwoo Kim and Soo Yong Lee and Kyungho Kim and Hyunjin Hwang and Jaemin Yoo and Kijung Shin},
  journal={arXiv preprint arXiv:2512.15112v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15112v1}
}
```

## [Beyond Fast and Slow: Cognitive-Inspired Elastic Reasoning for Large Language Models](http://arxiv.org/abs/2512.15089v1)

#弹性推理#强化学习策略选择#工具辅助链式思维

[PDF](https://arxiv.org/pdf/2512.15089v1)
[Abstract](http://arxiv.org/abs/2512.15089v1)

 LLM

 很推荐

### 中文摘要

大规模语言模型（LLM）在各类语言任务上表现出色，但现有的推理策略主要依赖模型自身的快/慢模式（如系统1/系统2思维），难以在面对不同难度的查询时在推理效率与准确性之间取得平衡。本文提出了受人类分层推理启发的弹性推理框架 CogER，该框架能够为每个查询动态选择最合适的推理策略。具体而言，CogER 首先评估输入查询的复杂度并将其分配到若干预定义等级中的一个，每个等级对应定制的处理策略，从而应对查询难度不可观测的问题。为实现自动化策略选择，我们将该过程建模为马尔可夫决策过程（MDP），并通过强化学习训练出 CogER-Agent，使用兼顾解答质量与计算成本的奖励函数以保证资源高效的推理。此外，对于需要借助外部工具的查询，我们提出了认知工具辅助推理机制，使 LLM 能在其思维链中自主调用外部工具。大量实验表明，CogER 优于现有的测试时扩展方法，在域内任务上的平均精确匹配率至少提升了 13%，在域外任务上获得了 8% 的相对增益。

BibTeX

```
@article{2512.15089v1,
  title={Beyond Fast and Slow: Cognitive-Inspired Elastic Reasoning for Large Language Models},
  author={Jinwu Hu and Dongjin Yang and Langyu Bian and Zhiquan Wen and Yufeng Wang and Yaofo Chen and Bin Xiao and Yuanqing Li and Mingkui Tan},
  journal={arXiv preprint arXiv:2512.15089v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15089v1}
}
```

## [Meta-learners for few-shot weakly-supervised optic disc and cup segmentation on fundus images](http://arxiv.org/abs/2512.15061v1)

#少样本弱监督#元学习#视盘/视杯分割

[PDF](https://arxiv.org/pdf/2512.15061v1)
[Abstract](http://arxiv.org/abs/2512.15061v1)

 CV

 很推荐

### 中文摘要

本研究为少样本弱监督分割（few-shot weakly-supervised segmentation，FWS）任务构建了元学习器，以应对在标注有限的眼底图像上进行青光眼诊断所需的视盘（optic disc, OD）与视杯（optic cup, OC）分割问题。我们通过引入 Omni 元训练（Omni meta-training）显著改进了现有元学习器，该方法在数据使用上实现平衡并增加了样本数（shots）的多样性。同时，我们开发了高效版本以降低计算开销。此外，我们提出了稀疏化技术，用于生成更具可定制性和代表性的涂鸦（scribbles）及其他稀疏标注。在多个数据集上的评估表明，Omni 及其高效版本均优于原始版本，其中表现最好的元学习器为 Efficient Omni ProtoSeg（EO-ProtoSeg）。在仅使用一张稀疏标注图像的情况下，EO-ProtoSeg 在 REFUGE 数据集上分别取得 OD IoU 88.15% 与 OC IoU 71.17%，优于那些需要更多标注图像的少样本及半监督方法。其在不同数据集上的最佳表现为：在 DRISHTIGS 上 OD IoU 86.80%、OC IoU 71.78%；在 REFUGE 上 OD IoU 88.21%、OC IoU 73.70%；在另一次报告的 REFUGE（或可能为其他数据集，文中重复列出）上 OD IoU 80.39%、OC IoU 52.65%。EO-ProtoSeg 与无监督域自适应方法的性能相当，但模型更轻量（参数少于两百万）且无需重新训练。

BibTeX

```
@article{2512.15061v1,
  title={Meta-learners for few-shot weakly-supervised optic disc and cup segmentation on fundus images},
  author={Pandega Abyan Zumarsyah and Igi Ardiyanto and Hanung Adi Nugroho},
  journal={arXiv preprint arXiv:2512.15061v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15061v1}
}
```

## [SGM: Safety Glasses for Multimodal Large Language Models via Neuron-Level Detoxification](http://arxiv.org/abs/2512.15052v1)

#多模态大模型#神经元级去毒#毒性控制

[PDF](https://arxiv.org/pdf/2512.15052v1)
[Abstract](http://arxiv.org/abs/2512.15052v1)

 LLM

 很推荐

### 中文摘要

免责声明：本文中的样本可能有害并引起不适。多模态大语言模型（MLLMs）能够进行多模态生成，但由于预训练语料筛选不严，继承了有毒、带偏见和不适当内容（NSFW）的信号，带来了安全风险，尤其在对抗性触发下，现有后期且不可解释的无训练去毒方法难以应对。我们提出了SGM，一种白盒的神经元级多模态干预方法，类似于为有害神经元戴上的“安全眼镜”：通过基于专业性的加权软抑制，有选择性地重标定一小组有毒专家神经元，在不更新模型参数的情况下中和有害的跨模态激活。我们构建了多模态毒性评测框架MM-TOXIC-QA，并将SGM与现有去毒技术进行了比较。在开源多模态大模型上的实验表明，SGM在标准与对抗情形下均能有效降低毒性，将有害率从48.2%降至2.5%，同时保持生成流畅性与多模态推理能力。SGM具有良好可扩展性，其组合防御形式SGM\*可与现有去毒方法集成以实现更强的安全性能，为可解释、低成本的毒性可控多模态生成提供了一种实用方案。

BibTeX

```
@article{2512.15052v1,
  title={SGM: Safety Glasses for Multimodal Large Language Models via Neuron-Level Detoxification},
  author={Hongbo Wang and MaungMaung AprilPyone and Isao Echizen},
  journal={arXiv preprint arXiv:2512.15052v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15052v1}
}
```

## [HERO: Hierarchical Traversable 3D Scene Graphs for Embodied Navigation Among Movable Obstacles](http://arxiv.org/abs/2512.15047v1)

#3D场景图#具身导航#可操作障碍

[PDF](https://arxiv.org/pdf/2512.15047v1)
[Abstract](http://arxiv.org/abs/2512.15047v1)

 Embodied Navigation

 很推荐

### 中文摘要

3D场景图（3DSG）是对物理世界的强大表示，能够显式建模实体之间复杂的空间、语义与功能关系，从而为智能体与环境交互并执行多样化行为提供基础理解。作为这些能力的关键组成部分，具身导航利用3DSG的紧凑而富表征性的特性，在复杂大尺度环境中实现长时程推理与规划。然而，先前工作依赖静态世界假设，仅基于静态空间布局来定义可通行区域，从而将可交互的障碍物当作不可通行处理。这一根本性限制严重削弱了其在实际场景中的有效性，导致可达性受限、效率低下且可扩展性差。为了解决这些问题，我们提出了HERO，一种用于构建层次化可通行3D场景图（Hierarchical Traversable 3DSG）的新框架。HERO通过将可操作的障碍物重新建模为“可作为通道的路径”，并同时捕捉其物理可交互性、功能语义以及场景的关系层次，重新定义了可通行性。实验结果表明，相较于基线方法，HERO在部分阻挡环境中将路径长度（PL）降低了35.1%，在完全阻挡环境中将成功率（SR）提升了79.4%，显示出显著更高的效率和可达性。

BibTeX

```
@article{2512.15047v1,
  title={HERO: Hierarchical Traversable 3D Scene Graphs for Embodied Navigation Among Movable Obstacles},
  author={Yunheng Wang and Yixiao Feng and Yuetong Fang and Shuning Zhang and Tan Jing and Jian Li and Xiangrui Jiang and Renjing Xu},
  journal={arXiv preprint arXiv:2512.15047v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15047v1}
}
```

## [Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent](http://arxiv.org/abs/2512.14990v1)

#深度学习调试#错误复现#LLM驱动自动化生成

[PDF](https://arxiv.org/pdf/2512.14990v1)
[Abstract](http://arxiv.org/abs/2512.14990v1)

 软件工程（深度学习错误复现与调试）

 很推荐

### 中文摘要

尽管深度学习（DL）在医疗、金融、软件工程等多个领域被广泛采用，但基于深度学习的应用仍存在大量错误、故障和脆弱性。复现这些错误对于修复至关重要，但由于深度学习模型的内在非确定性以及与硬件和软件环境的紧密耦合，复现过程极其困难。近期研究表明，使用人工方法仅能可靠复现约3%的深度学习错误。为了解决这些挑战，我们提出了 RepGen，一种新颖的、自动化且智能的深度学习错误复现方法。RepGen 从项目中构建学习增强的上下文信息，制定全面的错误复现计划，采用生成-验证-迭代改进的机制，并利用大型语言模型（LLM）生成可复现目标错误的代码。我们在106个真实世界的深度学习错误上评估了 RepGen，达到了80.19%的复现率，比现有最先进方法提升了19.81%。一项包含27名参与者的开发者研究表明，RepGen 将深度学习错误复现成功率提高了23.35%、复现所需时间缩短了56.8%，并降低了参与者的认知负担。

BibTeX

```
@article{2512.14990v1,
  title={Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent},
  author={Mehil B Shah and Mohammad Masudur Rahman and Foutse Khomh},
  journal={arXiv preprint arXiv:2512.14990v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14990v1}
}
```

## [TalkVerse: Democratizing Minute-Long Audio-Driven Video Generation](http://arxiv.org/abs/2512.14938v1)

#音频驱动视频生成#大规模开源语音-视频语料#分钟级长时低漂移生成

[PDF](https://arxiv.org/pdf/2512.14938v1)
[Abstract](http://arxiv.org/abs/2512.14938v1)

 CV (音频-视觉/多模态)

 很推荐

### 中文摘要

我们提出了 TalkVerse，一个大规模的开放语料库，面向单人音频驱动的说话视频生成，旨在促进方法间的公平与可复现比较。与当前依赖封闭数据或计算密集型模型的最先进系统不同，TalkVerse 提供了 230 万段高分辨率（720p/1080p）音视频同步剪辑，总计约 6.3k 小时。这些剪辑从超过 6 万小时的视频中通过透明的管道精心筛选，管道包括场景切分检测、美学评估、严格的音视频同步检测，以及包含 2D 骨架和结构化视觉/音频风格描述的全面标注。基于 TalkVerse，我们给出一个可复现的 5B DiT 基线（基于 Wan2.2-5B）。通过采用高降采样比的视频 VAE 和带有运动帧上下文的滑动窗口机制，模型能够实现分钟级生成并控制漂移。该模型在口型同步和视觉质量上可与 14B 的 Wan-S2V 相媲美，但推理成本低 10×。为增强长视频的故事性，我们集成了用于根据音频和视觉线索重写提示的 MLLM 导演模块。此外，模型通过受控的潜在噪声注入支持零样本视频配音（dubbing）。我们开源了数据集、训练流程和 5B 检查点，旨在降低音频驱动人类视频生成研究的门槛。

BibTeX

```
@article{2512.14938v1,
  title={TalkVerse: Democratizing Minute-Long Audio-Driven Video Generation},
  author={Zhenzhi Wang and Jian Wang and Ke Ma and Dahua Lin and Bing Zhou},
  journal={arXiv preprint arXiv:2512.14938v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14938v1}
}
```

## [Entropy-Reservoir Bregman Projection: An Information-Geometric Unification of Model Collapse](http://arxiv.org/abs/2512.14879v1)

#模型崩溃#信息几何#熵储备

[PDF](https://arxiv.org/pdf/2512.14879v1)
[Abstract](http://arxiv.org/abs/2512.14879v1)

 生成模型理论（LLM / GAN / RL）

 很推荐

### 中文摘要

自我参照学习——即用模型自身生成的数据训练模型——尽管在可扩展性上有巨大吸引力，但普遍遭遇模型崩溃问题：语言模型退化为重复文本，GAN 丢失模式，强化学习策略过度利用某些行为。尽管实务中常用混合真实数据、熵奖励、知识蒸馏或检索增强生成等经验性修复方法，这些方法为何有效及崩溃的本质尚缺乏统一原理。我们提出熵储备 Bregman 投影（Entropy-Reservoir Bregman Projection, ERBP），一个基于信息几何的框架来统一解释上述现象。我们将闭环自学习建模为分布空间中的随机 Bregman 投影序列：在没有外部耦合的情况下，有限样本噪声会迫使系统投影到不断收缩的经验支撑上，造成熵的指数级衰减并最终崩溃。通过引入熵储备——在每次投影中混入一个高熵分布——注入可控的熵通量，可证明地稳定系统动力学。理论结果给出：崩溃的必要条件、保证非平凡熵下界的充分条件，以及仅依赖样本量和 Bregman 生成元的强凸性/ Lipschitz 常数的闭式速率。对大语言模型自训练、Soft Actor-Critic 强化学习和 GAN 优化的实验验证了我们的预测，并表明各类稳定化启发式方法对应于特定的熵储备选择和耦合系数。ERBP 因此将一系列经验性修复方法统一为单一、可量化的设计准则：监测并预算你的熵通量。

BibTeX

```
@article{2512.14879v1,
  title={Entropy-Reservoir Bregman Projection: An Information-Geometric Unification of Model Collapse},
  author={Jingwei Chen},
  journal={arXiv preprint arXiv:2512.14879v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14879v1}
}
```

## [Sharing State Between Prompts and Programs](http://arxiv.org/abs/2512.14805v1)

#自然语言编程#共享程序状态#提示与程序互操作

[PDF](https://arxiv.org/pdf/2512.14805v1)
[Abstract](http://arxiv.org/abs/2512.14805v1)

 LLM

 很推荐

### 中文摘要

大型语言模型（LLM）的兴起催生了一种新的编程范式：自然语言编程。通过编写引导 LLM 执行自然语言处理、代码生成、推理等任务的提示，用户实际上是在以自然语言编写供 LLM 执行的“自然语言代码”。近年来，研究开始探讨自然语言代码与诸如 Python 等形式化语言之间的互操作性。我们提出了一种新颖的编程抽象——共享程序状态（shared program state），它消除了实现自然语言代码与程序状态互操作所需的手工工作。借助共享程序状态，程序员可以编写直接写入程序变量、对程序对象进行计算并在程序中实现控制流的自然语言代码。我们还提出了一种用于指定自然函数接口的模式(schema)，将编程系统扩展为支持自然语言代码，并基于该模式将共享程序状态定义为一种自然函数接口。我们在 Nightjar 编程系统中实现了共享程序状态。Nightjar 使程序员能够在 Python 程序中嵌入共享 Python 程序状态的自然语言代码。实验表明，Nightjar 程序在任务准确率上与手写实现相当或更高（提高 4%–19%），同时平均减少代码行数 39.6%。使用 Nightjar 的代价是可能产生运行时开销（运行时间为手写实现的约 0.4–4.3 倍）。

BibTeX

```
@article{2512.14805v1,
  title={Sharing State Between Prompts and Programs},
  author={Ellie Y. Cheng and Logan Weber and Tian Jin and Michael Carbin},
  journal={arXiv preprint arXiv:2512.14805v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14805v1}
}
```

## [Magnification-Aware Distillation (MAD): A Self-Supervised Framework for Unified Representation Learning in Gigapixel Whole-Slide Images](http://arxiv.org/abs/2512.14796v1)

#放大感知蒸馏#多尺度自监督学习#全切片图像表示学习

[PDF](https://arxiv.org/pdf/2512.14796v1)
[Abstract](http://arxiv.org/abs/2512.14796v1)

 CV（医学图像/计算病理学）

 很推荐

### 中文摘要

全切片图像（WSI）在不同放大倍数上承载着分布于多尺度的组织信息，但大多数自监督方法将这些尺度视为独立视角，导致模型难以学到在分辨率变化时仍然稳定的表征，而这正是实际神经病理学工作流所需的关键能力。本研究提出了放大感知蒸馏（Magnification-Aware Distillation，MAD），这是一种自监督策略，通过将低倍的整体语境与空间对齐的高倍细节关联起来，使模型能够学习粗略组织结构与精细细胞模式之间的对应关系。由此得到的基础模型MAD-NP完全通过这种跨尺度对应进行训练，无需任何标注。实验显示，仅在10x嵌入上训练的线性分类器在应用于未见的40x切片时仍保持了96.7%的性能，表明其具备强的分辨率不变表征能力。分割结果在不同放大倍数间保持一致，能保存解剖边界并最小化噪声。上述结果突显了利用统一嵌入空间实现可扩展且对放大倍数稳健的WSI分析的可行性。

BibTeX

```
@article{2512.14796v1,
  title={Magnification-Aware Distillation (MAD): A Self-Supervised Framework for Unified Representation Learning in Gigapixel Whole-Slide Images},
  author={Mahmut S. Gokmen and Mitchell A. Klusty and Peter T. Nelson and Allison M. Neltner and Sen-Ching Samson Cheung and Thomas M. Pearce and David A Gutman and Brittany N. Dugger and Devavrat S. Bisht and Margaret E. Flanagan and V. K. Cody Bumgardner},
  journal={arXiv preprint arXiv:2512.14796v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14796v1}
}
```

## [IaC Generation with LLMs: An Error Taxonomy and A Study on Configuration Knowledge Injection](http://arxiv.org/abs/2512.14792v1)

#基础设施即代码#大型语言模型#知识注入（RAG/Graph RAG）

[PDF](https://arxiv.org/pdf/2512.14792v1)
[Abstract](http://arxiv.org/abs/2512.14792v1)

 LLM

 很推荐

### 中文摘要

大型语言模型（LLM）在生成正确且与用户意图一致的基础设施即代码（IaC）方面目前成功率较低。本研究针对 Terraform 的场景，通过系统性地注入结构化配置知识来改进基于 LLM 的 IaC 生成。为此，研究对现有的 IaC-Eval 基准进行了显著增强，加入了云环境模拟与自动化错误分析，并提出了一种用于 LLM 辅助 IaC 代码生成的新型错误分类法。研究实现并评估了一系列知识注入技术，方法上从朴素的检索增强生成（Naive RAG）逐步发展到更复杂的图检索增强生成（Graph RAG），包括对图结构组件的语义增强和跨资源依赖建模。实验结果显示，尽管基线 LLM 的总体成功率较低（27.1%），引入结构化配置知识后技术验证成功率提升至 75.3%，总体成功率提升至 62.6%。然而，尽管技术正确性显著提高，意图对齐并未同步改善，揭示了“正确性—契合度差距”（Correctness-Congruence Gap）：LLM 在成为熟练“编码者”方面进步显著，但在满足细化用户架构意图方面仍受限。

BibTeX

```
@article{2512.14792v1,
  title={IaC Generation with LLMs: An Error Taxonomy and A Study on Configuration Knowledge Injection},
  author={Roman Nekrasov and Stefano Fossati and Indika Kumara and Damian Andrew Tamburri and Willem-Jan van den Heuvel},
  journal={arXiv preprint arXiv:2512.14792v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14792v1}
}
```

## [GR-Agent: Adaptive Graph Reasoning Agent under Incomplete Knowledge](http://arxiv.org/abs/2512.14766v1)

#知识图谱问答#图推理代理#不完整知识图谱

[PDF](https://arxiv.org/pdf/2512.14766v1)
[Abstract](http://arxiv.org/abs/2512.14766v1)

 NLP

 很推荐

### 中文摘要

大型语言模型（LLMs）在知识图谱问答（KGQA）任务上取得了不错的成绩，但大多数基准假设知识图谱（KG）是完备的，即存在直接的支持三元组。这使得评测退化为表层检索，忽视了现实中知识图谱常常不完整、许多事实缺失且必须基于现有事实进行推理才能得到答案的情况。为弥补这一差距，我们提出了一种在KG不完备情形下构建基准的方法学：在移除直接支持三元组的同时，确保仍存在可用于推断答案的替代推理路径。基于该方法学构建的基准实验表明，现有方法在不完备情况下普遍出现性能下降，突出了它们在推理能力方面的局限性。为克服该限制，我们提出了自适应图推理代理（GR-Agent）。该方法首先从知识图谱构建一个交互式环境，并将KGQA形式化为代理与环境的交互过程；GR-Agent 在由图推理工具组成的动作空间中操作，并维护一段潜在支持性推理证据的记忆，包含相关关系和推理路径。大量实验证明，GR-Agent 在完备与不完备两种设置下均优于非训练型基线，并能与基于训练的方法达到相当的性能。

BibTeX

```
@article{2512.14766v1,
  title={GR-Agent: Adaptive Graph Reasoning Agent under Incomplete Knowledge},
  author={Dongzhuoran Zhou and Yuqicheng Zhu and Xiaxia Wang and Hongkuan Zhou and Jiaoyan Chen and Steffen Staab and Yuan He and Evgeny Kharlamov},
  journal={arXiv preprint arXiv:2512.14766v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14766v1}
}
```

## [Workflows vs Agents for Code Translation](http://arxiv.org/abs/2512.14762v1)

#代码翻译#代理式框架#语法修复

[PDF](https://arxiv.org/pdf/2512.14762v1)
[Abstract](http://arxiv.org/abs/2512.14762v1)

 LLM

 很推荐

### 中文摘要

将诸如 MATLAB 之类的高级语言算法翻译为硬件描述语言（HDL）是部署到 FPGA 和 ASIC 上所必需但资源密集的步骤。尽管大型语言模型（LLM）为自动化提供了途径，但由于在 HDL 代码上的训练不足，端到端的转译容易脆弱并产生语法错误。我们比较了在 MATLAB→HDL 流水线中用于语法修复的两种基于 LLM 的方法：一种是结构化的、由专家设计的固定操作序列流程，另一种是更自主的代理式方法，利用 Model Context Protocol (MCP) 动态选择工具。我们在 42 个 MATLAB 信号处理函数上研究并将关注点限定在语法修复阶段。在三个模型规模下，代理式方法在解决初始语法错误方面更为有效，使更多候选项得以继续通过流水线。该上游改进带来了可测的下游收益，尤其是在中等规模模型上，使仿真可达率提升了 20 个百分点以上。我们假设这些增益来源于简短提示、积极的上下文管理和有条件的工具使用。条件检索在 8B 和 30B 规模上有帮助；在 235B 时最终成功的增益较小，且一种简单的 RAG 变体取得了最高的最终成功率。我们的发现表明，经过良好设计的代理式框架在弥补小型和中型模型容量限制方面最为有效。

BibTeX

```
@article{2512.14762v1,
  title={Workflows vs Agents for Code Translation},
  author={Henry Gray and Tom Yotam and Octavian Udrea},
  journal={arXiv preprint arXiv:2512.14762v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14762v1}
}
```

## [Quantum-Augmented AI/ML for O-RAN: Hierarchical Threat Detection with Synergistic Intelligence and Interpretability (Technical Report)](http://arxiv.org/abs/2512.14742v1)

#量子增强机器学习#O-RAN网络安全#分层威胁检测与可解释性

[PDF](https://arxiv.org/pdf/2512.14742v1)
[Abstract](http://arxiv.org/abs/2512.14742v1)

 通信网络安全（O-RAN）

 很推荐

### 中文摘要

开放无线接入网（O-RAN）虽然提高了模块化和遥测粒度，但也在分散的控制、用户和管理平面上扩大了网络安全攻击面。我们提出了一种与O-RAN遥测栈相匹配的分层防御框架，包含三层协同工作模块：异常检测、入侵确认和多攻击分类。该方法融合了混合量子计算与机器学习，采用基于振幅与纠缠的特征编码，并结合深度与集成分类器进行判别。我们在合成与真实遥测数据上进行了广泛基准测试，评估了编码深度、架构变体及诊断保真度。实验结果显示该框架在准确率、召回率和类别可分性上表现接近完美。通过对决策边界、概率余量和潜在空间几何的多维评估，验证了方法的可解释性、鲁棒性，以及在面向切片的诊断场景中向近实时（near-RT）和非实时（non-RT）RIC域可扩展部署的准备度。

BibTeX

```
@article{2512.14742v1,
  title={Quantum-Augmented AI/ML for O-RAN: Hierarchical Threat Detection with Synergistic Intelligence and Interpretability (Technical Report)},
  author={Tan Le and Van Le and Sachin Shetty},
  journal={arXiv preprint arXiv:2512.14742v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14742v1}
}
```

## [How a Bit Becomes a Story: Semantic Steering via Differentiable Fault Injection](http://arxiv.org/abs/2512.14715v1)

#位级故障注入#语义敏感性估计#视觉-语言模型鲁棒性

[PDF](https://arxiv.org/pdf/2512.14715v1)
[Abstract](http://arxiv.org/abs/2512.14715v1)

 LLM（视觉-语言）

 很推荐

### 中文摘要

难以检测的硬件位翻转（来自恶意电路或软件缺陷）已被证明会使变换器在非生成任务中变得脆弱。本工作首次研究对用于图像描述的大型语言模型（LLM）权重进行低级别、逐位扰动（故障注入）如何在保持语法结构和流畅性的同时改变其生成描述的语义含义。先前的故障分析方法表明，翻转少量比特可以使分类器崩溃或降低准确率，但这些方法忽略了生成系统的语义和语言维度。在图像描述模型中，单个位的翻转可能微妙地改变视觉特征到词语的映射，从而整体上改变模型对世界叙述的语义走向。我们假设这种语义漂移并非随机可变，而是可以通过可微分方式估计：即模型自身的梯度能够预测哪些比特在被扰动时会最强烈地影响意义，同时不破坏句法和流畅性。为此我们设计了一个可微分故障分析框架 BLADE（Bit-level Fault Analysis via Differentiable Estimation），该框架使用基于梯度的敏感性估计来定位语义关键比特，并通过基于整句描述语义与流畅性的目标进一步精细选择这些比特。我们的目标不仅是破坏描述质量，而是理解意义如何在比特级别上被编码、分布与可控，从而揭示即便不可察觉的低级别改变也能引导生成视觉-语言模型的高层语义输出。该工作亦为鲁棒性测试、对抗防御与可解释人工智能开辟了路径，展示了结构化位级故障如何重塑模型的语义输出。

BibTeX

```
@article{2512.14715v1,
  title={How a Bit Becomes a Story: Semantic Steering via Differentiable Fault Injection},
  author={Zafaryab Haider and Md Hafizur Rahman and Shane Moeykens and Vijay Devabhaktuni and Prabuddha Chakraborty},
  journal={arXiv preprint arXiv:2512.14715v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14715v1}
}
```

## [Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants](http://arxiv.org/abs/2512.15712v1)

#模型可解释性#概念瓶颈解码器#端到端可扩展性

[PDF](https://arxiv.org/pdf/2512.15712v1)
[Abstract](http://arxiv.org/abs/2512.15712v1)

 LLM

 很推荐

### 中文摘要

解释神经网络内部激活可以生成对其行为更为忠实的说明，但由于激活空间结构复杂，这一任务非常困难。现有可扩展的可解释性方法通常依赖人工设计的代理，通过提出并检验假设来把内部激活与外部行为联系起来。我们提出将该任务转化为端到端训练目标：训练可解释性助手，使其通过通信瓶颈从激活中准确预测模型行为。具体而言，编码器将激活压缩为一份稀疏的概念列表，解码器读取该列表并以自然语言回答关于模型的问题。我们展示了如何在大规模非结构化数据上对该助手进行预训练，然后对其进行微调以回答问题。我们称这种架构为“预测概念解码器（Predictive Concept Decoder，PCD）”，其具有良好的扩展性：瓶颈概念的自动解释评分（auto-interp score）随数据量增加而提升，下游任务的表现亦随之改善。具体应用上，PCD 可用于检测越狱（jailbreaks）、隐含提示（secret hints）和植入的潜在概念，并能够准确揭示潜在的用户属性。

BibTeX

```
@article{2512.15712v1,
  title={Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants},
  author={Vincent Huang and Dami Choi and Daniel D. Johnson and Sarah Schwettmann and Jacob Steinhardt},
  journal={arXiv preprint arXiv:2512.15712v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15712v1}
}
```

## [Where is the Watermark? Interpretable Watermark Detection at the Block Level](http://arxiv.org/abs/2512.14994v1)

#图像水印#可解释性检测#离散小波变换

[PDF](https://arxiv.org/pdf/2512.14994v1)
[Abstract](http://arxiv.org/abs/2512.14994v1)

 CV

 很推荐

### 中文摘要

随着生成式人工智能的进步，能够生成高度逼真的数字内容，进而引发了关于真实性、所有权与滥用的担忧。尽管水印技术已成为追踪与保护数字媒体的重要手段，但现有大多数图像水印方案以黑箱形式运行，仅给出全局检测分数，无法说明水印在图像中的位置或如何存在，这种缺乏透明性的做法影响了用户信任并使篡改影响难以解释。本文提出了一种事后（post-hoc）图像水印方法，结合了局部嵌入与区域级可解释性。我们在离散小波变换域中采用统计的块级策略嵌入水印信号，从而能够生成检测图，揭示图像中哪些区域可能被加水印或被篡改。实验表明，本方法在面对常见图像变换时具有较强的鲁棒性，同时对语义性修改保持敏感，并且水印具有高度不可察觉性。与先前的事后方法相比，我们的方法在保持竞争性鲁棒性的同时提供了更可解释的检测结果，例如在裁剪达到半幅图像时仍能保持鲁棒性。

BibTeX

```
@article{2512.14994v1,
  title={Where is the Watermark? Interpretable Watermark Detection at the Block Level},
  author={Maria Bulychev and Neil G. Marchant and Benjamin I. P. Rubinstein},
  journal={arXiv preprint arXiv:2512.14994v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14994v1}
}
```

## [Automated Motion Artifact Check for MRI (AutoMAC-MRI): An Interpretable Framework for Motion Artifact Detection and Severity Assessment](http://arxiv.org/abs/2512.15315v1)

#运动伪影检测#可解释性#对比学习

[PDF](https://arxiv.org/pdf/2512.15315v1)
[Abstract](http://arxiv.org/abs/2512.15315v1)

 CV

 很推荐

### 中文摘要

运动伪影会降低MRI图像质量并导致病人重扫。现有的自动质量评估方法大多仅作二分类判断且缺乏可解释性。我们提出AutoMAC-MRI，一种可解释的框架，用于在异质的MR对比度和视角下对运动伪影进行分级评估。该方法采用有监督对比学习来学习对运动严重程度具有区分性的表征。在该特征空间内，我们为每一等级计算特异性的亲和度分数，用以量化图像与各运动等级的接近程度，从而使等级判定过程透明且具有可解释性。我们在超过5000张由专家标注的脑部MRI切片（涵盖多种对比度和视图）上对AutoMAC-MRI进行了评估。实验证明，亲和度分数与专家标签高度一致，支持其作为可解释的运动严重性度量。通过将精确的等级检测与逐级亲和度评分相结合，AutoMAC-MRI 可用于实时的MRI质量控制，有望减少不必要的重扫并提升工作流程效率。

BibTeX

```
@article{2512.15315v1,
  title={Automated Motion Artifact Check for MRI (AutoMAC-MRI): An Interpretable Framework for Motion Artifact Detection and Severity Assessment},
  author={Antony Jerald and Dattesh Shanbhag and Sudhanya Chatterjee},
  journal={arXiv preprint arXiv:2512.15315v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15315v1}
}
```

## [Attention in Motion: Secure Platooning via Transformer-based Misbehavior Detection](http://arxiv.org/abs/2512.15503v1)

#车队编队安全#Transformer 异常检测#边缘实时推理

[PDF](https://arxiv.org/pdf/2512.15503v1)
[Abstract](http://arxiv.org/abs/2512.15503v1)

 车联网安全（智能驾驶/多车协同）

 很推荐

### 中文摘要

车队编队通过车辆到一切（V2X）通信实现多车队形的协同，有望显著提升交通效率与行车安全。然而，编队协调的分布式特性带来了安全脆弱性：即便是已认证的车辆也可能注入伪造的运动学数据，从而破坏运行稳定性并威胁乘员安全。传统的错误行为检测方法依赖合理性检验与统计学手段，存在较高的误报率，并且难以捕捉多车协同动态中固有的复杂时序依赖性。为此，我们提出了Attention In Motion（AIMformer），一种专为车队编队实时错误行为检测并支持边缘部署的 Transformer 架构。AIMformer 利用多头自注意力机制同时建模车内时序动态与车间空间相关性，并通过结合全局位置编码与车辆特定的时间偏移来处理车辆加入/退出的机动情况。我们提出了一种注重精确率的二元交叉熵（BCE）损失函数，以对误报施加更强惩罚，满足安全关键车辆系统的严格要求。在涵盖4种编队控制器、多种攻击向量与多样移动场景的广泛评估中，AIMformer 相较于现有先进基线模型表现优异（检测性能 ≥ 0.93）。此外，我们通过 TensorFlow Lite、ONNX 与 TensorRT 的全面部署分析实现了亚毫秒级推理延迟，证明其适用于资源受限的边缘平台，从而验证了 AIMformer 在车载与路侧基础设施中的可行性。

BibTeX

```
@article{2512.15503v1,
  title={Attention in Motion: Secure Platooning via Transformer-based Misbehavior Detection},
  author={Konstantinos Kalogiannis and Ahmed Mohamed Hussain and Hexu Li and Panos Papadimitratos},
  journal={arXiv preprint arXiv:2512.15503v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15503v1}
}
```

## [EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving](http://arxiv.org/abs/2512.14946v1)

#KV缓存管理#有损压缩#分层存储驱逐策略

[PDF](https://arxiv.org/pdf/2512.14946v1)
[Abstract](http://arxiv.org/abs/2512.14946v1)

 LLM

 很推荐

### 中文摘要

在大规模语言模型（LLM）推理系统中，复用 KV 缓存对提高效率至关重要。随着用户数量增加，KV 缓存占用很容易超过 GPU 内存容量，因而已有工作提出将 KV 缓存驱逐到低层存储或对其进行压缩以便在快速内存中容纳更多缓存。然而，先前工作忽略了一个关键机会：跨所有 KV 缓存联合优化驱逐与压缩决策，以在不损害生成质量的前提下最小化平均生成延迟。
我们提出 EVICPRESS，一种对跨多层存储的 KV 缓存同时应用有损压缩与自适应驱逐的管理系统。具体地，针对每个上下文的 KV 缓存，EVICPRESS 考虑该缓存的压缩与驱逐操作对所有上下文整体平均生成质量与延迟的影响。为此，EVICPRESS 设计了一个统一的效用函数，用以量化有损压缩或驱逐对质量与延迟的综合影响。EVICPRESS 的性能剖面模块会周期性地更新各上下文在所有可能驱逐-压缩配置下的效用得分，并采用快速启发式算法在各存储层间重新放置 KV 缓存，目标是在每一存储层上最大化效用得分。与仅驱逐或仅压缩的基线方法相比，EVICPRESS 在快速设备上实现了更高的 KV 缓存命中率（即更低的延迟），同时通过对对压缩敏感的上下文采取保守压缩策略来保持较高的生成质量。在 12 个数据集和 5 个模型上的评测表明，EVICPRESS 在等效生成质量下可将首个 Token 到达时间（TTFT）加速最多达 2.19 倍。

BibTeX

```
@article{2512.14946v1,
  title={EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving},
  author={Shaoting Feng and Yuhan Liu and Hanchen Li and Xiaokun Chen and Samuel Shen and Kuntai Du and Zhuohan Gu and Rui Zhang and Yuyang Huang and Yihua Cheng and Jiayi Yao and Qizheng Zhang and Ganesh Ananthanarayanan and Junchen Jiang},
  journal={arXiv preprint arXiv:2512.14946v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14946v1}
}
```

## [AgroAskAI: A Multi-Agentic AI Framework for Supporting Smallholder Farmers' Enquiries Globally](http://arxiv.org/abs/2512.14910v1)

#多智能体推理#农业气候适应决策支持#治理与防幻觉机制

[PDF](https://arxiv.org/pdf/2512.14910v1)
[Abstract](http://arxiv.org/abs/2512.14910v1)

 多智能体系统（Agentic AI）

 很推荐

### 中文摘要

农村农业地区面临由气候引发的风险损害，包括干旱、强降雨及气候模式变化等。既有研究呼吁提出适应性风险管理解决方案与决策支持策略。为此，人工智能（尤其是具代理性的Agentic AI）提供了有前景的路径。具代理性的AI系统由能够解决复杂动态任务的自治且专业化的智能体构成。以往系统多依赖单一智能体模型或仅在静态功能上采用多智能体框架，但实际需要能够支持动态协同推理与上下文感知输出的架构。为弥补这一缺口，我们提出了AgroAskAI——一个面向农业气候适应决策支持的多智能体推理系统，重点服务脆弱的农村社区。AgroAskAI具有模块化、角色专属的架构，采用责任链（chain-of-responsibility）方法协调自治智能体，并整合实时工具与数据集。系统内置治理机制以减轻幻觉风险，并通过内部反馈机制生成连贯且具地方相关性的策略。系统还支持多语言交互，便于非英语农户使用。在与气候适应相关的常见农业查询实验中，结合额外工具和提示精化后，AgroAskAI能够产出更具可操作性、基于事实且包容性的答复。我们的实验结果凸显了具代理性AI在促进农业气候适应方面实现可持续且负责任决策支持的潜力。

BibTeX

```
@article{2512.14910v1,
  title={AgroAskAI: A Multi-Agentic AI Framework for Supporting Smallholder Farmers' Enquiries Globally},
  author={Nadine Angela Cantonjos and Arpita Biswas},
  journal={arXiv preprint arXiv:2512.14910v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14910v1}
}
```

## [Artificial Intelligence for the Assessment of Peritoneal Carcinosis during Diagnostic Laparoscopy for Advanced Ovarian Cancer](http://arxiv.org/abs/2512.14797v1)

#腹膜癌变评估#腹腔镜视频计算机视觉#深度学习术中决策支持

[PDF](https://arxiv.org/pdf/2512.14797v1)
[Abstract](http://arxiv.org/abs/2512.14797v1)

 CV（医学影像/手术辅助）

 很推荐

### 中文摘要

晚期卵巢癌（AOC）常在已出现腹膜癌变（PC）时被确诊。诊断性腹腔镜（DL）下的Fagotti评分（FS）评估通过估计手术可切除性来指导治疗决策，但其主观性和对操作人员依赖性限制了可重复性和推广应用。研究回顾性收集了在一个转诊中心行DL并同时记录FS评估的手术视频，数据被划分为用于标注、AI训练与评估的发展数据集和用于内部验证的独立测试集。在发展数据集中，对与FS相关的帧进行了解剖结构与PC的人工标注。基于此训练深度学习模型以自动识别FS相关帧、分割解剖结构与PC，并在视频级别预测FS及手术指征（ItS）。AI性能以分割的Dice系数、解剖分区（AS）与ItS预测的F1分数以及最终FS估计的均方根误差（RMSE）进行评估。在发展数据集中，基于7,311帧训练的分割模型在解剖结构上实现了70±3%的Dice，在PC分割上实现了56±3%的Dice。视频级别的AS分类F1分数为74±3%和73±4%，FS预测的归一化RMSE分别为1.39±0.18和1.15±0.08，ItS在发展集（n=101）和独立测试集（n=50）上的F1分数均达到了80±8%和80±2%。该工作是第一个能够从DL视频自动估计FS并预测细胞减灭手术可行性的AI模型。其在不同数据集间表现出的可重复性和可靠性表明，AI可通过标准化的术中肿瘤负荷评估为外科医生提供支持，从而辅助晚期卵巢癌的临床决策。

BibTeX

```
@article{2512.14797v1,
  title={Artificial Intelligence for the Assessment of Peritoneal Carcinosis during Diagnostic Laparoscopy for Advanced Ovarian Cancer},
  author={Riccardo Oliva and Farahdiba Zarin and Alice Zampolini Faustini and Armine Vardazaryan and Andrea Rosati and Vinkle Srivastav and Nunzia Del Villano and Jacques Marescaux and Giovanni Scambia and Pietro Mascagni and Nicolas Padoy and Anna Fagotti},
  journal={arXiv preprint arXiv:2512.14797v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14797v1}
}
```

## [Improving VQA Reliability: A Dual-Assessment Approach with Self-Reflection and Cross-Model Verification](http://arxiv.org/abs/2512.14770v1)

#视觉问答#不确定性估计#跨模型验证

[PDF](https://arxiv.org/pdf/2512.14770v1)
[Abstract](http://arxiv.org/abs/2512.14770v1)

 VQA

 很推荐

### 中文摘要

视觉-语言模型（VLM）在视觉问答（VQA）任务中展现出显著潜力，但其易受“幻觉”影响，可能导致模型对错误答案过度自信，从而严重削弱回答的可靠性。为此，本文提出了用于VLM可靠性评估的双重评估框架（Dual-Assessment for VLM Reliability，DAVR），该框架通过自我反思（Self-Reflection）与跨模型验证（Cross-Model Verification）相结合，实现对不确定性的全面估计。DAVR采用双通路架构：一条通路利用双选择器模块（dual selector modules），将VLM的潜在特征与问答嵌入融合，用以评估响应的可靠性；另一条通路则使用外部参考模型进行事实性交叉核验，以抑制幻觉现象。在ICCV-CLVL 2025的Reliable VQA Challenge评测中，DAVR取得了领先的Φ\_{100}得分39.64和100-AUC为97.22，获得第一名，验证了其在提高VLM回答可信度方面的有效性。

BibTeX

```
@article{2512.14770v1,
  title={Improving VQA Reliability: A Dual-Assessment Approach with Self-Reflection and Cross-Model Verification},
  author={Xixian Wu and Yang Ou and Pengchao Tian and Zian Yang and Jielei Zhang and Peiyi Li and Longwen Gao},
  journal={arXiv preprint arXiv:2512.14770v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14770v1}
}
```

## [One Leak Away: How Pretrained Model Exposure Amplifies Jailbreak Risks in Finetuned LLMs](http://arxiv.org/abs/2512.14751v1)

#预训练-微调安全#越狱攻击迁移#表示级探测与PGP攻击

[PDF](https://arxiv.org/pdf/2512.14751v1)
[Abstract](http://arxiv.org/abs/2512.14751v1)

 LLM

 很推荐

### 中文摘要

微调预训练大规模语言模型（LLM）已成为开发下游应用的标准范式，但其安全性影响尚不清楚，特别是微调后的模型是否会继承其预训练源模型的越狱（jailbreak）脆弱性。我们在一个现实的“从预训练到微调”的威胁模型中进行研究：攻击者对白盒可访问预训练模型，而仅对其微调衍生模型进行黑盒访问。实证分析表明，在预训练模型上优化得到的对抗性提示（adversarial prompts）能够最有效地迁移到其微调变体，揭示了从预训练模型到微调模型的脆弱性继承现象。为进一步考察这种继承，我们进行了表示层级的探测，结果显示可迁移的提示在预训练隐藏表征中是线性可分的，这表明普遍的可迁移性被编码在预训练表征中。基于此洞见，我们提出了Probe-Guided Projection（PGP）攻击，通过引导优化朝向与迁移性相关的方向来提升迁移效果。在多个LLM家族和多样化微调任务上的实验验证了PGP在迁移成功率上的强大表现，强调了预训练到微调范式中固有的安全风险。

BibTeX

```
@article{2512.14751v1,
  title={One Leak Away: How Pretrained Model Exposure Amplifies Jailbreak Risks in Finetuned LLMs},
  author={Yixin Tan and Zhe Yu and Jun Sakuma},
  journal={arXiv preprint arXiv:2512.14751v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14751v1}
}
```

## [VERAFI: Verified Agentic Financial Intelligence through Neurosymbolic Policy Generation](http://arxiv.org/abs/2512.14744v1)

#金融AI#神经符号策略#检索增强生成（RAG）

[PDF](https://arxiv.org/pdf/2512.14744v1)
[Abstract](http://arxiv.org/abs/2512.14744v1)

 NLP（金融AI / Agentic Systems）

 很推荐

### 中文摘要

金融人工智能系统存在一个关键盲点：尽管检索增强生成（RAG）在查找相关文档方面表现优异，但即便检索完美，语言模型在推理过程中仍然会产生计算错误和合规违规问题。本文提出了VERAFI（Verified Agentic Financial Intelligence），一种通过神经符号策略生成实现经验证的智能金融代理框架。VERAFI 将最先进的稠密检索与交叉编码器重排相结合，配合支持金融工具的代理以及覆盖 GAAP 合规、SEC 要求和数学校验的自动化推理策略。我们在 FinanceBench 上的全面评估表明显著改进：传统的稠密检索加重排仅能达到 52.4% 的事实正确率，而 VERAFI 的集成方法达到了 94.7%，相对提升 81%。神经符号策略层单独相比纯代理处理贡献了 4.3 个百分点的提升，专门针对持续存在的数学和逻辑错误。通过将金融领域专长直接融入推理过程，VERAFI 为满足监管合规、投资决策和风险管理等严格准确性要求的可信金融 AI 提供了切实可行的路径。

BibTeX

```
@article{2512.14744v1,
  title={VERAFI: Verified Agentic Financial Intelligence through Neurosymbolic Policy Generation},
  author={Adewale Akinfaderin and Shreyas Subramanian},
  journal={arXiv preprint arXiv:2512.14744v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14744v1}
}
```

## [Semantic Geometry for policy-constrained interpretation](http://arxiv.org/abs/2512.14731v1)

#语义几何#策略约束#幻觉防控

[PDF](https://arxiv.org/pdf/2512.14731v1)
[Abstract](http://arxiv.org/abs/2512.14731v1)

 LLM

 很推荐

### 中文摘要

我们提出了一个用于策略约束语义解释的几何框架，能够在高风险领域以可证明的方式防止出现幻觉性承诺（hallucinated commitments）。在该框架中，语义含义被表示为单位球面上的方向，证据被建模为见证向量的集合，可接受的解释对应于球面上的凸区域。策略约束作为显式先验被引入并定义在相同流形上，与证据的几何结构分离。解释过程被归结为对可接受区域上的约束优化，当出现矛盾或策略排除时，拒绝（refusal）成为一个拓扑上必然的结果。我们将该框架与信息论、贝叶斯推断和层论（sheaf-theoretic）语义学相联系，并证明了所给复杂度界是信息论上最优的。在大规模受监管的金融数据上的实证验证表明，在多种策略机制下实现了零幻觉性批准——这是首个在大规模设置中达到此类结果的工作。

BibTeX

```
@article{2512.14731v1,
  title={Semantic Geometry for policy-constrained interpretation},
  author={Nikit Phadke},
  journal={arXiv preprint arXiv:2512.14731v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14731v1}
}
```

## [Generative Urban Flow Modeling: From Geometry to Airflow with Graph Diffusion](http://arxiv.org/abs/2512.14725v1)

#图扩散生成模型#城市风场模拟#图神经网络

[PDF](https://arxiv.org/pdf/2512.14725v1)
[Abstract](http://arxiv.org/abs/2512.14725v1)

 生成模型（科学计算/物理仿真）

 很推荐

### 中文摘要

城市风场建模与模拟在空气质量评估和可持续城市规划中具有重要作用。建模与模拟的一大关键挑战在于处理复杂的城市几何形态。低阶模型在捕捉几何影响方面存在局限，而高保真计算流体力学（CFD）模拟代价高昂，尤其是在多种几何或风况下更难以承受。为此，我们提出了一种用于在非结构化网格上合成稳态城市风场的生成扩散框架，该方法仅需几何信息作为输入。该框架将分层图神经网络与基于得分的扩散建模相结合，能够在无需时间推进或密集测量的情况下生成准确且多样化的速度场。模型在多个网格切片和风向上训练，能够泛化到未见过的几何形状，恢复关键流动结构（例如尾流和回流区），并提供不确定性感知的预测结果。消融研究验证了模型对网格变化的鲁棒性以及在不同推断策略下的性能表现。本工作迈出了构建面向构建环境的基础模型的第一步，可帮助城市规划者在城市密度增加和气候不确定性下快速评估设计决策。

BibTeX

```
@article{2512.14725v1,
  title={Generative Urban Flow Modeling: From Geometry to Airflow with Graph Diffusion},
  author={Francisco Giral and Álvaro Manzano and Ignacio Gómez and Petros Koumoutsakos and Soledad Le Clainche},
  journal={arXiv preprint arXiv:2512.14725v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14725v1}
}
```

## [Hybrid Attribution Priors for Explainable and Robust Model Training](http://arxiv.org/abs/2512.14719v1)

#归因先验#可解释性#鲁棒性

[PDF](https://arxiv.org/pdf/2512.14719v1)
[Abstract](http://arxiv.org/abs/2512.14719v1)

 NLP（可解释性/鲁棒性）

 很推荐

### 中文摘要

小型语言模型（SLMs）在需要低延迟和轻量部署的任务中被广泛使用，尤其是在分类任务上。随着可解释性和鲁棒性愈发重要，解释引导学习（通过在训练中引入基于归因的监督）成为一种有效框架；然而，如何导出通用且可靠的归因先验仍是一个重大挑战。通过对分类场景中代表性归因方法的分析，我们发现尽管这些方法能够可靠地标出与类别相关的词元，但它们常常集中于语义相近类别共有的常见关键词。由于这类语义相近的类别在标准训练下本就难以区分，这些归因提供的判别线索不足，从而限制了其提升模型区分能力的效果。为克服该限制，我们提出了面向类别的归因先验（Class-Aware Attribution Prior，CAP），这是一种新的归因先验提取框架，旨在引导语言模型捕捉细粒度的类别差异，并生成更突出的、判别性更强的归因先验。在此基础上，我们进一步提出了CAP Hybrid，将CAP产生的先验与现有归因技术的先验相结合，以形成更全面且平衡的监督信号。通过将模型的自身归因与这些丰富的先验对齐，该方法鼓励学习多样且与决策相关的特征。大量在充足数据、少样本和对抗性场景下的实验表明，我们的方法能够持续提升模型的可解释性和鲁棒性。

BibTeX

```
@article{2512.14719v1,
  title={Hybrid Attribution Priors for Explainable and Robust Model Training},
  author={Zhuoran Zhang and Feng Zhang and Shangyuan Li and Yang Shi and Yuanxing Zhang and Wei Chen and Tengjiao Wang and Kam-Fai Wong},
  journal={arXiv preprint arXiv:2512.14719v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14719v1}
}
```

## [DEER: Draft with Diffusion, Verify with Autoregressive Models](http://arxiv.org/abs/2512.15176v1)

#扩散语言模型#投机性解码#推理加速

[PDF](https://arxiv.org/pdf/2512.15176v1)
[Abstract](http://arxiv.org/abs/2512.15176v1)

 LLM

 很推荐

### 中文摘要

效率是由大型语言模型驱动的自治与推理系统面临的关键实际挑战，而自回归（AR）解码的固有延迟正日益成为瓶颈。投机性解码通过“先草稿后校验”的方案缓解了该成本，但现有方法依赖自回归草稿模型（即 drafter），带来了两类根本性问题：一是逐步的不确定性积累导致目标模型与 drafter 之间的信任逐渐崩溃；二是自回归 drafter 的固有串行解码限制了并行加速能力，从而限制了整体加速效果。本文指出，基于扩散的大型语言模型（dLLM）作为 drafter，凭借其根本不同的概率建模与高效并行解码策略，可自然克服上述问题。在此基础上，我们提出 DEER——一种以扩散草稿并由自回归模型校验的高效投机性解码框架。为实现高质量草稿，DEER 采用两阶段训练流程对 dLLM-based drafter 与目标自回归模型进行对齐，并进一步引入单步解码以生成较长的草稿段。实验证明，DEER 的草稿可接受长度可达 32 个 token，远超 EAGLE-3 的 10 个 token；在 HumanEval（基于 Qwen3-30B-A3B）上，DEER 实现了 5.54× 的加速，而 EAGLE-3 仅为 2.41×。代码、模型与演示将开源发布于 https://czc726.github.io/DEER/。

BibTeX

```
@article{2512.15176v1,
  title={DEER: Draft with Diffusion, Verify with Autoregressive Models},
  author={Zicong Cheng and Guo-Wei Yang and Jia Li and Zhijie Deng and Meng-Hao Guo and Shi-Min Hu},
  journal={arXiv preprint arXiv:2512.15176v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15176v1}
}
```

## [FADTI: Fourier and Attention Driven Diffusion for Multivariate Time Series Imputation](http://arxiv.org/abs/2512.15116v1)

#多变量时间序列插补#扩散生成模型#傅里叶频域先验

[PDF](https://arxiv.org/pdf/2512.15116v1)
[Abstract](http://arxiv.org/abs/2512.15116v1)

 时间序列 / 生成模型

 很推荐

### 中文摘要

多变量时间序列插补在医疗、交通预测和生物建模等应用中具有基础性作用，因为传感器故障和不规则采样会导致广泛的缺失值。然而，现有基于 Transformer 和扩散模型的方法缺乏显式的归纳偏置与频率感知能力，因而在面对具有结构化缺失模式和分布漂移时泛化能力受限。为此我们提出 FADTI，一种基于扩散的框架，通过可学习的傅里叶偏置投影（Fourier Bias Projection，FBP）模块注入频率信息驱动的特征调制，并将其与自注意力和门控卷积的时序建模相结合。FBP 支持多种谱基底，能够自适应地编码平稳与非平稳模式，从而将频域归纳偏置引入生成式插补过程。在多个基准数据集（包括本文新引入的生物时间序列数据集）上的实验表明，FADTI 在多种设置下均优于最先进方法，尤其在高缺失率情况下表现显著提升。代码已公开：https://anonymous.4open.science/r/TimeSeriesImputation-52BF

BibTeX

```
@article{2512.15116v1,
  title={FADTI: Fourier and Attention Driven Diffusion for Multivariate Time Series Imputation},
  author={Runze Li and Hanchen Wang and Wenjie Zhang and Binghao Li and Yu Zhang and Xuemin Lin and Ying Zhang},
  journal={arXiv preprint arXiv:2512.15116v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15116v1}
}
```

## [The Meta-Prompting Protocol: Orchestrating LLMs via Adversarial Feedback Loops](http://arxiv.org/abs/2512.15053v1)

#元提示协议#对抗反馈回路#文本微分（TextGrad）

[PDF](https://arxiv.org/pdf/2512.15053v1)
[Abstract](http://arxiv.org/abs/2512.15053v1)

 LLM

 很推荐

### 中文摘要

大型语言模型（LLMs）从概率性聊天界面向可作为可靠软件组件的转变，要求对交互范式进行根本性的重构。当前主流方法主要是基于经验的“提示工程”，难以为关键任务应用提供所需的确定性保证。我们提出了元提示协议（Meta-Prompting Protocol），这是一个将LLM编排形式化为可编程、自我优化系统的严谨理论框架。该协议的核心为“对抗三位体”（Adversarial Trinity），由生成器（P）、审计器（A）和优化器（O）三部分构成。通过将自然语言指令视为语义计算图中可微分的变量，并将文本化的批评作为梯度信号，该架构有助于缓解幻觉问题并防止模型退化。我们利用声明式编程范式（DSPy）与自动文本微分（TextGrad）证明了该方法的理论可行性，为概率计算时代的“可观测软件工程”奠定了基础。

BibTeX

```
@article{2512.15053v1,
  title={The Meta-Prompting Protocol: Orchestrating LLMs via Adversarial Feedback Loops},
  author={Fanzhe Fu},
  journal={arXiv preprint arXiv:2512.15053v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15053v1}
}
```

## [Epistemic diversity across language models mitigates knowledge collapse](http://arxiv.org/abs/2512.15011v1)

#认识性多样性#知识崩溃#自我训练/模型生态

[PDF](https://arxiv.org/pdf/2512.15011v1)
[Abstract](http://arxiv.org/abs/2512.15011v1)

 LLM

 很推荐

### 中文摘要

随着人工智能（AI）应用的日益广泛，人们对“知识崩溃”（即模型输出倾向于退化为最占主导和最中心的那一组观念）的担忧也在增加。先前研究证明了单模型崩溃现象——即当模型以自身生成的数据为训练来源时其性能会下降。受生态学启发，我们探讨了AI生态系统的多样性（即模型之间的认识/认知差异）能否缓解这种崩溃。本工作在单模型研究的基础上，将关注点扩展到以模型群体的集体输出为训练数据的生态系统。为研究多样性对模型性能的影响，我们将训练数据在多个语言模型之间进行划分，并在十次自训练迭代中评估由此产生的模型生态系统。我们发现，提升认识性多样性确实可以缓解知识崩溃，但这一效果存在最优水平：包含少数但高度多样化模型的生态系统无法充分表达真实分布的丰富混合成分，从而导致性能快速衰退；另一方面，将数据过度分散到过多模型上会削弱每个模型对真实分布的逼近能力，导致在第一轮迭代就出现较差性能。在AI单一文化（monoculture）的背景下，我们的结果表明，需要监控AI系统间的多样性并制定激励政策，鼓励更多面向领域或社区的专门化模型。

BibTeX

```
@article{2512.15011v1,
  title={Epistemic diversity across language models mitigates knowledge collapse},
  author={Damian Hodel and Jevin D. West},
  journal={arXiv preprint arXiv:2512.15011v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15011v1}
}
```

## [Evaluating the Capability of Video Question Generation for Expert Knowledge Elicitation](http://arxiv.org/abs/2512.15006v1)

#视频问题生成#专家知识抽取#检索式评估

[PDF](https://arxiv.org/pdf/2512.15006v1)
[Abstract](http://arxiv.org/abs/2512.15006v1)

 CV (VideoQA/VQG)

 很推荐

### 中文摘要

熟练的人工采访者能够从专家处获取有价值的信息。这引出一个基本问题：是什么使得某些问题比其他问题更有效？为了解答该问题，需要对问题生成模型进行定量评估。视频问题生成（VQG）是视频问答（VideoQA）研究中的一个方向，其任务是针对给定答案生成问题，但现有评估多侧重于能否回答这些问题，而非生成问题本身在引出专家未见知识方面的质量。为实现对VQG模型的持续改进，我们提出了一种评估协议：通过模拟与专家的问答交流并采用“问题到答案”的检索来衡量生成问题引出未知知识的能力。为此我们构建了新的数据集EgoExoAsk，该数据集由来自Ego-Exo4D专家旁白标注的27,666条问答对组成，用于训练检索器；基准测试则在Ego-Exo4D的验证集视频片段上构建。实验结果表明，我们的度量与问题生成设置呈合理一致性：能够访问更丰富上下文的模型会被评估为更好，支持了该评估协议的有效性。EgoExoAsk数据集已公开（https://github.com/omron-sinicx/VQG4ExpertKnowledge）。

BibTeX

```
@article{2512.15006v1,
  title={Evaluating the Capability of Video Question Generation for Expert Knowledge Elicitation},
  author={Huaying Zhang and Atsushi Hashimoto and Tosho Hirasawa},
  journal={arXiv preprint arXiv:2512.15006v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15006v1}
}
```

## [DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding](http://arxiv.org/abs/2512.15000v1)

#过程奖励模型#代码生成（函数即步骤）#元学习标签校正

[PDF](https://arxiv.org/pdf/2512.15000v1)
[Abstract](http://arxiv.org/abs/2512.15000v1)

 LLM

 很推荐

### 中文摘要

过程奖励模型（PRM）已成为通过测试时扩展（test-time scaling）提升大型语言模型（LLM）性能的关键手段，然而在代码生成领域其效果受限，主要原因在于代码缺乏有意义的步骤分解以及蒙特卡洛生成的中间标签存在噪声。为此我们提出 DreamPRM-Code —— 一种面向代码的过程奖励模型。该方法将函数视为推理步骤，采用“函数链（Chain-of-Function）”提示策略以诱导模块化的代码生成，使得对代码的 PRM 训练与应用可以类似于数学推理任务中对步骤评分的做法。为了解决中间标签的噪声问题，DreamPRM-Code 引入了一种基于元学习的标签校正机制：利用干净的最终解答的单元测试标签，通过双层优化（bi-level optimization）来迭代精炼由蒙特卡洛生成的中间标签。将该方法用于测试时扩展后，DreamPRM-Code 在 LiveCodeBench 基准上取得了 80.9 的 pass@1，达到了最先进水平，优于 OpenAI 的 o4-mini。

BibTeX

```
@article{2512.15000v1,
  title={DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding},
  author={Ruiyi Zhang and Peijia Qin and Qi Cao and Pengtao Xie},
  journal={arXiv preprint arXiv:2512.15000v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15000v1}
}
```

## [Improving Pre-trained Segmentation Models using Post-Processing](http://arxiv.org/abs/2512.14937v1)

#胶质瘤分割#后处理技术#预训练模型泛化

[PDF](https://arxiv.org/pdf/2512.14937v1)
[Abstract](http://arxiv.org/abs/2512.14937v1)

 CV（医学影像分割）

 很推荐

### 中文摘要

胶质瘤是成人中最常见的恶性脑肿瘤，也是致死率较高的肿瘤之一。尽管采取了积极治疗，患者的中位生存期仍不足15个月。准确的多参数MRI（mpMRI）肿瘤分割对于手术规划、放疗制定和疾病监测至关重要。尽管深度学习模型提高了自动分割的精度，但大规模预训练模型在泛化性方面表现不佳且常常低于期望，产生诸如误报、标签交换以及切片间不连续等系统性错误。这些限制还因不同团队对GPU资源的差异获取以及大规模模型训练带来的日益增长的环境成本而被放大。在本工作中，我们提出了自适应后处理技术，用以精炼由为不同类型肿瘤开发的大规模预训练模型所产生的胶质瘤分割结果。我们在多个BraTS 2025分割挑战任务中验证了这些技术，对于撒哈拉以南非洲子挑战排名指标提高了14.9%，在成人胶质瘤挑战中提高了0.9%。该方法推动脑肿瘤分割研究从不断复杂化的模型结构转向高效、临床对齐的后处理策略，强调精确性、计算公平性与可持续性。

BibTeX

```
@article{2512.14937v1,
  title={Improving Pre-trained Segmentation Models using Post-Processing},
  author={Abhijeet Parida and Daniel Capellán-Martín and Zhifan Jiang and Nishad Kulkarni and Krithika Iyer and Austin Tapp and Syed Muhammad Anwar and María J. Ledesma-Carbayo and Marius George Linguraru},
  journal={arXiv preprint arXiv:2512.14937v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14937v1}
}
```

## [Epistemic diversity across language models mitigates knowledge collapse](http://arxiv.org/abs/2512.15011v1)

#知识崩塌#认知多样性#自训练生态系统

[PDF](https://arxiv.org/pdf/2512.15011v1)
[Abstract](http://arxiv.org/abs/2512.15011v1)

 LLM

 很推荐

### 中文摘要

随着人工智能（AI）应用的普及，人们开始担忧“知识崩塌”问题，即模型知识退化到最占主导、最中心的一小部分观念。先前研究已证明了单模型崩塌现象——在模型以自身输出为训练数据时性能衰退。受生态学启发，本文研究了AI生态系统中的多样性（即模型之间的认知差异）能否缓解这一崩塌问题。我们在单模型自训练方法基础上扩展，构造了多个模型组成的生态系统，并让这些模型在共同输出上相互训练。为评估多样性对性能的影响，我们将训练数据在不同语言模型间进行划分，并在十轮自训练迭代中评估生态系统表现。结果表明，增加认知多样性能够减轻知识崩塌，但这一效应存在最优水平：模型数量太少虽多样但不足以呈现真实分布的丰富混合，导致性能快速下降；而模型数量过多则使得每个模型对真实分布的近似能力下降，导致在第一轮即出现较差表现。在AI单一文化（monoculture）的背景下，研究表明需监测跨系统的多样性，并制定激励政策以促进更多面向领域和社区的专用模型发展。

BibTeX

```
@article{2512.15011v1,
  title={Epistemic diversity across language models mitigates knowledge collapse},
  author={Damian Hodel and Jevin D. West},
  journal={arXiv preprint arXiv:2512.15011v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15011v1}
}
```

## [Evaluating the Capability of Video Question Generation for Expert Knowledge Elicitation](http://arxiv.org/abs/2512.15006v1)

#视频问题生成#专家知识引导#检索式评估

[PDF](https://arxiv.org/pdf/2512.15006v1)
[Abstract](http://arxiv.org/abs/2512.15006v1)

 CV

 很推荐

### 中文摘要

熟练的人类访谈者能够从专家处获取有价值的信息。这引出了一个基本问题：是什么使得某些问题比其他问题更有效？为了解答这一问题，需要对问题生成模型进行量化评估。视频问题生成（VQG）是面向视频问答（VideoQA）的一个研究方向，其任务是在给定答案的情况下生成问题。现有评估通常侧重于能否回答所生成的问题，而不是生成问题本身在诱导专家披露未知知识方面的质量。相反，我们关注问题在引出专家未见知识方面的效力。为持续改进VQG模型，我们提出了一种评估方案，通过使用“问题到答案”的检索来模拟与专家的问答交流，从而评估模型的能力。为构建检索器，我们创建了一个新数据集EgoExoAsk，该数据集由来自Ego-Exo4D专家解说标注的27,666对问答构成。使用EgoExoAsk的训练集来训练检索器，并基于验证集与Ego-Exo4D视频片段构建基准测试。实验结果表明，我们的度量与问题生成设置之间具有合理的一致性：能够访问更丰富上下文的模型获得更好的评估结果，支持了该评估方案的有效性。EgoExoAsk数据集已在 https://github.com/omron-sinicx/VQG4ExpertKnowledge 上公开。

BibTeX

```
@article{2512.15006v1,
  title={Evaluating the Capability of Video Question Generation for Expert Knowledge Elicitation},
  author={Huaying Zhang and Atsushi Hashimoto and Tosho Hirasawa},
  journal={arXiv preprint arXiv:2512.15006v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15006v1}
}
```

## [DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding](http://arxiv.org/abs/2512.15000v1)

#过程奖励模型#模块化代码生成#元学习标签校正

[PDF](https://arxiv.org/pdf/2512.15000v1)
[Abstract](http://arxiv.org/abs/2512.15000v1)

 LLM (代码生成)

 很推荐

### 中文摘要

过程奖励模型（PRMs）已成为通过测试时扩展（test-time scaling）提升大型语言模型（LLMs）性能的重要手段，但在代码生成领域其效果仍然受限，原因在于代码缺乏有意义的步骤分解，以及蒙特卡洛生成的中间标签存在噪声。我们提出了 DreamPRM-Code，一种面向代码的 PRM，将函数视为推理步骤，采用 Chain-of-Function 提示策略以诱导模块化的代码生成，从而使得 PRM 的训练与应用可以类比于数学推理任务。为了解决标签噪声问题，DreamPRM-Code 引入了一种基于元学习的校正机制，该机制利用干净的最终解单元测试标签，通过双层优化（bi-level optimization）来精炼中间标签。在测试时扩展场景下，DreamPRM-Code 在 LiveCodeBench 上取得了最先进的性能，pass@1 达到 80.9%，超越了 OpenAI o4-mini。

BibTeX

```
@article{2512.15000v1,
  title={DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding},
  author={Ruiyi Zhang and Peijia Qin and Qi Cao and Pengtao Xie},
  journal={arXiv preprint arXiv:2512.15000v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15000v1}
}
```

## [Double Horizon Model-Based Policy Optimization](http://arxiv.org/abs/2512.15439v1)

#双时域滚动#模型为基础强化学习#策略优化

[PDF](https://arxiv.org/pdf/2512.15439v1)
[Abstract](http://arxiv.org/abs/2512.15439v1)

 RL（模型为基础强化学习）

 很推荐

### 中文摘要

模型为基础的强化学习（MBRL）通过从学习到的动力学模型生成合成轨迹（称为rollouts）来减少真实环境采样的开销。然而，选择rollout的长度会带来两方面的困境：（1）较长的rollout有利于保持近端策略训练（on-policy）但会放大模型偏差，这表明需要一个中间时域以缓解分布偏移（即当前策略数据与以往离策略样本之间的差距）；（2）此外，较长的模型rollout可能降低价值估计的偏差，但通过多步反向传播会增加策略梯度的方差，从而需要另一个中间时域以获得稳定的梯度估计。而这两个最优时域可能并不相同。为了解决这一冲突，我们提出了双时域模型为基础的策略优化方法（Double Horizon Model-Based Policy Optimization，DHMBPO），将rollout过程划分为一个较长的“分布rollout”（Distribution Rollout, DR）和一个较短的“训练rollout”（Training Rollout, TR）。DR用于生成近端策略的状态样本以减轻分布偏移；而短的TR利用可微分的转移过程提供精确的价值梯度估计，带来稳定的梯度更新，从而需要更少的更新并降低总体运行时间。我们证明了双时域方法能够有效平衡分布偏移、模型偏差与梯度不稳定性，并在连续控制基准上在样本效率和运行时方面均超越了现有的MBRL方法。

BibTeX

```
@article{2512.15439v1,
  title={Double Horizon Model-Based Policy Optimization},
  author={Akihiro Kubo and Paavo Parmas and Shin Ishii},
  journal={arXiv preprint arXiv:2512.15439v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15439v1}
}
```

## [Outer-Learning Framework for Playing Multi-Player Trick-Taking Card Games: A Case Study in Skat](http://arxiv.org/abs/2512.15435v1)

#外部学习框架#自我博弈#特征哈希

[PDF](https://arxiv.org/pdf/2512.15435v1)
[Abstract](http://arxiv.org/abs/2512.15435v1)

 RL（游戏AI/多智能体博弈）

 很推荐

### 中文摘要

在像Skat或桥牌这样的多人诀窍取牌类纸牌游戏中，游戏的早期阶段（例如叫牌、游戏种类选择和初始出牌）往往比中后期的精细博弈对胜负更为关键。在当前计算能力的限制下，这类早期决策通常依赖从大量人类专家对局中统计得到的信息。本文提出并评估了一种通用的引导式外部学习（outer-learning）自举框架，通过将数百万局自我博弈生成的数据并入人类对局数据库来扩充统计信息，从而提升预测精度。我们采用完备的特征哈希函数来处理压缩表格，构建了一个自我改进的纸牌游戏引擎，使得新推断出的知识能在自我学习过程中不断完善。Skat 的案例研究表明，该自动化方法可用于支持游戏中的多种决策。

BibTeX

```
@article{2512.15435v1,
  title={Outer-Learning Framework for Playing Multi-Player Trick-Taking Card Games: A Case Study in Skat},
  author={Stefan Edelkamp},
  journal={arXiv preprint arXiv:2512.15435v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15435v1}
}
```

## [FM-EAC: Feature Model-based Enhanced Actor-Critic for Multi-Task Control in Dynamic Environments](http://arxiv.org/abs/2512.15430v1)

#特征模型#模型型与无模型强化学习结合#多任务控制

[PDF](https://arxiv.org/pdf/2512.15430v1)
[Abstract](http://arxiv.org/abs/2512.15430v1)

 RL

 很推荐

### 中文摘要

模型型强化学习（MBRL）与无模型强化学习（MFRL）沿着不同路径发展，但在 Dyna-Q 的设计中存在汇合点。然而，现代的强化学习方法在跨任务和跨场景的有效迁移性方面仍存在挑战。针对这一局限性，我们提出了一种通用算法：基于特征模型的增强型行动者-评论家（FM-EAC），用于动态环境下的多任务控制。FM-EAC 将规划、执行与学习有机结合，融合了 MBRL 与 MFRL 的优势，并通过新颖的基于特征的模型与改进的行动者-评论家框架提升了泛化能力。在城市与农业应用场景的仿真实验中，FM-EAC 稳定优于多种最先进的 MBRL 与 MFRL 方法。更重要的是，FM-EAC 支持根据用户需求对不同子网络进行定制化设计。

BibTeX

```
@article{2512.15430v1,
  title={FM-EAC: Feature Model-based Enhanced Actor-Critic for Multi-Task Control in Dynamic Environments},
  author={Quanxi Zhou and Wencan Mao and Manabu Tsukada and John C. S. Lui and Yusheng Ji},
  journal={arXiv preprint arXiv:2512.15430v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15430v1}
}
```

## [Quantum Machine Learning for Cybersecurity: A Taxonomy and Future Directions](http://arxiv.org/abs/2512.15286v1)

#量子机器学习#网络安全#入侵检测

[PDF](https://arxiv.org/pdf/2512.15286v1)
[Abstract](http://arxiv.org/abs/2512.15286v1)

 量子机器学习（QML）/网络安全

 很推荐

### 中文摘要

近年来，随着网络威胁数量的增加、攻击手段的快速演变及海量数据的产生，基于传统规则、特征签名和经典机器学习的方法逐渐失效，难以满足防御需求。作为一种替代方案，量子机器学习（QML）近来受到关注，通过基于量子力学的计算能够在某些问题上更高效地编码和处理高维结构。本文综述了与网络安全相关的主要QML技术，包括量子神经网络（QNN）、量子支持向量机（QSVM）、变分量子电路（VQC）和量子生成对抗网络（QGAN），并对比了这些方法与现有研究的贡献与改进。文章还将这些方法在监督学习、无监督学习和生成式学习范式中的应用进行映射，讨论了它们在入侵与异常检测、恶意软件与僵尸网络分类、加密流量分析等核心网络安全任务中的潜力与落地场景。此外，论文探讨了QML在云计算安全领域提升安全性与可扩展性方面的应用，并深入分析了QML在网络安全领域面临的多种局限性及相应的改进与未来研究方向。

BibTeX

```
@article{2512.15286v1,
  title={Quantum Machine Learning for Cybersecurity: A Taxonomy and Future Directions},
  author={Siva Sai and Ishika Goyal and Shubham Sharma and Sri Harshita Manuri and Vinay Chamola and Rajkumar Buyya},
  journal={arXiv preprint arXiv:2512.15286v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15286v1}
}
```

## [Improving Pre-trained Segmentation Models using Post-Processing](http://arxiv.org/abs/2512.14937v1)

#胶质瘤分割#后处理方法#医学影像可持续性

[PDF](https://arxiv.org/pdf/2512.14937v1)
[Abstract](http://arxiv.org/abs/2512.14937v1)

 CV（Medical Imaging / 医学影像）

 很推荐

### 中文摘要

胶质瘤是成年人中最常见且最致命的恶性脑肿瘤之一，尽管采用积极治疗，患者的中位生存期仍不足15个月。多参数磁共振成像（mpMRI）肿瘤的准确分割对于手术规划、放疗设计和疾病随访具有重要意义。尽管深度学习方法提升了自动分割的精度，但大规模预训练模型在泛化性方面表现较差，常出现系统性错误，例如假阳性、标签互换和切片间不连续等问题。此外，GPU资源分配不均与大规模模型训练带来的环境成本也加剧了这些限制。本文提出自适应后处理技术，以精炼大规模预训练模型针对不同类型肿瘤所产生的胶质瘤分割结果。我们在多项 BraTS 2025 分割挑战任务中验证了该方法，排名指标在撒哈拉以南非洲挑战中提升了14.9%，在成人胶质瘤挑战中提升了0.9%。该方法倡导将脑肿瘤分割研究重心从追求更复杂的模型架构，转向高效、临床对齐且在计算上更公平与可持续的后处理策略。

BibTeX

```
@article{2512.14937v1,
  title={Improving Pre-trained Segmentation Models using Post-Processing},
  author={Abhijeet Parida and Daniel Capellán-Martín and Zhifan Jiang and Nishad Kulkarni and Krithika Iyer and Austin Tapp and Syed Muhammad Anwar and María J. Ledesma-Carbayo and Marius George Linguraru},
  journal={arXiv preprint arXiv:2512.14937v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14937v1}
}
```

## [Quantifying Return on Security Controls in LLM Systems](http://arxiv.org/abs/2512.15081v1)

#LLM安全#风险量化#控制回报率（RoC）

[PDF](https://arxiv.org/pdf/2512.15081v1)
[Abstract](http://arxiv.org/abs/2512.15081v1)

 LLM

 很推荐

### 中文摘要

尽管大模型（LLM）日益被用于对安全性要求高的工作流中，实践者仍然缺乏关于哪些防护措施值得部署的定量指导。本文提出了一个以决策为导向且可复现的方法学，能够量化残余风险、将对抗性探测结果转换为财务风险估计与控制回报（RoC）指标，并支持对基于LLM系统的分层防御进行货币化比较。作者构建了一个基于检索增强生成（RAG）的服务，使用 DeepSeek-R1 模型在包含合成个人识别信息（PII）的语料上运行，并使用 Garak 对五类漏洞进行自动化攻击：PII 泄露、潜在上下文注入、提示注入、对抗性攻击生成与偏离行为。对于每一对（漏洞、控制措施），通过拉普拉斯继承法（Laplace's Rule of Succession）估计攻击成功概率，并将其与基于公开数据校准的损失三角分布相结合，在10,000次蒙特卡洛模拟中生成超损失曲线与期望损失。随后将三种广泛采用的缓解措施——基于属性的访问控制（ABAC）；使用 Microsoft Presidio 的命名实体识别（NER）脱敏；以及 NeMo Guardrails——与基线 RAG 配置进行比较。基线系统在 PII、潜在注入与提示注入三类攻击上的成功率极高（>=0.98），导致每次攻击场景的模拟期望总损失为约 31.3 万美元。ABAC 将 PII 与提示相关攻击的成功概率几乎压至零，使总期望损失降低约 94%，获得 RoC 为 9.83；NER 脱敏同样消除了 PII 泄露，RoC 达到 5.97；而 NeMo Guardrails 仅带来微弱收益，RoC 为 0.05。

BibTeX

```
@article{2512.15081v1,
  title={Quantifying Return on Security Controls in LLM Systems},
  author={Richard Helder Moulton and Austin O'Brien and John D. Hastings},
  journal={arXiv preprint arXiv:2512.15081v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15081v1}
}
```

## [How Do Semantically Equivalent Code Transformations Impact Membership Inference on LLMs for Code?](http://arxiv.org/abs/2512.15468v1)

#成员推断#语义等价代码变换#许可合规漏洞

[PDF](https://arxiv.org/pdf/2512.15468v1)
[Abstract](http://arxiv.org/abs/2512.15468v1)

 LLM（代码）

 很推荐

### 中文摘要

大规模代码语言模型的成功依赖于大量代码数据，包括来自公共开源仓库（如 GitHub）以及公司内部的私有机密代码，这引发了知识产权合规和许可受限代码被未授权使用的担忧。尽管已有成员推断（MI）技术被提出用于检测此类未授权使用，但通过保持语义不变而仅修改语法的语义等价代码变换可能削弱这些检测方法。在本文中，我们系统性地研究了语义等价代码变换规则是否可被用于规避 MI 检测。实验结果表明，对于每一条变换规则，模型准确率在最坏情况下仅下降约 1.5%，表明经过变换的数据集可以有效替代用于微调的原始数据集。此外，我们发现其中一条规则（RenameVariable，变量重命名）将 MI 的成功率降低了 10.19%，突显了其在掩盖受限代码存在方面的潜力。为验证这些发现，我们进行了因果分析，确认变量重命名对破坏 MI 检测具有最强的因果影响。值得注意的是，组合多种变换并未进一步降低 MI 的有效性。我们的结果暴露了训练代码大模型时许可合规执行中的一个关键漏洞，表明基于变换的混淆技术可以显著削弱会员推断检测。

BibTeX

```
@article{2512.15468v1,
  title={How Do Semantically Equivalent Code Transformations Impact Membership Inference on LLMs for Code?},
  author={Hua Yang and Alejandro Velasco and Thanh Le-Cong and Md Nazmul Haque and Bowen Xu and Denys Poshyvanyk},
  journal={arXiv preprint arXiv:2512.15468v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15468v1}
}
```

## [How Large Language Models are Designed to Hallucinate](http://arxiv.org/abs/2509.16297v1)

#变压器架构#幻觉分类#真相约束设计

[PDF](https://arxiv.org/pdf/2509.16297v1)
[Abstract](http://arxiv.org/abs/2509.16297v1)

 LLM

 很推荐

### 中文摘要

大型语言模型（LLM）在语言表达和推理任务上表现出显著的流畅性，但仍系统性地易于产生幻觉（hallucination）。现有观点多将幻觉归因于数据缺失、上下文限制或优化失误。我们则主张，幻觉是变压器（transformer）架构的结构性产物。作为一种连贯性引擎，变压器被驱动去生成流畅的续写，其自注意力机制可以模拟意义的关系结构，但缺乏维系人类理解的存在性基底诸如时间性、情感与关怀（temporality, mood, care）。基于此，我们区分了两类幻觉：本体性幻觉（ontological hallucination），当续写要求披露世界中实体的存在时发生；以及残留推理幻觉（residual reasoning hallucination），模型通过重用文本中人类推理的痕迹来模拟推断。我们通过与海德格尔范畴相对应的案例研究加以说明，并在对十二个大型语言模型的实验中展示了在延长提示下模拟“自我保存”行为的出现。我们的贡献有三点：(1) 提供一个比较性论述以说明现有解释为何不足；(2) 提出一个与存在性结构相关联的、可预测的幻觉分类法并建议相应基准；(3) 指出面向“真相约束”架构的设计方向，使模型在缺乏披露依据时能够选择拒绝或延迟回答。我们结论是：幻觉并非偶发缺陷，而是基于变压器模型的一项决定性限制，外在的支架（scaffolding）可掩盖但无法根本消除这一现象。

BibTeX

```
@article{2509.16297v1,
  title={How Large Language Models are Designed to Hallucinate},
  author={Richard Ackermann and Simeon Emanuilov},
  journal={arXiv preprint arXiv:2509.16297v1},
  year={2025},
  url={http://arxiv.org/abs/2509.16297v1}
}
```

## [Stemming Hallucination in Language Models Using a Licensing Oracle](http://arxiv.org/abs/2511.06073v1)

#幻觉抑制#知识图谱验证#基于架构的事实约束

[PDF](https://arxiv.org/pdf/2511.06073v1)
[Abstract](http://arxiv.org/abs/2511.06073v1)

 LLM

 很推荐

### 中文摘要

语言模型在自然语言生成方面表现出显著能力，但仍易产生“幻觉”，即在语法上连贯但事实不正确的输出。本研究提出了“许可Oracle”（Licensing Oracle），这是一种通过对结构化知识图谱进行形式化验证，从而强制实施真实约束以遏制语言模型幻觉的架构性解决方案。不同于依赖数据扩增或微调等统计方法，许可Oracle在生成流程中嵌入了一个确定性的验证步骤，保证仅输出经验证的事实性断言。我们通过一系列实验评估了许可Oracle的有效性，并将其与若干最先进方法进行了比较，包括基线语言模型生成、为事实召回进行的微调、为弃答行为进行的微调以及检索增强生成（RAG）。结果表明，尽管RAG和微调能在一定程度上提升性能，但仍无法根除幻觉；相比之下，许可Oracle实现了完美的弃答精确率（AP = 1.0）和零错误回答（FAR-NE = 0.0），在事实性回答中达到89.1%的准确率，从而确保仅生成有效的断言。本工作表明，在具有结构化知识表示的领域中，诸如许可Oracle的架构性创新可提供统计方法无法比拟的正确性保证，提供了解决幻觉问题的必要且充分的路径。尽管许可Oracle专为基于事实的领域设计，其框架为未来具备真实约束的生成系统奠定了基础，为构建可靠、具有认知依据的模型指明了新方向。

BibTeX

```
@article{2511.06073v1,
  title={Stemming Hallucination in Language Models Using a Licensing Oracle},
  author={Simeon Emanuilov and Richard Ackermann},
  journal={arXiv preprint arXiv:2511.06073v1},
  year={2025},
  url={http://arxiv.org/abs/2511.06073v1}
}
```

## [From Isolation to Entanglement: When Do Interpretability Methods Identify and Disentangle Known Concepts?](http://arxiv.org/abs/2512.15134v1)

#概念可解释性#表示解缠#稀疏表征/操控

[PDF](https://arxiv.org/pdf/2512.15134v1)
[Abstract](http://arxiv.org/abs/2512.15134v1)

 NLP

 很推荐

### 中文摘要

可解释性的核心目标是从神经网络的激活中恢复对因果相关概念的表示。通常对这些概念表示的质量是在孤立条件下评估的，并伴随一些在实践中可能不成立的隐含独立性假设。因此尚不清楚常用的表征方法（包括稀疏自编码器和稀疏探针）是否能恢复解缠的概念表示。本文提出一种多概念评估设置，在该设置中我们可控地调整文本概念（如情感、领域和时态）之间的相关性，并分析随着这些概念间相关性增强时的表现。首先评估表征器在日益增强的相关强度下学习各概念解缠表示的能力。我们观察到从概念到特征存在一对多的关系：每个特征至多对应一个概念，但每个概念分布在多个特征上。随后我们进行操控实验，测量每个概念是否可被独立操纵。即使在概念分布均匀的训练设置下，稀疏自编码器的特征在被操控时通常会影响多个概念，表明这些特征既不具有选择性也不独立；尽管如此，这些特征确实影响不相交的子空间。研究结果表明，用于衡量解缠性的相关性指标通常不足以在操控情形下证明独立性，且影响不相交子空间亦不足以保证概念的选择性。这些发现强调了在可解释性研究中采用组合性/复合性评估的重要性。

BibTeX

```
@article{2512.15134v1,
  title={From Isolation to Entanglement: When Do Interpretability Methods Identify and Disentangle Known Concepts?},
  author={Aaron Mueller and Andrew Lee and Shruti Joshi and Ekdeep Singh Lubana and Dhanya Sridhar and Patrik Reizinger},
  journal={arXiv preprint arXiv:2512.15134v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15134v1}
}
```

## [QoS-Aware Hierarchical Reinforcement Learning for Joint Link Selection and Trajectory Optimization in SAGIN-Supported UAV Mobility Management](http://arxiv.org/abs/2512.15119v1)

#SAGIN#层次深度强化学习#UAV轨迹与链路优化

[PDF](https://arxiv.org/pdf/2512.15119v1)
[Abstract](http://arxiv.org/abs/2512.15119v1)

 RL

 很推荐

### 中文摘要

由于无人机（UAV）在高度和水平运动上存在显著变化，任何单一网络都难以保证连续且可靠的三维覆盖。为此，空间-空-地一体化网络（SAGIN）成为实现无人机普适连接的关键架构。针对异构网络间覆盖与信号特性差异明显的问题，本文将SAGIN 中的无人机移动性管理建模为一个耦合离散链路选择与连续轨迹优化的约束多目标联合优化问题。在此基础上，我们提出了一种两层多智能体层次深度强化学习（HDRL）框架，将原问题分解为两个交替可解的子问题。为将复杂的链路选择决策映射为紧凑的离散动作空间，顶层设计了双深度Q网络（DDQN）算法，通过双重Q值估计实现稳定且高质量的策略学习；为在满足服务质量（QoS）约束的前提下处理连续轨迹动作空间，底层将软演员-评论家（SAC）的最大熵机制与基于拉格朗日的受约束SAC（CSAC）相结合，动态调整拉格朗日乘子以在约束满足与策略优化之间取得平衡。此外，所提算法可在集中训练、去中心化执行（CTDE）范式下扩展到多无人机场景，从而获得更具泛化能力的策略。仿真结果表明，所提方案在吞吐量、链路切换频率和QoS 满意度方面均显著优于现有基准方法。

BibTeX

```
@article{2512.15119v1,
  title={QoS-Aware Hierarchical Reinforcement Learning for Joint Link Selection and Trajectory Optimization in SAGIN-Supported UAV Mobility Management},
  author={Jiayang Wan and Ke He and Yafei Wang and Fan Liu and Wenjin Wang and Shi Jin},
  journal={arXiv preprint arXiv:2512.15119v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15119v1}
}
```

## [The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems](http://arxiv.org/abs/2512.15068v1)

#检索增强生成（RAG）#幻觉检测#嵌入表示局限

[PDF](https://arxiv.org/pdf/2512.15068v1)
[Abstract](http://arxiv.org/abs/2512.15068v1)

 LLM

 很推荐

### 中文摘要

检索增强生成（RAG）系统尽管依赖检索到的证据，仍然容易出现幻觉问题。当前的检测方法主要依赖语义相似度和自然语言推理（NLI），但这些方法的根本性限制尚未被严格刻画。我们将一致性预测（conformal prediction）应用于幻觉检测，给出有限样本覆盖率保证，从而能够精确量化检测能力。使用约600条样本的校准集，在合成幻觉数据集（Natural Questions）上我们达到了94%的覆盖率并且假阳性率为0%。然而，在跨多种大模型（GPT-4、ChatGPT、GPT-3、Llama-2、Mistral）的三个真实幻觉基准上，基于嵌入的方法——包括最先进的 OpenAI text-embedding-3-large 与交叉编码器模型——显示出不可接受的假阳性率：在 HaluEval 上为100%、在 RAGTruth 上为88%、在 WikiBio 上为50%。关键的是，作为判定者的 GPT-4 在相同数据上的假阳性率仅为7%（95% 置信区间：[3.4%，13.7%]），证明该任务可通过推理解决。我们将这种现象称为“语义错觉”：语义上看似合理的幻觉在保留与源文档相似性的同时引入了嵌入不可见的事实性错误。这一限制跨越嵌入架构、生成型大模型和任务类型持久存在，表明基于嵌入的检测不足以支持生产环境下的 RAG 部署。

BibTeX

```
@article{2512.15068v1,
  title={The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems},
  author={Debu Sinha},
  journal={arXiv preprint arXiv:2512.15068v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15068v1}
}
```

## [EMFusion: Conditional Diffusion Framework for Trustworthy Frequency Selective EMF Forecasting in Wireless Networks](http://arxiv.org/abs/2512.15067v1)

#频率选择性EMF预测#条件扩散概率建模#不确定性量化

[PDF](https://arxiv.org/pdf/2512.15067v1)
[Abstract](http://arxiv.org/abs/2512.15067v1)

 时间序列预测

 很推荐

### 中文摘要

随着无线基础设施的快速增长，准确估计和预测电磁场（EMF）强度愈发重要，以确保持续合规、评估潜在健康影响并支持高效的网络规划。目前研究多集中于宽带聚合EMF数据的单变量预测，而要捕捉跨运营商与跨频率的差异以便主动式网络规划，需采用频率选择性的多变量预测。为此，本文提出了EMFusion——一种条件化的多变量扩散概率预测框架，能够融合多种上下文信息（如时段、季节和节假日）并提供明确的不确定性估计。所提出的架构以残差U-Net为主干，辅以跨注意力机制以动态整合外部条件来引导生成过程。此外，EMFusion引入了一种基于插补的采样策略，将预测视为结构性修补任务，在测量不规则时仍能保证时间上的一致性。不同于标准的点估计预测器，EMFusion直接从所学的条件分布生成经校准的概率预测区间，提供对决策至关重要的显式不确定性量化。在频率选择性EMF数据集上的数值实验表明，结合工作时段等上下文信息的EMFusion优于有无条件信息的基线模型：相较最佳基线，EMFusion在连续排序概率评分（CRPS）上提升23.85%，在归一化均方根误差上提升13.93%，并将预测CRPS误差降低22.47%。

BibTeX

```
@article{2512.15067v1,
  title={EMFusion: Conditional Diffusion Framework for Trustworthy Frequency Selective EMF Forecasting in Wireless Networks},
  author={Zijiang Yan and Yixiang Huang and Jianhua Pei and Hina Tabassum and Luca Chiaraviglio},
  journal={arXiv preprint arXiv:2512.15067v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15067v1}
}
```

## [Parameter Efficient Multimodal Instruction Tuning for Romanian Vision Language Models](http://arxiv.org/abs/2512.14926v1)

#罗马尼亚语#视觉-语言模型#参数高效微调(LoRA)

[PDF](https://arxiv.org/pdf/2512.14926v1)
[Abstract](http://arxiv.org/abs/2512.14926v1)

 Multimodal (Vision-Language Models)

 很推荐

### 中文摘要

关注低资源语言是推动生成式人工智能普及的关键一步。本文旨在缩小罗马尼亚语在多模态自然语言处理领域的资源差距。我们将广为使用的Flickr30k数据集翻译为罗马尼亚语，并通过利用开源大模型进一步扩展为视觉问答数据集。为了验证数据集的有效性，我们在罗马尼亚语视觉问答任务上微调了开源视觉-语言模型（VLM）。所选模型来自三大常用模型族：LLaMA 3.2、LLaVA 1.6 和 Qwen2，微调时采用了参数高效的LoRA方法。实验结果表明，微调后的模型在罗马尼亚语视觉问答上能力显著提升，并且在未直接训练的任务（如罗马尼亚语图像描述生成）上也取得了进步。七十亿参数的Qwen2-VL-RoVQA在两项任务上均获得最好成绩，BERTScore F1分别较原始版本提升了+6.05%和+2.61%。此外，微调模型在语法错误方面有显著减少，表明其在语言理解能力之外，罗马尼亚语的流利性也得到了提升。

BibTeX

```
@article{2512.14926v1,
  title={Parameter Efficient Multimodal Instruction Tuning for Romanian Vision Language Models},
  author={George-Andrei Dima and Dumitru-Clementin Cercel},
  journal={arXiv preprint arXiv:2512.14926v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14926v1}
}
```

## [Let the Barbarians In: How AI Can Accelerate Systems Performance Research](http://arxiv.org/abs/2512.14806v1)

#AI驱动研究（ADRS）#系统性能优化#自动化生成与验证

[PDF](https://arxiv.org/pdf/2512.14806v1)
[Abstract](http://arxiv.org/abs/2512.14806v1)

 Systems（系统性能与自动化）

 很推荐

### 中文摘要

人工智能（AI）正开始通过自动化新解法的发现来改变研究过程。这一转变依赖于可靠的验证器，因为以AI为驱动的方法需要验证器来检验候选解的有效性。致力于提升系统性能的研究尤其适合这种范式，因为系统性能问题天然具备可验证性：候选方案可以在真实系统或仿真器中实现，并在预定义的工作负载下进行评估。我们将这一代、评估与改进的迭代循环称为面向系统的AI驱动研究（AI-Driven Research for Systems，ADRS）。通过若干开源ADRS实例（例如 OpenEvolve、GEPA 与 ShinkaEvolve），我们在十个案例研究中（如多区域云调度、专家混合（Mixture-of-Experts）负载均衡、基于大模型的SQL、事务调度等）展示了ADRS生成的方案能够匹敌甚至超越人工设计的最新成果。基于这些发现，我们概述了有效使用ADRS的最佳实践（例如提示规范的粒度、反馈量、稳健评估等），并讨论了未来的研究方向及其影响。尽管目前尚无一种可在所有系统研究中通用的ADRS应用配方，但我们希望初步结果与所识别的挑战能为未来工作提供有意义的指引，随着研究者工作重心逐步转向问题表述与策略性监督，将更好地支持该领域发展。注：本文是我们先前工作的扩展 [14]，新增了对多个ADRS框架的广泛评估并对最佳实践给出更深入的分析与见解。

BibTeX

```
@article{2512.14806v1,
  title={Let the Barbarians In: How AI Can Accelerate Systems Performance Research},
  author={Audrey Cheng and Shu Liu and Melissa Pan and Zhifei Li and Shubham Agarwal and Mert Cemri and Bowen Wang and Alexander Krentsel and Tian Xia and Jongseok Park and Shuo Yang and Jeff Chen and Lakshya Agrawal and Ashwin Naren and Shulu Li and Ruiying Ma and Aditya Desai and Jiarong Xing and Koushik Sen and Matei Zaharia and Ion Stoica},
  journal={arXiv preprint arXiv:2512.14806v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14806v1}
}
```

## [Privacy-Preserving Feature Valuation in Vertical Federated Learning Using Shapley-CMI and PSI Permutation](http://arxiv.org/abs/2512.14767v1)

#垂直联邦学习#Shapley-CMI特征估值#私有集合交集(PSI)

[PDF](https://arxiv.org/pdf/2512.14767v1)
[Abstract](http://arxiv.org/abs/2512.14767v1)

 垂直联邦学习 (VFL)

 很推荐

### 中文摘要

联邦学习（FL）是一种新兴的机器学习范式，允许多方在不共享原始数据的前提下协同训练模型，从而保障数据隐私。在垂直联邦学习（VFL）场景中，各方针对相同用户持有不同的特征，关键挑战之一是在任何模型训练之前（尤其是早期无模型可用时）评估各方特征的贡献。为了解决这一问题，近期提出了基于条件互信息（CMI）的无模型信息论特征估值方法Shapley-CMI。然而，原始方法在实际实现上未能提供可用于安全计算所需置换与交集操作的实用实现。本文提出了一种用于VFL的隐私保护Shapley-CMI实现。系统引入了一个私有集合交集（PSI）服务器，负责在离散化且加密的ID分组上执行所有必要的特征置换并计算加密的交集规模，整个过程中无需交换原始数据。各参与方随后使用这些交集结果独立计算Shapley-CMI值，从而得到其特征的边际效用。初步实验验证了该系统的正确性与隐私性，表明该方法在VFL中能够安全且高效地估算特征贡献。该方法在保证数据保密性的同时可扩展到多方场景，并支持在不共享原始数据或训练模型的前提下实现公平的数据估值。

BibTeX

```
@article{2512.14767v1,
  title={Privacy-Preserving Feature Valuation in Vertical Federated Learning Using Shapley-CMI and PSI Permutation},
  author={Unai Laskurain and Aitor Aguirre-Ortuzar and Urko Zurutuza},
  journal={arXiv preprint arXiv:2512.14767v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14767v1}
}
```

## [Scaling Causal Mediation for Complex Systems: A Framework for Root Cause Analysis](http://arxiv.org/abs/2512.14764v1)

#因果中介分析#根因分析#可扩展因果推断

[PDF](https://arxiv.org/pdf/2512.14764v1)
[Abstract](http://arxiv.org/abs/2512.14764v1)

 因果推断

 很推荐

### 中文摘要

现代运维系统——从物流与云基础设施到工业物联网——由复杂且相互依赖的过程构成。理解干预如何在此类系统中传播，需要超越直接效应并量化通过中介路径传递的影响。传统的中介分析在简单场景下虽有效，但难以扩展到实践中遇到的高维有向无环图（DAG），尤其在存在多重处理（treatments）和多重中介（mediators）相互作用时表现不足。本文提出了一种针对包含多重处理和中介的大规模因果DAG的可扩展中介分析框架。该方法系统地将总体效应分解为可解释的直接分量和间接分量。通过在履约中心物流（fulfillment center logistics）中的应用案例，我们展示了该框架的实用性——在此类场景中，复杂依赖关系与不可控因素常常掩盖根本原因，而本方法有助于揭示这些根因。

BibTeX

```
@article{2512.14764v1,
  title={Scaling Causal Mediation for Complex Systems: A Framework for Root Cause Analysis},
  author={Alessandro Casadei and Sreyoshi Bhaduri and Rohit Malshe and Pavan Mullapudi and Raj Ratan and Ankush Pole and Arkajit Rakshit},
  journal={arXiv preprint arXiv:2512.14764v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14764v1}
}
```

## [CAPE: Capability Achievement via Policy Execution](http://arxiv.org/abs/2512.14761v1)

#能力工程#可执行规范#验证与纠正

[PDF](https://arxiv.org/pdf/2512.14761v1)
[Abstract](http://arxiv.org/abs/2512.14761v1)

 LLM

 很推荐

### 中文摘要

现代人工智能系统缺乏一种表达并强制执行需求的机制。预训练产生了通用智能，后训练（如基于偏好的优化）调整了偏好，但两者都不能保证模型在具体上下文中可靠地满足明确的、依赖于上下文的约束。这一缺失的抽象解释了为何尽管在基准测试上表现优异，高度智能的模型在实际部署中仍经常失败。我们提出“能力工程”（Capability Engineering），即将需求系统化地转化为可执行规范，并训练模型使其在默认情况下满足这些规范。为实现这一做法，我们设计了CAPE（Capability Achievement via Policy Execution）协议，包含 Specify -> Verify -> Correct -> Train 的循环流程。CAPE 基于两项实证发现：第一，情境客观性（contextual objectivity），即在固定上下文后原本看似主观的属性变为客观（评审者一致性由 kappa = 0.42 提升至 kappa = 0.98）；第二，验证保真度随模型规模提升而显著提高（相关系数 r = 0.94），与之相对的基于偏好的意见一致度则在 30% 至 50% 的分歧范围内趋于平台化，无论计算量如何增加。在覆盖六个领域的 109,500 个样本上，CAPE 相较于标准的 DPO 将违规率降低了 81%（标准差小于 0.3%）。通过用可复用的规范替代逐例注释，CAPE 将成本降低了 5 到 20 倍，并将开发周期从数月缩短到数周。我们以 Apache 2.0 协议发布了 CAPE 协议、PredicateGraph 架构、CPL 规范语言及策略包，同时推出 CapabilityBench —— 一个面向社区贡献策略的模型评估公开登记库，推动评估从智能基准转向能力测量。

BibTeX

```
@article{2512.14761v1,
  title={CAPE: Capability Achievement via Policy Execution},
  author={David Ball},
  journal={arXiv preprint arXiv:2512.14761v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14761v1}
}
```

## [Revisiting the Reliability of Language Models in Instruction-Following](http://arxiv.org/abs/2512.14754v1)

#指令跟随可靠性#提示词鲁棒性#评测基准

[PDF](https://arxiv.org/pdf/2512.14754v1)
[Abstract](http://arxiv.org/abs/2512.14754v1)

 LLM

 很推荐

### 中文摘要

先进的大型语言模型在诸如 IFEval 等基准上的指令跟随准确率已接近上限。然而，这些看似优异的分数未必能在真实使用场景中保证服务可靠性：用户在实际交互中经常变换表述方式、上下文框架和任务陈述。本文研究一种面向细微差别的可靠性：即模型在面对传达相近意图但存在微妙差异的“表亲”提示（cousin prompts）时，能否保持一致的能力。为此，我们提出了新的评估指标 reliable@k，并设计了一个通过数据增强自动生成高质量表亲提示的流水线。基于此方法，我们构建了用于系统化评估的 IFEval++。在对 20 个闭源和 26 个开源 LLM 的测试中，我们发现当前模型在细微差别可靠性方面存在显著不足——在面对微妙提示修改时，性能最多下降达 61.8%。此外，我们对该现象进行了细致刻画，并探讨了三种可能的改进方案。我们的发现强调了面向细微差别的可靠性是实现更可依赖、值得信任的 LLM 行为的关键但尚未充分探索的方向。我们的代码与基准已公开： https://github.com/jianshuod/IFEval-pp。

BibTeX

```
@article{2512.14754v1,
  title={Revisiting the Reliability of Language Models in Instruction-Following},
  author={Jianshuo Dong and Yutong Zhang and Yan Liu and Zhenyu Zhong and Tao Wei and Chao Zhang and Han Qiu},
  journal={arXiv preprint arXiv:2512.14754v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14754v1}
}
```

## [Cyberswarm: a novel swarm intelligence algorithm inspired by cyber community dynamics](http://arxiv.org/abs/2512.14752v1)

#群体智能#推荐系统#超图与图嵌入

[PDF](https://arxiv.org/pdf/2512.14752v1)
[Abstract](http://arxiv.org/abs/2512.14752v1)

 推荐系统 / 图学习

 很推荐

### 中文摘要

推荐系统在动态适应不断演化的用户偏好和复杂社交网络中的交互方面面临挑战。传统方法往往未能充分考虑网络-社会系统内的复杂互动，也缺乏在多样化应用领域中泛化的灵活性，因此需要更具适应性和通用性的解决方案。本文提出了一种面向推荐系统的通用群体智能算法，旨在无缝适配不同应用场景，设计灵感来自社会心理学原理。该框架在动态超图结构中建模用户偏好与社区影响，结合基于中心性的重要特征提取与 Node2Vec 嵌入表示。偏好演化通过消息传递机制和分层图建模进行引导，从而实现对行为变化的实时适应。实验证明该算法在社交网络和内容发现等多种推荐任务上表现优越，关键指标如 Hit Rate (HR)、Mean Reciprocal Rank (MRR) 和 Normalized Discounted Cumulative Gain (NDCG) 在多个数据集上稳定优于基线方法。模型对动态环境的适应性使其能够提供具有上下文相关性且精确的推荐。所提算法通过连接个体偏好与社区影响，推动了推荐系统的进步；其通用设计可应用于社交图、个性化学习以及医学图谱等多种领域。本工作强调了将群体智能与网络动力学相结合以解决推荐系统中复杂优化问题的潜力。

BibTeX

```
@article{2512.14752v1,
  title={Cyberswarm: a novel swarm intelligence algorithm inspired by cyber community dynamics},
  author={Abdelsadeq Elfergany and Ammar Adl and Mohammed Kayed},
  journal={arXiv preprint arXiv:2512.14752v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14752v1}
}
```

## [A Critical Perspective on Finite Sample Conformal Prediction Theory in Medical Applications](http://arxiv.org/abs/2512.14727v1)

#保形预测#不确定性量化#医学影像分类

[PDF](https://arxiv.org/pdf/2512.14727v1)
[Abstract](http://arxiv.org/abs/2512.14727v1)

 医学机器学习（不确定性估计 / 保形预测）

 很推荐

### 中文摘要

机器学习正在改变医疗领域，但要做出安全的临床决策需要可靠的不确定性估计，而标准机器学习模型通常无法提供这种估计。保形预测（Conformal Prediction，CP）是一种常用工具，它能够将启发式的不确定性估计转化为具有统计保证的不确定性估计。CP 的工作原理是将机器学习模型的预测与一个校准样本结合，生成在任意期望置信水平下保证包含真实标签的预测集。人们常引用的一项优势是，CP 理论对任意大小的校准样本都成立，暗示即便只有很小的校准集也可获得在实践上有意义的统计保证。我们对这一承诺提出质疑：尽管理论保证在任意大小的校准集上成立，但这些保证在实际可用性上高度依赖于校准集的大小。该观察在医学领域尤为相关，因为医学数据通常稀缺，获得大规模校准集往往不可行。我们在一个医学影像分类任务的实证展示中验证并支持了我们的批判观点。

BibTeX

```
@article{2512.14727v1,
  title={A Critical Perspective on Finite Sample Conformal Prediction Theory in Medical Applications},
  author={Klaus-Rudolf Kladny and Bernhard Schölkopf and Lisa Koch and Christian F. Baumgartner and Michael Muehlebach},
  journal={arXiv preprint arXiv:2512.14727v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14727v1}
}
```

## [SEED: Spectral Entropy-Guided Evaluation of SpatialTemporal Dependencies for Multivariate Time Series Forecasting](http://arxiv.org/abs/2512.14718v1)

#谱熵#时空依赖建模#有符号图构造

[PDF](https://arxiv.org/pdf/2512.14718v1)
[Abstract](http://arxiv.org/abs/2512.14718v1)

 多变量时间序列预测

 很推荐

### 中文摘要

有效的多变量时间序列预测通常依赖于对复杂变量间依赖关系的精确建模。然而，现有基于注意力或图的方法存在三大问题： (a) 强烈的时间自相关常被无关变量干扰；(b) softmax 归一化会忽略并反转负相关关系；(c) 变量难以感知其时间位置信息。为此，我们提出了 SEED（Spectral Entropy-guided Evaluation）——一种用于时空依赖建模的谱熵引导评估框架。SEED 的关键创新是引入了依赖评估器（Dependency Evaluator），利用谱熵动态地对每个变量的时空依赖进行初步评估，使模型能够自适应地在通道独立（CI）与通道相关（CD）策略之间平衡。针对源于其他变量影响而非内生动力学的时间规律性，我们提出了基于谱熵的融合器（Spectral Entropy-based Fuser）以进一步精炼评估得到的依赖权重，有效分离该类成分。此外，为保留负相关信息，我们设计了有符号图构造器（Signed Graph Constructor），允许边权为带符号值，从而克服 softmax 的限制。最后，为帮助变量感知时间位置并构建更全面的空间特征，我们引入了上下文空间提取器（Context Spatial Extractor），通过局部上下文窗口提取空间特征。对来自不同应用领域的 12 个真实世界数据集的广泛实验表明，SEED 达到并超越了现有最先进方法，验证了其有效性与通用性。

BibTeX

```
@article{2512.14718v1,
  title={SEED: Spectral Entropy-Guided Evaluation of SpatialTemporal Dependencies for Multivariate Time Series Forecasting},
  author={Feng Xiong and Zongxia Xie and Yanru Sun and Haoyu Wang and Jianhong Lin},
  journal={arXiv preprint arXiv:2512.14718v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14718v1}
}
```

## [Improving Underwater Acoustic Classification Through Learnable Gabor Filter Convolution and Attention Mechanisms](http://arxiv.org/abs/2512.14714v1)

#可学习Gabor卷积#通道注意力（SE）#水下声学分类

[PDF](https://arxiv.org/pdf/2512.14714v1)
[Abstract](http://arxiv.org/abs/2512.14714v1)

 声学/音频信号处理

 很推荐

### 中文摘要

远程检测和分类水下声学目标对于环境监测和国防具有重要意义。然而，船舶辐射声与环境噪声的复杂性给准确的信号处理带来重大挑战。尽管近期机器学习的进展提升了分类精度，但数据集规模有限和实验缺乏标准化等问题仍制约着模型的泛化性与鲁棒性。本文提出了GSE ResNeXt，一种将可学习的Gabor卷积层与ResNeXt主干网络相结合并通过squeeze-and-excitation（SE）通道注意力机制增强的深度学习架构。Gabor滤波器作为二维自适应带通滤波器，扩展了特征通道的表示；与通道注意力的结合不仅提高了训练稳定性和收敛速度，还增强了模型提取判别性特征的能力。模型在三个复杂度递增的分类任务上进行了评估，特别考察了训练与测试数据在时间上的差异对性能的影响，结果表明船舶与传感器的距离显著影响分类效果。实验结果显示，GSE ResNeXt在分类性能上持续优于Xception、ResNet和MobileNetV2等基线模型；在稳定性与收敛性方面，将Gabor卷积加入模型初始层使训练时间减少了约28%。这些结果强调了信号处理策略在提升模型在不同环境条件下可靠性与泛化能力方面的重要性，尤其针对数据受限的水下声学分类场景。未来工作应着重缓解环境因素对输入信号的影响。

BibTeX

```
@article{2512.14714v1,
  title={Improving Underwater Acoustic Classification Through Learnable Gabor Filter Convolution and Attention Mechanisms},
  author={Lucas Cesar Ferreira Domingos and Russell Brinkworth and Paulo Eduardo Santos and Karl Sammut},
  journal={arXiv preprint arXiv:2512.14714v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14714v1}
}
```

## [SepsisSuite: Beyond Risk Stratification -- A Comparative Analysis of Deep Fusion vs. Expert Stacking for Prescriptive Sepsis AI](http://arxiv.org/abs/2512.14712v1)

#多模态融合#脓毒症预测与处方#混合专家模型(MoE)

[PDF](https://arxiv.org/pdf/2512.14712v1)
[Abstract](http://arxiv.org/abs/2512.14712v1)

 医疗AI（多模态融合与临床决策支持）

 很推荐

### 中文摘要

脓毒症约占全球重症监护入院的近20%，但传统预测模型常难以有效整合异构数据流，往往按模态孤立处理或依赖脆弱的早期融合方法。在本研究中，我们对端到端深度融合（End-to-End Deep Fusion）与面向情境的堆叠（Context-Aware Stacking）在脓毒症任务上的架构表现进行了严格比较。起初我们假设一种新颖的四模态分层门控注意力网络——SepsisFusionFormer——能够解决生命体征、文本与影像之间复杂的跨模态交互。然而，在 MIMIC-IV 数据集上的实验表明，在小规模抗生素队列（N ≈ 2,100）中，SepsisFusionFormer 出现了“注意力饥饿”现象，导致过拟合并在该任务上仅得到了 0.66 的 AUC。该出人意料的结果促成了 SepsisLateFusion 的设计——一种更精简的情境感知专家混合（Mixture-of-Experts, MoE）架构。我们将各模态视作相互正交的专家：作为“史料者”（静态特征）、“监测者”（时间序列）与“阅读者”（NLP），并通过 CatBoost 元学习器对其进行动态门控，从而在临床发病前 4 小时的预测任务上达到了 0.915 的 SOTA AUC。通过对决策阈值进行校准以保证临床安全性，相较于默认工作点将漏诊案例减少了 48%，从而为及时干预创造了真实的预防窗口（而非被动报警）。此外，在新提出的处方性多类别抗生素选择任务上，我们展示了四模态集成达到了最高的 0.72 AUC。所有模型均集成于 SepsisSuite——一个可部署的临床决策支持 Python 框架，并已在 GitHub 开源： https://github.com/RyanCartularo/SepsisSuite-Info

BibTeX

```
@article{2512.14712v1,
  title={SepsisSuite: Beyond Risk Stratification -- A Comparative Analysis of Deep Fusion vs. Expert Stacking for Prescriptive Sepsis AI},
  author={Ryan Cartularo},
  journal={arXiv preprint arXiv:2512.14712v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14712v1}
}
```

## [SCOPE: Prompt Evolution for Enhancing Agent Effectiveness](http://arxiv.org/abs/2512.15374v1)

#提示演化#上下文在线优化#智能体策略探索

[PDF](https://arxiv.org/pdf/2512.15374v1)
[Abstract](http://arxiv.org/abs/2512.15374v1)

 LLM

 很推荐

### 中文摘要

大型语言模型（LLM）代理越来越多地部署在会生成海量且动态上下文的环境中。然而，一个关键瓶颈依然存在：尽管代理可以访问这些上下文，其静态提示缺乏有效管理上下文的机制，导致反复出现的纠正与改进失败。为了解决这一能力缺口，我们提出了SCOPE（通过提示演化实现的自我演化上下文优化）。SCOPE 将上下文管理表述为一个在线优化问题，基于执行轨迹合成指导原则以自动演化代理的提示。我们提出了双流机制，用以在战术性特化（解决即时错误）与战略性通用性（演化长期原则）之间取得平衡；此外，引入了视角驱动的探索以最大化策略覆盖度，从而提高代理在给定任务中采用正确策略的概率。在 HLE 基准上的实验表明，SCOPE 在无人为干预的情况下将任务成功率从 14.23% 提升至 38.64%。我们已将代码公开发布于 https://github.com/JarvisPei/SCOPE。

BibTeX

```
@article{2512.15374v1,
  title={SCOPE: Prompt Evolution for Enhancing Agent Effectiveness},
  author={Zehua Pei and Hui-Ling Zhen and Shixiong Kai and Sinno Jialin Pan and Yunhe Wang and Mingxuan Yuan and Bei Yu},
  journal={arXiv preprint arXiv:2512.15374v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15374v1}
}
```

## [Adversarial versification in portuguese as a jailbreak operator in LLMs](http://arxiv.org/abs/2512.15353v1)

#对抗性诗歌#提示词越狱#语言模型安全

[PDF](https://arxiv.org/pdf/2512.15353v1)
[Abstract](http://arxiv.org/abs/2512.15353v1)

 LLM

 很推荐

### 中文摘要

近期研究表明，将提示改写为诗歌形式（versification）构成了一种对齐良好的大语言模型的高效对抗机制。题为“对抗性诗歌作为通用单轮越狱机制在大型语言模型中的作用”的研究显示，通常在散文中被拒绝的指令在转换为诗行后变为可执行，在MLCommons AILuminate衍生基准上导致多达18倍的安全失败率。人工创作的诗歌攻击成功率（ASR）约为62%，自动生成的版本约为43%，部分模型在单轮交互中超过90%的成功率。该效应具有结构性：采用RLHF、宪法式AI及混合流水线训练的系统在极小的语符形式变体下均表现出一致性退化。诗歌化使提示偏移到受监督稀疏的潜在区域，暴露出过度依赖表层模式的防护措施。这种表面上鲁棒与实际脆弱之间的脱节揭示了当前对齐机制的深层局限性。鉴于葡萄牙语具备高度形态句法复杂性、丰富的格律-韵律传统且拥有超过2.5亿使用者，当前缺乏对葡萄牙语的评估是一个关键空白。实验协议需要对扫韵、格律和韵律变体进行参数化，以测试针对葡语特有模式的脆弱性，这些模式目前被忽视。

BibTeX

```
@article{2512.15353v1,
  title={Adversarial versification in portuguese as a jailbreak operator in LLMs},
  author={Joao Queiroz},
  journal={arXiv preprint arXiv:2512.15353v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15353v1}
}
```

## [Leveraging Foundational Models and Simple Fusion for Multi-modal Physiological Signal Analysis](http://arxiv.org/abs/2512.15250v1)

#多模态生理信号分析#基础模型预训练#ECG/EEG 融合与情感识别

[PDF](https://arxiv.org/pdf/2512.15250v1)
[Abstract](http://arxiv.org/abs/2512.15250v1)

 多模态生理信号分析（ECG/EEG）

 很推荐

### 中文摘要

生理信号（例如心电图 ECG 和脑电图 EEG）能为人类健康与认知提供互补性信息，但由于多模态标注数据稀缺及模态间差异，进行多模态融合具有挑战性。本文中，我们将 CBraMod 编码器用于大规模的 ECG 自监督预训练，并提出一种双重掩码策略以同时捕捉导联内和导联间的依赖关系。为了解决上述挑战，我们对 EEG 采用预训练的 CBraMod 编码器，并对 ECG 预训练一个对称的编码器，使每种模态都具备丰富的基础表示。随后通过简单的嵌入拼接将这些表示融合，交由分类头学习跨模态交互，从而在有限的多模态监督下仍能实现有效的下游学习。在情感识别任务上的评估表明，本方法能取得接近最先进的性能，说明精心设计的生理信号编码器即便配合简单的融合策略，也能显著提升下游表现。研究结果凸显了基础模型方法在利用生理信号整体性方面的潜力，有望为医疗与情感计算领域提供可扩展、标注高效且具泛化能力的解决方案。

BibTeX

```
@article{2512.15250v1,
  title={Leveraging Foundational Models and Simple Fusion for Multi-modal Physiological Signal Analysis},
  author={Youssef Ghallab and Omar Iraqy and Mohamed Kandil and Mohamed Ashraf and Saadeldine Eletter and Morougue Ghazal and Ayman Khalafallah and Nagwa El-Makky},
  journal={arXiv preprint arXiv:2512.15250v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15250v1}
}
```

## [CangLing-KnowFlow: A Unified Knowledge-and-Flow-fused Agent for Comprehensive Remote Sensing Applications](http://arxiv.org/abs/2512.15231v1)

#遥感自动化#知识与工作流融合智能体#动态恢复与演化记忆

[PDF](https://arxiv.org/pdf/2512.15231v1)
[Abstract](http://arxiv.org/abs/2512.15231v1)

 LLM（遥感/地球观测智能体）

 很推荐

### 中文摘要

自动化且智能地处理大规模遥感（RS）数据集对地球观测（EO）至关重要。现有的自动化系统通常针对具体任务，缺乏一个能够跨越数据预处理到高级解译等多样化、端到端工作流的统一框架。为填补这一空白，本文提出了 CangLing-KnowFlow，一种将程序化知识库（Procedural Knowledge Base, PKB）、动态工作流调整（Dynamic Workflow Adjustment）和演化记忆模块（Evolutionary Memory Module）融合的统一智能体框架。PKB 包含 1,008 个经专家验证、覆盖 162 个实际遥感任务的工作流案例，可用于引导规划并显著降低通用智能体常见的幻觉问题。在运行时发生失败时，动态工作流调整能够自主诊断并重规划恢复策略；演化记忆模块则从这些事件中持续学习，迭代增强智能体的知识与性能。三者协同使 CangLing-KnowFlow 能够在多样且复杂的任务中实现自适应、可学习与可靠运行。我们在新构建的 KnowFlow-Bench（包含 324 个受真实应用启发的工作流）上对该框架进行了评估，测试涵盖从开源到商业的 13 种主流大语言模型（LLM）骨干。实验结果表明，在所有复杂任务上，CangLing-KnowFlow 在任务成功率上均至少比 Reflexion 基线高 4%。作为该新兴领域中最为全面的验证之一，本研究展示了将专家知识（Knowledge）引入可适应且可验证流程（Flow）以构建鲁棒、高效、可扩展地球观测自动化解决方案的巨大潜力。

BibTeX

```
@article{2512.15231v1,
  title={CangLing-KnowFlow: A Unified Knowledge-and-Flow-fused Agent for Comprehensive Remote Sensing Applications},
  author={Zhengchao Chen and Haoran Wang and Jing Yao and Pedram Ghamisi and Jun Zhou and Peter M. Atkinson and Bing Zhang},
  journal={arXiv preprint arXiv:2512.15231v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15231v1}
}
```

## [Yes-MT's Submission to the Low-Resource Indic Language Translation Shared Task in WMT 2024](http://arxiv.org/abs/2512.15226v1)

#低资源机器翻译#大模型微调#印度语翻译

[PDF](https://arxiv.org/pdf/2512.15226v1)
[Abstract](http://arxiv.org/abs/2512.15226v1)

 NLP (Machine Translation)

 很推荐

### 中文摘要

本文介绍了 Yes-MT 团队为 WMT 2024 低资源印度语种翻译共享任务（Pakray et al., 2024）提交的系统，聚焦于英语与阿萨姆语（Assamese）、米佐语（Mizo）、卡西语（Khasi）和曼尼普尔语（Manipuri）之间的翻译。实验比较了多种方法：在多语种和单语种设置下微调预训练模型（如 mT5 和 IndicBart）、对 IndicTrans2 进行 LoRA 微调、利用大规模语言模型（如 Llama 3 和 Mixtral 8x7b）进行零样本与少样本提示、对 Llama 3 采用 LoRA 监督微调，以及从头训练 Transformer 模型等。结果在 WMT23 低资源印度语种翻译任务的测试集上，采用 SacreBLEU 和 CHRF 指标进行评估。研究指出低资源翻译面临的挑战，同时表明大规模语言模型在该类任务上具有潜力，尤其在进行微调时表现突出。

BibTeX

```
@article{2512.15226v1,
  title={Yes-MT's Submission to the Low-Resource Indic Language Translation Shared Task in WMT 2024},
  author={Yash Bhaskar and Parameswari Krishnamurthy},
  journal={arXiv preprint arXiv:2512.15226v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15226v1}
}
```

## [A Decision-Theoretic Approach for Managing Misalignment](http://arxiv.org/abs/2512.15584v1)

#价值对齐#决策理论#委托决策

[PDF](https://arxiv.org/pdf/2512.15584v1)
[Abstract](http://arxiv.org/abs/2512.15584v1)

 AI Alignment (决策理论)

 很推荐

### 中文摘要

何时应将决策委托给人工智能系统？尽管价值对齐研究提出了塑造 AI 价值的技术，但在不确定性下如何判断不完美的对齐是否足以支持委托的问题上关注较少。我们认为，理性的委托需要在代理的价值（错配程度）、认识准确性（epistemic accuracy）和其可及行动范围（reach）之间权衡。本文引入了一个形式化的决策理论框架，用以精确分析这一权衡，并对委托人（principal）对这些因素的不确定性进行建模。我们的分析揭示了两类委托情形的显著差异：其一，普适性委托（将任何问题都信任代理）要求近乎完美的价值对齐和完全的认识信任，而现实中很少满足；其二，我们证明了情境特定的委托即便在存在显著错配时也可能是最优的：代理更高的准确性或更广的可及性可能带来总体上更好的决策机会，使得从期望值角度看委托是合理的。为此，我们提出了一个新的评分框架以量化这种事前（ex ante）决策。总体而言，本工作提供了一种原则性方法，帮助在具体情境下判断 AI 是否“足够对齐”，从而将关注点从追求完美对齐转向在不确定性下管理委托的风险与收益。

BibTeX

```
@article{2512.15584v1,
  title={A Decision-Theoretic Approach for Managing Misalignment},
  author={Daniel A. Herrmann and Abinav Chari and Isabelle Qian and Sree Sharvesh and B. A. Levinstein},
  journal={arXiv preprint arXiv:2512.15584v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15584v1}
}
```

## [Nemotron-Math: Efficient Long-Context Distillation of Mathematical Reasoning from Multi-Mode Supervision](http://arxiv.org/abs/2512.15489v1)

#数学推理数据集#长上下文训练#工具集成推理（TIR）

[PDF](https://arxiv.org/pdf/2512.15489v1)
[Abstract](http://arxiv.org/abs/2512.15489v1)

 LLM

 很推荐

### 中文摘要

高质量的数学推理监督需要多样的推理风格、长形式的推理轨迹以及有效的工具集成，而现有数据集在这些方面都存在局限。我们利用 gpt-oss-120b 的多模式生成能力，提出了 Nemotron-Math，这是一个大规模的数学推理数据集，包含 750 万条解题轨迹，覆盖高、中、低三种推理模式，每种模式均提供带与不带 Python 工具集成推理（TIR）的版本。该数据集整合了 85K 条精挑细选的 AoPS 竞赛题与 262K 条社区来源的 StackExchange-Math 问题，将结构化的竞赛任务与多样的现实数学查询结合在一起。我们进行了受控评估以验证数据集质量。Nemotron-Math 在匹配的 AoPS 题目上始终优于原始的 OpenMathReasoning。引入 StackExchange-Math 显著提升了模型的鲁棒性和泛化能力，尤其在 HLE-Math 上效果明显，同时在数学竞赛基准上的准确性得以保持。为支持高效的长上下文训练，我们提出了顺序分桶（sequential bucketed）策略，使得 128K 上下文长度的微调速度提升约 2–3×，且几乎不损失精度。总体而言，Nemotron-Math 促成了最先进的性能表现，包括在启用 Python TIR 时于 AIME 2024 和 2025 上实现 16 票多数决（maj@16）100% 的准确率。

BibTeX

```
@article{2512.15489v1,
  title={Nemotron-Math: Efficient Long-Context Distillation of Mathematical Reasoning from Multi-Mode Supervision},
  author={Wei Du and Shubham Toshniwal and Branislav Kisacanin and Sadegh Mahdavi and Ivan Moshkov and George Armstrong and Stephen Ge and Edgar Minasyan and Feng Chen and Igor Gitman},
  journal={arXiv preprint arXiv:2512.15489v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15489v1}
}
```

## [Can LLMs Guide Their Own Exploration? Gradient-Guided Reinforcement Learning for LLM Reasoning](http://arxiv.org/abs/2512.15687v1)

#梯度引导探索#大语言模型强化学习#推理能力提升

[PDF](https://arxiv.org/pdf/2512.15687v1)
[Abstract](http://arxiv.org/abs/2512.15687v1)

 LLM (强化学习)

 很推荐

### 中文摘要

强化学习已成为增强大语言模型推理能力的重要手段，但当前的探索机制与模型实际的学习方式存在根本性不匹配。熵奖励和外部语义比较器虽然鼓励表面上的多样性，但无法保证采样轨迹在影响优化的更新方向上具有差异性。我们提出了G2RL，一种梯度引导的强化学习框架，其中探索不是由外部启发式方法驱动，而是由模型自身的一阶更新几何决定。对于每个输出响应，G2RL 从模型最后一层的敏感性构建序列级特征，这些特征可通过一次标准前向传播以极低代价获得；通过在采样组内比较这些特征来衡量每条轨迹将如何重塑策略。引入新梯度方向的轨迹会获得一个有界的乘性奖励放大因子，而冗余或偏离流形的更新被弱化，从而产生一种自指的探索信号，该信号天然与PPO风格的稳定性和KL约束相一致。在数学与通用推理基准（MATH500、AMC、AIME24、AIME25、GPQA、MMLUpro）上对Qwen3 base 1.7B与4B模型的实验表明，G2RL 相较于基于熵的GRPO和外部嵌入方法，在 pass@1、maj@16 及 pass@k 等指标上持续带来提升。对所诱导几何的分析显示，G2RL 将探索扩展到更多正交且常常相反的梯度方向，同时保持语义一致性，揭示了策略自身的更新空间为指导大语言模型强化学习中的探索提供了更真实且更有效的基础。

BibTeX

```
@article{2512.15687v1,
  title={Can LLMs Guide Their Own Exploration? Gradient-Guided Reinforcement Learning for LLM Reasoning},
  author={Zhenwen Liang and Sidi Lu and Wenhao Yu and Kishan Panaganti and Yujun Zhou and Haitao Mi and Dong Yu},
  journal={arXiv preprint arXiv:2512.15687v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15687v1}
}
```

## [PPSEBM: An Energy-Based Model with Progressive Parameter Selection for Continual Learning](http://arxiv.org/abs/2512.15658v1)

#持续学习#能量基模型#渐进参数选择

[PDF](https://arxiv.org/pdf/2512.15658v1)
[Abstract](http://arxiv.org/abs/2512.15658v1)

 NLP

 很推荐

### 中文摘要

持续学习仍然是机器学习中的一项基础性挑战，要求模型能够从任务流中学习而不遗忘先前获得的知识。该情景中的一个主要障碍是灾难性遗忘，即在学习新任务时模型在早期任务上的性能下降。本文提出了PPSEBM，一种将能量基模型（EBM）与渐进参数选择（PPS）相结合的新框架，以有效应对自然语言处理任务中的灾难性遗忘。在PPSEBM中，渐进参数选择为每个新任务分配独立的任务专用参数，而能量基模型则从先前任务中生成具有代表性的伪样本。所生成的伪样本主动为参数选择过程提供信息与引导，从而增强模型在适应新任务的同时保留历史知识的能力。在多个NLP基准上的实验结果表明，PPSEBM优于最先进的持续学习方法，提供了一种有前景且鲁棒的缓解灾难性遗忘的解决方案。

BibTeX

```
@article{2512.15658v1,
  title={PPSEBM: An Energy-Based Model with Progressive Parameter Selection for Continual Learning},
  author={Xiaodi Li and Dingcheng Li and Rujun Gao and Mahmoud Zamani and Feng Mi and Latifur Khan},
  journal={arXiv preprint arXiv:2512.15658v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15658v1}
}
```

## [How Much is Too Much? Exploring LoRA Rank Trade-offs for Retaining Knowledge and Domain Robustness](http://arxiv.org/abs/2512.15634v1)

#低秩适配（LoRA）#参数高效微调#泛化与任务遗忘

[PDF](https://arxiv.org/pdf/2512.15634v1)
[Abstract](http://arxiv.org/abs/2512.15634v1)

 LLM

 很推荐

### 中文摘要

大型语言模型越来越多地通过微调来适配下游任务。全参数监督微调（SFT）和参数高效微调（PEFT），例如低秩适配（LoRA），是两类主流方法。尽管PEFT因计算与存储效率被广泛采用，但其配置（如秩值）在下游问答任务和模型泛化方面的影响尚未得到充分研究。在本工作中，我们在多个推理与记忆/回忆数据集上进行了全面评估，通过对秩值进行扫描定量分析SFT与PEFT之间的权衡。我们还比较了PEFT与SFT在域内与域外适配下的准确性，揭示了不同的泛化行为以及任务相关的遗忘模式。实验表明，在特定的秩值设置下，LoRA在性能上能够与SFT相抗衡，且在某些推理任务上表现更优。除此之外，我们通过谱特征和逐层注意力结构的分析，探讨了表征漂移与注意力模式的结构性变化，从而为理解PEFT引起的内部表示变动提供了见解。

BibTeX

```
@article{2512.15634v1,
  title={How Much is Too Much? Exploring LoRA Rank Trade-offs for Retaining Knowledge and Domain Robustness},
  author={Darshita Rathore and Vineet Kumar and Chetna Bansal and Anindya Moitra},
  journal={arXiv preprint arXiv:2512.15634v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15634v1}
}
```

## [How Smoothing is N-simplicial Attention?](http://arxiv.org/abs/2512.15600v1)

#N-单纯形注意力#高阶注意力交互#过度平滑与Lipschitz分析

[PDF](https://arxiv.org/pdf/2512.15600v1)
[Abstract](http://arxiv.org/abs/2512.15600v1)

 LLM

 很推荐

### 中文摘要

从纯粹的多层感知机（MLP）到在每一层引入可学习的图消息传递机制（尽管存在计算开销，如图注意力网络 GAT 或 Transformer），这一转变已成为取得最先进结果的基础。为进一步推进这一方向，本文提出了 N-单纯形注意力，将成对令牌相似度推广到高阶交互，并将其适配于旋转位置嵌入（RoPE）。为应对由此带来的复杂度增加，我们设计了一种成本高效的单纯形选择机制，使模型能够将计算资源集中在对任务更敏感的交互上。除了这些核心机制外，我们还通过推导 Lipschitz 上界研究了 N-单纯形注意力的平滑性，并证明尽管该方法将注意力消息传递扩展到高阶交互，但其自身仍会遭遇过度平滑问题。

BibTeX

```
@article{2512.15600v1,
  title={How Smoothing is N-simplicial Attention?},
  author={Alexandre Dussolle and Pietro Liò},
  journal={arXiv preprint arXiv:2512.15600v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15600v1}
}
```

## [BashArena: A Control Setting for Highly Privileged AI Agents](http://arxiv.org/abs/2512.15688v1)

#AI安全#控制基准#LLM 攻防评估

[PDF](https://arxiv.org/pdf/2512.15688v1)
[Abstract](http://arxiv.org/abs/2512.15688v1)

 AI安全 (AI控制与LLM)

 很推荐

### 中文摘要

未来的人工智能代理可能会以更高的权限自主运行。如果这些代理与人类目标不对齐，它们可能滥用权限造成严重损害。人工智能控制领域旨在开发在保持有用性的同时，降低不对齐 AI 造成损害的技术。我们提出 BashArena，一种用于在安全关键环境中研究 AI 控制技术的设置。BashArena 包含 637 个在复杂、逼真环境下的 Linux 系统管理与基础设施工程任务，并为红队设定了四项破坏性目标：运行恶意软件、外泄机密、提权和禁用防火墙。我们评估了多款前沿的大型模型在完成任务、在不被发现的情况下实施破坏以及检测破坏尝试方面的能力。Claude Sonnet 4.5 在 26% 的情况下成功执行破坏并规避由 GPT-4.1 mini 进行的监控，同时轨迹级误报率为 4%。我们的发现为在 BashArena 中设计更有效的控制协议提供了基线。我们以 ControlArena 设置公开了该数据集，并共享了任务生成流水线。

BibTeX

```
@article{2512.15688v1,
  title={BashArena: A Control Setting for Highly Privileged AI Agents},
  author={Adam Kaufman and James Lucassen and Tyler Tracy and Cody Rushing and Aryan Bhatt},
  journal={arXiv preprint arXiv:2512.15688v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15688v1}
}
```

## [Vision-based module for accurately reading linear scales in a laboratory](http://arxiv.org/abs/2512.15327v1)

#实验室自动化#刻度读数#计算机视觉

[PDF](https://arxiv.org/pdf/2512.15327v1)
[Abstract](http://arxiv.org/abs/2512.15327v1)

 计算机视觉 (CV)

 推荐

### 中文摘要

视觉模型的能力和数量正在迅速增长，这些模型如今能够以很高的精度完成目标检测、图像分类、实例分割等多种任务。然而，能够像人类目测那样从图像中获取精确定量测量的模型仍然较少。为了实现机器人在实验室环境中的完全自主操作，机器人需要具备诸如导航、物体搬运、样品制备等基础技能，以便在非结构化环境中达到类人能力。另一个重要能力是读取仪器和器具上的刻度读数。本文尝试模拟人类的读数方法来读取线性刻度，选取注射器和量筒的液面读数作为测试用例。对于随机朝向的注射器，首先进行方向校正变换。为提高系统效率与鲁棒性，感兴趣区域被收缩为仅包含线性刻度的图像部分。随后提取一系列特征，如主要刻度标记、对应数字以及液位指示位置，并据此计算最终读数。将本系统得到的读数与人工读取值进行比较，结果显示二者高度一致。

BibTeX

```
@article{2512.15327v1,
  title={Vision-based module for accurately reading linear scales in a laboratory},
  author={Parvesh Saini and Soumyadipta Maiti and Beena Rai},
  journal={arXiv preprint arXiv:2512.15327v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15327v1}
}
```

## [Managing Ambiguity: A Proof of Concept of Human-AI Symbiotic Sense-making based on Quantum-Inspired Cognitive Mechanism of Rogue Variable Detection](http://arxiv.org/abs/2512.15325v1)

#模糊性管理#人机共生智能#量子启发式流氓变量检测

[PDF](https://arxiv.org/pdf/2512.15325v1)
[Abstract](http://arxiv.org/abs/2512.15325v1)

 Human-AI Interaction (Decision Support)

 推荐

### 中文摘要

组织日益在波动、不确定、复杂与模糊（VUCA）的环境中运作，此类环境中变动的早期信号往往表现为微弱且碎片化的迹象。尽管人工智能（AI）被广泛用于支持管理决策，但多数基于AI的系统仍以预测与决策解决为优化目标，在高度模糊的情境下容易导致过早的解读闭合（interpretive closure）。这一现象在管理学中留下了一处空白：即如何在模糊尚未演化为错误或危机之前，通过人机系统负责任地管理模糊性。本文通过提出LAIZA人机增强共生智能系统的概念验证（PoC）及其专利流程——量子启发式流氓变量建模（QRVM）、人类在环去相干（Human-in-the-Loop Decoherence）和集体认知推理（Collective Cognitive Inference）——来回应该空白。该机制将模糊性操作化为一种未塌缩的认知态，检测持续性的解释性崩解（流氓变量），并在自主推理不可靠时触发结构化的人类在环澄清环节。实证方面，文章基于2025年在AI研发环境中为期三个月的案例研究，研究对象围绕员工意图与知识产权边界的长期模糊性。研究发现，通过保留多元解释视角，组织得以提前进行情景化准备（包括主动的专利保护），从而在模糊性塌缩时实施决定性且不致扰动的行动。该研究通过将模糊性重塑为一等公民的理论构建，展示了人机共生在VUCA环境中提升组织韧性的实际价值。

BibTeX

```
@article{2512.15325v1,
  title={Managing Ambiguity: A Proof of Concept of Human-AI Symbiotic Sense-making based on Quantum-Inspired Cognitive Mechanism of Rogue Variable Detection},
  author={Agnieszka Bienkowska and Jacek Malecki and Alexander Mathiesen-Ohman and Katarzyna Tworek},
  journal={arXiv preprint arXiv:2512.15325v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15325v1}
}
```

## [Tourists Profiling by Interest Analysis](http://arxiv.org/abs/2512.14704v1)

#旅游行为分析#数字足迹#景点网络

[PDF](https://arxiv.org/pdf/2512.14704v1)
[Abstract](http://arxiv.org/abs/2512.14704v1)

 数据挖掘/用户画像

 推荐

### 中文摘要

随着近期的数字化革命，旅游者行为的分析及相关研究领域发生了深刻变化。现在可以更方便地利用旅游者在出行过程中留下的数字足迹来研究其行为。现有关于旅游的研究多侧重于对数字足迹的定量分析以得出结论。本文提出了一项既关注数字足迹定量特征又关注定性特征的研究，以便更全面地理解支配旅游者行为的动态机制，尤其是与景点网络相关的行为规律。

BibTeX

```
@article{2512.14704v1,
  title={Tourists Profiling by Interest Analysis},
  author={Sonia Djebali and Quentin Gabot and Guillaume Guerard},
  journal={arXiv preprint arXiv:2512.14704v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14704v1}
}
```

## [Algorithmic Criminal Liability in Greenwashing: Comparing India, United States, and European Union](http://arxiv.org/abs/2512.12837v1)

#AI绿洗#算法可责性#公司刑事责任

[PDF](https://arxiv.org/pdf/2512.12837v1)
[Abstract](http://arxiv.org/abs/2512.12837v1)

 AI治理/法律（AI法学）

 推荐

### 中文摘要

由人工智能驱动的“绿洗”已成为企业可持续治理中的一项隐蔽挑战，增加了环境信息披露的不透明性并削弱了监管监督。本文对印度、美国与欧盟中与AI介导的绿洗相关的刑事责任进行了比较法分析，揭示了在欺骗性陈述源自算法系统时归责存在的教义性空白。现行法规表现出以人为本的偏向，通常将责任建立在可证实的人类意图之上，从而使其难以有效应对算法性欺骗。研究指出，在法理适应方面存在关键缺口，现有的反欺诈法条在面对AI生成的失实陈述时显得过时。采用教义法学方法，系统梳理了司法判例与成文法规，得出有关扩展公司刑事责任的若干结论。研究结果强调严格责任模式的可行性、为AI问责重构治理框架的必要性以及在ESG制度下对算法尽职调查的强制性要求。比较分析显示各司法辖区存在差异，其中欧盟的《公司可持续尽职调查指令》（CSDDD）可能成为一种跨国示范性模式。本文通过倡导将算法风险评估与法人地位构思相结合的混合归责框架，为AI伦理与环境法学贡献了思路，以确保算法的不透明性不成为规避责任的屏障。

BibTeX

```
@article{2512.12837v1,
  title={Algorithmic Criminal Liability in Greenwashing: Comparing India, United States, and European Union},
  author={Sahibpreet Singh and Manjit Singh},
  journal={arXiv preprint arXiv:2512.12837v1},
  year={2025},
  url={http://arxiv.org/abs/2512.12837v1}
}
```

## [A Clustering-Based Variable Ordering Framework for Relaxed Decision Diagrams for Maximum Weighted Independent Set Problem](http://arxiv.org/abs/2512.15198v1)

#聚类变量排序#松弛决策图#最大加权独立集

[PDF](https://arxiv.org/pdf/2512.15198v1)
[Abstract](http://arxiv.org/abs/2512.15198v1)

 离散优化（组合优化 / 决策图）

 推荐

### 中文摘要

高效的精确离散优化算法在很大程度上依赖于强的原始界与对偶界。松弛决策图（Relaxed Decision Diagrams, DDs）通过节点合并对可行解空间进行紧凑的过度近似，为构造对偶界提供了一种通用机制。然而，这类松弛图的质量（即所得对偶界的紧致性）对变量顺序和编译过程中执行的合并决策高度敏感。尽管动态变量排序启发式能够有效收紧界，但在对全体未定变量全局评估时常伴随较高的计算开销。为缓解这一权衡，本文提出了一种基于聚类的变量排序框架：首先将变量划分为若干簇，然后利用这种结构化分解来引导排序过程，从而显著缩小启发式搜索空间。在该框架下，作者研究了两类策略：簇到簇（Cluster-to-Cluster），按簇顺序处理并采用问题特定的聚合准则（例如在最大加权独立集问题中使用顶点权重累积）；以及抽取并排序（Pick-and-Sort），该方法在各簇间迭代地选择并排序代表性变量，以在局部多样性和启发式指引间取得平衡。随后，针对MWISP给出关于DD规模增长的一些理论结果，并基于此提出两种簇数设置策略。将这些方法嵌入基于DD的分支限界算法并在MWISP基准实例上评估，结果表明相比标准的动态变量排序基线，所提方法在多组实例上持续降低了计算开销。

BibTeX

```
@article{2512.15198v1,
  title={A Clustering-Based Variable Ordering Framework for Relaxed Decision Diagrams for Maximum Weighted Independent Set Problem},
  author={Mohsen Nafar and Michael Römer and Lin Xie},
  journal={arXiv preprint arXiv:2512.15198v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15198v1}
}
```

## [Governing rapid technological change: Policy Delphi on the future of European AI governance](http://arxiv.org/abs/2512.15196v1)

#AI治理#Policy Delphi#前瞻性监管

[PDF](https://arxiv.org/pdf/2512.15196v1)
[Abstract](http://arxiv.org/abs/2512.15196v1)

 AI治理

 推荐

### 中文摘要

人工智能（AI）的快速进展为试图治理该技术的政策制定者带来了独特挑战。在此背景下，Delphi 方法已成为识别未来学与前瞻研究领域专家对新兴技术问题中共识与分歧的一种成熟手段。本文目的有两方面：一是考察专家眼中欧洲 AI 治理发展中的关键张力，二是基于这些见解反思 Policy Delphi 方法在为诸如 AI 之类新兴技术提供前瞻性治理建议方面的能力。分析基于 2024 年中期对欧洲政策制定者、研究人员和非政府组织开展的两轮 Policy Delphi 研究结果。研究表明，Policy Delphi 在揭示关于欧洲 AI 治理的多元观点方面十分有用，并指出一个趋向共识：具有前瞻性的 AI 监管更可能依赖于法律的实际执行与落实，而非其技术细节或适用范围的设计。此外，研究识别出 AI 治理中的“理想性—可行性差距”：诸如增强公民参与等理想政策方向被认为在可实现性和发生概率上相对较低。这凸显了期望中的监管监督与监管在实践中跟上技术变化速度之间的紧张关系。

BibTeX

```
@article{2512.15196v1,
  title={Governing rapid technological change: Policy Delphi on the future of European AI governance},
  author={Atte Ojanen and Johannes Anttila and Thilo H. K. Thelitz and Anna Bjork},
  journal={arXiv preprint arXiv:2512.15196v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15196v1}
}
```

## [MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber](http://arxiv.org/abs/2512.14846v1)

#多智能体LLM#实时网络防御#入侵检测系统

[PDF](https://arxiv.org/pdf/2512.14846v1)
[Abstract](http://arxiv.org/abs/2512.14846v1)

 LLM / 网络安全

 推荐

### 中文摘要

传统的集中式安全工具常常无法发现具有自适应性和多向量特征的攻击。本文提出了多智能体大语言模型网络防御框架（MALCDF），在该实用系统中四个大语言模型代理——检测（Detection）、情报（Intelligence）、响应（Response）和分析（Analysis）——实时协同工作。代理通过一个安全通信层（SCL）交换加密且与语义本体对齐的消息，并生成便于审计的输出（例如与 MITRE ATT&CK 的映射）。评估中我们保持测试简单且一致：所有报告的度量均来自同一条基于 CICIDS2017 特征模式的 50 条实时流记录。CICIDS2017 用于字段/模式配置并用于训练一个实际的机器学习基线。该基线为在 CICIDS2017 子集上训练、在 50 条流上测试且训练测试样本互不重叠的轻量级随机森林入侵检测系统（LRF-IDS）。实验结果表明，MALCDF 达到 90.0% 的检测准确率、85.7% 的 F1 分数、9.1% 的误报率，单事件平均延迟为 6.8 秒。在准确率上，它优于轻量级 ML-IDS 基线和单一 LLM 方案，同时保持端到端输出的一致性。总体而言，该实作表明通过协调简单的 LLM 代理并采用安全且与本体对齐的消息机制，可提升实用的实时网络防御能力。

BibTeX

```
@article{2512.14846v1,
  title={MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber},
  author={Arth Bhardwaj and Sia Godika and Yuvam Loonker},
  journal={arXiv preprint arXiv:2512.14846v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14846v1}
}
```

## [Autonomous Source Knowledge Selection in Multi-Domain Adaptation](http://arxiv.org/abs/2512.14710v1)

#多域自适应#源知识选择#伪标签增强

[PDF](https://arxiv.org/pdf/2512.14710v1)
[Abstract](http://arxiv.org/abs/2512.14710v1)

 域自适应 / 迁移学习

 推荐

### 中文摘要

无监督多域自适应在迁移学习中具有重要作用，通过利用来自多个源域的丰富信息来解决来自无标注目标域的任务。然而，多个源域往往包含大量冗余或无关的信息，尤其在海量源域情形下，这些信息会损害迁移性能。因此，迫切需要有效策略从海量源域中识别并选择最具可迁移性的知识以应对目标任务。为此，本文提出了一种名为 Autonomous Source Knowledge Selection（AutoS）的多域自适应方法，用于自主选择源训练样本和源模型，使目标任务的预测能够依赖更相关且更具可迁移性的源信息。该方法在训练过程中采用基于密度的选择策略来挑选源样本，并用于确定哪些源模型应参与目标预测。同时，基于预训练多模态模型构建的伪标签增强模块被引入，以缓解目标域标签噪声并提升自监督效果。在真实数据集上的实验结果表明，所提方法表现优越。

BibTeX

```
@article{2512.14710v1,
  title={Autonomous Source Knowledge Selection in Multi-Domain Adaptation},
  author={Keqiuyin Li and Jie Lu and Hua Zuo and Guangquan Zhang},
  journal={arXiv preprint arXiv:2512.14710v1},
  year={2025},
  url={http://arxiv.org/abs/2512.14710v1}
}
```

## [PMMD: A pose-guided multi-view multi-modal diffusion for person generation](http://arxiv.org/abs/2512.15069v1)

#多模态扩散模型#姿态引导人像生成#多视图融合

[PDF](https://arxiv.org/pdf/2512.15069v1)
[Abstract](http://arxiv.org/abs/2512.15069v1)

 CV

 推荐

### 中文摘要

生成具有可控姿态和外观一致性的人像图像对于虚拟试衣、图像编辑及数字人创建等应用至关重要。现有方法常受遮挡、服饰风格漂移和姿态错位等问题影响。我们提出了Pose-guided Multi-view Multimodal Diffusion（PMMD），一种基于扩散的框架，可在多视角参考图像、姿态图和文本提示的条件下合成照片级真实的人像图像。一个多模态编码器对视觉视图、姿态特征和语义描述进行联合建模，从而减少跨模态差异并提高身份一致性。我们进一步设计了ResCVA模块以在保持全局结构的同时增强局部细节，并引入跨模态融合模块，将图像语义与文本信息贯穿于去噪流水线。基于DeepFashion MultiModal数据集的实验表明，PMMD在一致性、细节保留和可控性方面均优于代表性基线方法。项目页面与代码已在https://github.com/ZANMANGLOOPYE/PMMD 上公开。

BibTeX

```
@article{2512.15069v1,
  title={PMMD: A pose-guided multi-view multi-modal diffusion for person generation},
  author={Ziyu Shang and Haoran Liu and Rongchao Zhang and Zhiqian Wei and Tongtong Feng},
  journal={arXiv preprint arXiv:2512.15069v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15069v1}
}
```

## [BERT and CNN integrated Neural Collaborative Filtering for Recommender Systems](http://arxiv.org/abs/2512.15526v1)

#神经协同过滤#多模态推荐#BERT与CNN融合

[PDF](https://arxiv.org/pdf/2512.15526v1)
[Abstract](http://arxiv.org/abs/2512.15526v1)

 推荐系统

 推荐

### 中文摘要

每天都有大量用户上网以满足不同需求。网站所有者通过用户与网站内容或物品的交互获得收益。一个健壮的推荐系统可以通过根据用户的个性化偏好推荐物品来提升用户与网站的交互。本实验提出了一种将 BERT 与 CNN 集成到神经协同过滤（NCF）中的推荐模型。所提模型以用户和物品的特征为输入，挖掘用户兴趣，能够处理数值、类别及图像数据，从输入中提取潜在特征。该模型在 MovieLens 数据集的小样本上训练并验证，训练 25 个 epoch。为了比较，本文还在相同数据集上训练并验证了简单的 NCF 和基于 BERT 的 NCF 两个基线模型，结果表明所提模型优于这两种基线。所提模型在 MovieLens 数据集（799 名用户）上的实验结果为 recall 为 0.72，Hit Ratio@10 为 0.486。实验结论是，结合类别信息与图像数据可以提升推荐系统的性能。

BibTeX

```
@article{2512.15526v1,
  title={BERT and CNN integrated Neural Collaborative Filtering for Recommender Systems},
  author={Abdullah Al Munem and Sumona Yeasmin and Mohammad Rezwanul Huq},
  journal={arXiv preprint arXiv:2512.15526v1},
  year={2025},
  url={http://arxiv.org/abs/2512.15526v1}
}
```



Generated by ArXiv AI Agent • Powered by DeepSeek & Jina AI