# DeepSeek-V4-Flash SFT 后 SWE-bench-verified 回退分析报告

> 对象:`kaynzhang_128k_iter1164-fp8fp4`(在 DeepSeek-V4-Flash 上用 albaliang_057 ≤128k 数据 SFT 3 epoch)
> 现象:SWE-bench-verified 解题率从 ~69% 掉到 ~53%
> 评测:Harbor `cbc-agent` + `swebench-verified` + `ags` 沙盒,temp=1,单次采样(n_attempts=1)
> 结论日期:2026-06-10

---

## 0. TL;DR

- **掉点幅度**:69.2%(346/500)→ **53.4%(267/500)**,净 **-15.8 个点 / -79 题**(损失 100、新得 21)。
- **根因**:**SFT 训练数据的分布错配 + "收尾即停"行为先验**。
  - 训练数据 98% 是**中文 CodeBuddy IDE 交互式会话**(Claude 蒸馏),SWE 风格 ≈0.2%;评测是**英文、全自主的 Python 仓库修 bug**。
  - 模型被正常训练出"做完→发总结→停"的收尾先验,在分布外的自主 SWE 场景里**过早触发** → 空补丁。
- **已逐一排除**(均有实测):
  - ❌ fp8fp4 量化(仅 -1.8 点;且 baseline 本身已是 fp8/fp4)
  - ❌ 部署模板 / stop token(tokenizer/generation config 与 base 逐字节相同)
  - ❌ 上下文溢出、输出截断、死循环、proxy 错误
  - ❌ **loss mask / 工具调用被 mask**(数据层 + token 级 `.pt` + miles 源码三重证据,mask 健康)
- **两套失败机制**:① 过早终止/空补丁(行为/校准问题,**可被降温缓解**);② 改错(真实的分布外能力退化,降温救不了)。
- **修复杠杆在数据配比**,不在 mask/encoding。

---

## 1. 背景与对比对象

| | base | **"original"(对照)** | **SFT(本次)** |
|---|---|---|---|
| 模型 | DeepSeek-V4-Flash | DeepSeek-V4-Flash-roundtrip-v2-full | kaynzhang_128k_iter1164-fp8fp4 |
| 精度 | 原生 fp8/fp4 | fp8/fp4(量化往返,无 SFT) | fp8/fp4(同) |
| 是否 SFT | 否 | 否 | **是** |
| 解题率 | 71.0% | **69.2%** | **53.4%** |

两个评测 run 的配置**除模型端点外完全一致**(cbc-agent / CBC_MAX_TURNS=100 / CBC_MAX_OUTPUT_TOKENS=8192 / swebench-verified / ags / temp=1)。

> **关键澄清**:用户拿来对比的 "original"(roundtrip-v2-full)**本身就是 fp8/fp4 量化版**(其 `config.json` 与 SFT 模型逐字节相同,md5 `7f34d48…`)。因此 69.2%→53.4% 这个对比**天然已控制住量化变量**,差值是**纯 SFT 效应**。

---

## 2. 量化结论:分数与失败分解

逐题 join(归一化任务名后):

| 量 | 值 |
|---|---|
| both resolved | 246 |
| **newly failed**(orig 过、sft 挂) | **100** |
| newly passed(sft 过、orig 挂) | 21 |
| 净 | **-79 题 = -15.8pt** |

**100 个新失败的主因拆解**:

| 主因 | 数量 | 性质 |
|---|---|---|
| **空补丁**(完全没产出 patch) | **43** | 行为退化:过早终止 |
| **改对流程但补丁错**(CLEAN_BUT_WRONG) | **54** | 能力退化:浅探索/改错 |
| length 截断 | 2 | 可忽略 |
| 极少轮数(带 patch) | 1 | 可忽略 |

**轨迹形态对比**(全量 500 题):

| 指标 | original | SFT | 解读 |
|---|---|---|---|
| 中位轮数 | 32 | **20** | 轨迹变短 |
| 平均轮数 | 40.9 | **23.5** | |
| ≤5 轮收场 | 2 | **51** | 大量草草收场 |
| 撞满 100 轮 | 26 | **2** | **不是死循环** |
| 每任务 completion tokens | 12509 | **7553** | 输出腰斩 |
| 空补丁数 | 4 | **76** | 19× |
| max_prompt_tokens 均值 | 48k | **42k**(更低) | **不是 context 爆炸** |
| length 截断 response 数 | 3 | 9 | 可忽略 |

---

## 3. 失败机制一:过早终止 / 空补丁(≈43 题 + 全局 76 空补丁)

**机制**:SFT 模型经常在一句"前言"后直接吐 `finish_reason=stop`、**不发工具调用**。cbc-agent 把"无 tool call 的回合"判为"任务完成"→ 立即终止 → 空补丁 → 必败。

**实测**(每条轨迹最后一个 response):

| 指标 | original | SFT |
|---|---|---|
| 近空过早终止(finish=stop, 无tool_call, <40 token) | **2** | **72**(36×) |
| 第 1 轮就 stop | 0 | 3 |

76 个空补丁的结尾分类:**57 个是"开场白后过早 stop"**,18 个是"假装完成的总结",1 个 other。

**典型样例**(铁证):
- `django__django-13410`:**第 1 轮**输出 `"Let me read the current file to understand the bug."` 然后 stop。说要读文件却没读,空补丁。
- `scikit-learn-10908` / `django-14725`:**第 1 轮**只吐 `'\n\n'`(8 token)就 stop。
- `django-13807`:`"Now I understand the issue. The PRAGMA… don't properly quote table names…"` 诊断对了,说到一半 stop。

> 模型常常**诊断对了 bug 才停**,说明这是**行为/校准**退化,不是推理能力丧失。

---

## 4. 失败机制二:改错(≈54 题)

跑完完整轨迹、产出合法 patch、但测试不过。抽样 8 个任务(django/astropy/sympy/sklearn/pytest)逐轨迹对比 original(全对)vs SFT(全错):

**最强信号:轨迹长度坍缩**(SFT 探索更少、过早下结论):
django-11400 50→8、astropy-14096 52→16、sympy-14976 55→16、sympy-22456 136→39、pytest-7205 26→6。

**失败模式**:
- **no-op/占位补丁**:`sympy-22456` 只加一行注释 return 没改;`django-13297` 写出 `value=value` 自赋值。
- **半成品/内部不一致**:`sklearn-25931` 加了 `check_input` 参数却没接 `if` 守卫,校验照跑;`django-11400` 三个该改的文件只改一个,新路径运行时 `TypeError`,却只跑不命中的旧测试就宣布通过。
- **铁证(知道对的、选了错的)**:`pytest-7205` 最终输出**明确写出正确修复**("应该用 saferepr/repr()",正是金标准),然后用一段**编造的错误理由**把自己劝退,改成无效等价改动。

**占比估计**(外推到 54 题):真实能力退化 **~75-85%**;temp=1 纯采样噪声 ~10-20%;灰色 ~5-10%。无任何指向量化损坏的证据(补丁语法合法、改对文件区域,失败全在语义决策层)。

---

## 5. 已排除的备择假设(均有实测)

| 假设 | 判据 | 结论 |
|---|---|---|
| fp8fp4 量化掉点 | `roundtrip-v2-full` 是"只量化不 SFT"对照组:71.0→69.2 = **-1.8**;且 baseline 本就 fp8/fp4 | ❌ 否 |
| 部署模板 / stop token 配错 | base 与 SFT 的 `tokenizer_config.json` + `generation_config.json` **逐字节相同**(md5 一致) | ❌ 否,是权重 |
| 上下文溢出 | max_prompt_tok 均值 48k→**42k**(更低) | ❌ 否 |
| 输出被 8192 截断 | length 截断 response 3→9,可忽略 | ❌ 否 |
| 死循环 / 撞满轮数 | 撞满 100 轮 26→**2**,SFT 反而更短 | ❌ 否 |
| proxy / API 错误 | http 错误 0,success 全绿 | ❌ 否 |
| **loss mask / 工具调用被 mask** | 见 §6,三重证据 | ❌ **否,mask 健康** |

---

## 6. 训练侧深查:真实数据与 loss mask

### 6.1 真实训练管线(来自 wandb)

`outputs/stageAlb128K-3ep-H20-20260602-010411/wandb` 的 `wandb-metadata.json`:

- **框架 = `miles`**(github.com/radixark/miles,Megatron 系),入口 `miles/train.py` —— **不是 ms-swift**(`agent-sft/convert.py` 那套是废弃的独立管线,勿混淆)。
- **数据 = `data/albaliang_057_le128k.jsonl`**,`--input-key messages --tool-key tools`。
- **mask = `--loss-mask-type deepseek_v4`** + `--calculate-per-token-loss` + `--loss-type sft_loss`。
- base = `DeepSeek-V4-Flash-bf16-unpacked`,**3 epochs**,lr 5e-6 cosine,128k ctx,16×H20,TP8/PP4/CP4/EP8。
- `MILES_DSV4_THINKING_MODE=chat`、`DROP_THINKING=0`(thinking 段为空)。

### 6.2 数据画像(albaliang_057_le128k,1500 条样本统计)

| 维度 | 实况 | 与评测的错配 |
|---|---|---|
| 来源 | **98% CodeBuddy IDE 会话**(Claude-Opus-4.5/4.6 蒸馏),OpenHands/SWE 风格 ≈0.2% | 🔴 评测是 SWE Python 修 bug |
| 语言 | **99% 中文** | 🔴 评测全英文 |
| system prompt | 91% IDE 版"powered by Claude-Opus-4.x";评测用的 "CodeBuddy Code" CLI 版仅 **~5%** | 🔴 训练/推理提示词分布错位 |
| step mask | **49.7% 的 assistant 步整步 mask** | 高,但前言/调用同步,健康 |
| 收尾形态 | **41% 的对话以"被训练的、纯文本无工具调用的总结"收尾**("修复完成!✅/已完成…") | 🟡 教模型"总结后停"的强先验 |
| 工具分布 | read_file 32% / replace_in_file 20% / search_content 12% / execute_command 12% | 工具调用训练充分 |
| `ask_followup_question` | 仅占被训练工具调用的 **0.01%** | "停下问用户"假说被排除 |

### 6.3 loss mask 核查:三重证据,mask 健康

**问题**:SFT 是否把工具调用 token mask 掉、或在前言后训练了 EOS,导致模型学会"前言后过早停"?

**(a) 数据层**:assistant message 自带 `_loss_content` / `_loss_tool_calls`(逐调用) / `step_loss_mask`。全量统计联合分布只有两种——`(content=1, calls 全1, step=1)` 与 `(content=0, calls 全0, step=0)`。**"训前言/mask 调用"= 0 条,反向 = 0 条。** 工具调用与其前言永远同 loss。

**(b) token 级 ground truth**:直接读训练产物 `dump_details/train_data/*.pt`(模型真实训练过的 tokenized+mask 样本;用 stdlib 无 torch 反序列化器读出)。1024 样本 / 9992 个被训练回合:

| 检查 | 结果 |
|---|---|
| [B] 工具调用标记 token(`｜DSML｜` 等)训练率 | **93424/93424 = 100%** |
| [C] EOS 训练率(工具调用回合) | 8176/8176 = 100% |
| [C] EOS 训练率(纯文本回合) | 1816/1816 = 100% |
| **[A] 工具调用前出现【被训练的 EOS】的回合** | **0**(结构上不可能) |

实证片段:`…</｜DSML｜tool_calls> <｜end▁of▁sentence｜>`,每个 token(含工具标记、含 EOS)都是 `mask=1`;EOS 永远在**完整工具调用之后**。

**(c) miles 源码**(subagent 读 62 文件):
- mask 实现 `miles/utils/mask_utils.py:135 gen_multi_turn_loss_mask_deepseek_v4`:按 char span,整个 assistant 段 mask=1,`step_loss_mask=0` 时整段清零。
- chat 模板 `encoding_dsv4.py:45` 写死 `assistant = "{reasoning}{content}{tool_calls}" + eos` → **EOS 永远在 tool_calls 之后**,且整条 message 同 mask → "前言/调用间夹被训练的 EOS" **结构上不可能**。
- 工具调用 arguments(渲染成 `<｜DSML｜parameter…>`)整体计入 loss;user/system/tool 结果 mask=0,正确。

> **三条独立证据一致:mask/EOS 渲染健康,掉点不在 token mask。** 这也解释了为什么"降温会让分数回升"——病在分布的熵/尾部,不在确定性管线。

### 6.4 两个值得跟进的训练侧线索(非主因)

1. **`thinking_mode=chat` → 整个 SFT 没有 reasoning 监督**(thinking 段为空)。可能弱化"先想再动",助长浅探索/改错。评测时 base/SFT 都是非 thinking,故非 train/eval 失配,但值得做消融。
2. **死字段隐患**:miles **只读 `step_loss_mask`**,`_loss_content`/`_loss_tool_calls` 在全仓零引用。当前数据三字段同步所以无害;**将来若用"content 训、部分 tool_calls 想 mask"的数据,miles 会把整段都按 content 待遇训练,产生 mask 偏差**——需在数据准备侧保证三字段一致,或给 miles 加读取细粒度字段的能力。

---

## 7. 根因总结

```
SWE-bench 回退 -15.8pt
├─ 量化           -1.8pt   (已隔离,可忽略)
└─ SFT            -14~16pt (主因)
   ├─ 过早终止/空补丁  ≈43题  ← 数据分布(中文IDE"收尾即停"先验)+ temp=1 → 自主场景过早 stop
   │                          【可被降温缓解;属校准/熵问题】
   └─ 改错            ≈54题  ← 分布外能力退化(中文IDE→英文自主SWE;可能叠加无reasoning监督)
                              【降温救不了;属能力问题】
```

两半共享同一个根因:**SFT 把模型推向了一个"更短、更早收尾、面向中文 IDE 交互"的分布**,而评测要求"英文、长程、全自主"。

排除项:量化(-1.8)、模板、上下文、截断、循环、proxy、**loss mask** 全部不成立。

---

## 8. 修复建议(分层)

**A. 止血 / 验证(廉价,先做)**
1. **降温重测**:temp 1.0→0.6/0.7,只重跑 100 个新失败。若回升集中在 43 个空补丁 → 坐实"过早终止=采样校准问题"。同时给 base 也降温,看**敏感度差**。
2. **harness 兜底**:对"非最终轮、无 tool call、无补丁、像未完成开场白"的回合,改为重新 prompt"continue"而非判完成,可捞回 ~57 个开场白早停。
3. **pass@4**:对 100 题做多 seed 重采样,把系统性退化与采样噪声分开。

**B. 根治(数据/训练配方,主要杠杆)**
4. **掺入英文自主 SWE-agent 轨迹**:Python 仓库修 bug、cbc-agent/CodeBuddy-Code 格式、长自主工具循环、以**验证过的 patch 收尾**、不要中文总结式早停。提高配比直至模型重获自主持续性。
5. **对齐训练 system prompt 与评测**:评测的 CLI 版当前仅 ~5%,应大幅提高。
6. **控过训**:98% 窄分布上 3 epoch 会把"收尾即停 + 领域"先验刻深,考虑减 epoch 或混通用 agentic 数据。
7. **消融 `thinking_mode`**:试保留 reasoning 监督,看是否改善"改错"半。

**C. 工程隐患**
8. 数据准备侧保证 `_loss_content`/`_loss_tool_calls`/`step_loss_mask` 三字段一致(或给 miles 加细粒度 mask 读取),避免将来数据触发 mask 偏差。

---

## 9. 附录:方法与可复现脚本

所有脚本与中间数据归档在 `/data_train/kaynzhang/v4-sft/swebench_analysis/`。

**评测产物**(CFS):
- original:`/data_fast_v3/eremite/cache/harbor_eval/output/DeepSeek-V4-Flash-roundtrip-v2-full/2026-06-09__11-33-02__…__swebench-verified__all/`
- SFT:`…/kaynzhang_128k_iter1164-fp8fp4/2026-06-10__00-49-32__…__swebench-verified__all/`
- base:`…/DeepSeek-V4-Flash/2026-06-09__09-13-01__…/`
- 每任务:`verifier/reward.txt`(0/1)、`result.json`、`code_diff/agent.patch`、`proxy/*.{input,output}.json`(SSE 流)

**脚本**:
| 脚本 | 作用 |
|---|---|
| `swe_analyze.py` | 全量抽取两个 run 的 reward/轮数/patch/finish_reason/token,存 `orig.json`/`sft.json` |
| `dump_traj.py` | 把单个任务的完整对话(含工具调用、最终输出、patch、reward)dump 成可读文本 |
| `scan_premature.py` | 统计两个 run 的"过早终止"率 |
| `empty_patch_forensics.py` | 76 个空补丁的结尾形态分类 |
| `pt_notorch.py` | **无 torch** 读取 `dump_details/*.pt`(zip+pickle+array)的反序列化器 |
| `inspect_pt_mask.py` | token 级 loss_mask 核查(训练机有 torch 时用;[A]/[B]/[C] 判定) |

**关键命令**(在本 CFS 上,默认 python3 即可,无需 torch):
```bash
cd /data_train/kaynzhang/v4-sft/swebench_analysis
python3 swe_analyze.py                 # 重建 orig.json / sft.json(~10min)
python3 -c "import sys;sys.path.insert(0,'.');from pt_notorch import load_pt; \
  obj,mat=load_pt('/data_train/kaynzhang/v4-sft/outputs/stageAlb128K-3ep-H20-20260602-010411/dump_details/train_data/0_0.pt'); \
  print(list(obj['rollout_data'].keys()))"   # 验证 .pt 可读
```

**真实训练数据 / 配置**:
- 数据:`/data_train/kaynzhang/v4-sft/data/albaliang_057_le128k.jsonl`
- wandb:`/data_train/kaynzhang/v4-sft/outputs/stageAlb128K-3ep-H20-20260602-010411/wandb/`
- mask 代码:`/data_train/kaynzhang/v4-sft/miles/miles/utils/mask_utils.py:135`
- chat 模板:`/data_train/kaynzhang/v4-sft/models/DeepSeek-V4-Flash/encoding/encoding_dsv4.py:45`
- 训练 tokenized 产物:`…/stageAlb128K-3ep-H20-20260602-010411/dump_details/train_data/*.pt`(21888 个)
