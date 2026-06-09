# 训练 ckpt → 原始 FP8+FP4 转换

把训练产出的 Megatron `torch_dist` checkpoint(BF16)转换回 DeepSeek 官方发布的
**dpsk MegaBlocks FP8+FP4** 布局(46 shards,与 `models/DeepSeek-V4-Flash` 逐字段一致),
用于部署 / 对外发布 / 节省存储。

## 流水线

```
outputs/<stage>/checkpoints/iter_N      训练产物 (torch_dist, 权重存 BF16)
        │  ① tools/convert_torch_dist_to_hf.py   (miles, --model-name deepseekv4)
        ▼
        <name>-hf-bf16                  HF 命名的 BF16 safetensors (~540 GB)
        │  ② tools/hf_bf16_to_megablocks.py       (本仓,纯 PyTorch 再量化)
        ▼
        <name>-fp8fp4                   dpsk FP8+FP4,46 shards (~149 GB)
```

- **FP8/FP4 量化只发生在第②步。** torch_dist 与 HF 这两种中间态都是 BF16。
- 第②步是 `tools/megablocks_to_hf_bf16.py`(官方 dpsk→bf16 反量化工具)的**精确逆**:
  - dense / attn / shared_experts / e_proj / indexer.wq_b → FP8 `e4m3`,block 128×128,`ue8m0` scale;
  - 路由专家 `ffn.experts.*` → FP4 `e2m1` 打包 int8,block-32,`ue8m0` scale;
  - norm / embed / gate / hc_* / compressor 等 → 原样保留 BF16/F32。

## 一键转换

```bash
# 0. 构建精简转换镜像(只需一次)。若 docker build 内访问 PyPI 需走代理:
#    docker build --network host \
#      --build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$https_proxy \
#      -f docker/Dockerfile.convert -t v4-convert:latest docker/
docker build -f docker/Dockerfile.convert -t v4-convert:latest docker/

# 1. 跑转换(换成你的 stage / iter 和一个输出名)
tools/ckpt_to_fp8fp4.sh \
  --iter outputs/<stage>/checkpoints/iter_NNNNNNN \
  --name my-sft-v1
# 产出:  $V4_MODELS/my-sft-v1-fp8fp4   (中间的 -hf-bf16 默认转换后删除)
```

脚本会依次执行 ①→ keymap 抽检 →②(默认 `--fill-missing-from-template` 补 MTP)→③ 结构校验,
全程在 `v4-convert` 镜像里、挂载 `$V4_WORK` 到 `/work`。

常用参数:

| 参数 | 含义 |
|---|---|
| `--iter <dir>` | 训练 ckpt 的 `iter_*` 目录(相对 `$V4_WORK` 或绝对路径) |
| `--name <prefix>` | 输出目录前缀,生成 `<prefix>-hf-bf16` / `<prefix>-fp8fp4` |
| `--keep-bf16` | 保留 ~540 GB 的中间 HF BF16(默认转换后删除) |
| `--no-mtp` | 不从原始模板补 MTP 层(见下方「MTP」) |
| `--cpu` | 不用 GPU(更慢但可在无卡机器跑) |
| `--force` | 输出目录已存在时先删后写 |

环境变量覆盖:`V4_WORK`、`V4_CONVERT_IMAGE`、`V4_MILES_REPO`、`V4_TRAINING_REPO`、
`V4_TEMPLATE`、`V4_BF16_DIR`。

## 需要哪些组件(两个镜像分工)

两步对环境的要求不同,因此用**两个镜像**:

| 步骤 | 镜像 | 为什么 |
|---|---|---|
| ① torch_dist → HF bf16 | **训练镜像** `radixark/miles:dev-fht-v4deps-20260529` | 走 miles 的 `megatron_to_hf`,其 `processors/quantizer_fp8` **硬 import sglang**(`MultiprocessingSerializer` 等),离不开完整训练栈 |
| ② HF bf16 → FP8/FP4 + 全部校验 | **精简镜像** `v4-convert`(`docker/Dockerfile.convert`) | 纯 PyTorch 再量化,**不需要** megatron / TE / tilelang / sglang / deep_ep / FHT / TileKernels |

精简镜像(`v4-convert`)只装运行时,≈ 7 GB(对比训练镜像 ~50 GB):

| 组件 | 用途 |
|---|---|
| **torch 2.9.x** | `float4_e2m1fn_x2` / `float8_e8m0fnu` / `float8_e4m3fn` dtype(FP8/FP4 量化) |
| safetensors / transformers / tqdm / numpy | 读写分片 / 基础依赖 |

代码(本仓 `tools/*` + step① 用的 miles `megatron_to_hf`)**运行时挂载**,不打进镜像,避免漂移。
一键脚本 `tools/ckpt_to_fp8fp4.sh` 已自动用对镜像(① 训练镜像,②/③ 精简镜像)。

> 第①步还需 miles 里 **v4 版** `megatron_to_hf`(`miles/backends/megatron_utils/megatron_to_hf/deepseekv4.py`);
> 训练镜像 baked 的旧 miles 不支持 v4,所以脚本用 `PYTHONPATH=$V4_MILES_REPO` 指向 CFS 上的 miles checkout。
> 因为传了 `--model-name deepseekv4`,第①步不会真正调用到需要 sglang 的量化分支(只是 import 时需要它存在)。

## 手动分步(调试 / 断点续跑用)

```bash
ROOT=$V4_WORK
TRAIN_IMG=radixark/miles:dev-fht-v4deps-20260529   # 训练镜像(step①)
CONV_IMG=v4-convert:latest                          # 精简镜像(step②/③)

# ① torch_dist -> HF bf16  (训练镜像 + PYTHONPATH 指向 v4-aware miles)
docker run --rm --gpus all -e PYTHONPATH=/work/miles \
  -v $ROOT:/work -w /work/miles $TRAIN_IMG \
  python3 tools/convert_torch_dist_to_hf.py \
    --input-dir /work/outputs/<stage>/checkpoints/iter_N \
    --output-dir /work/models/<name>-hf-bf16 \
    --model-name deepseekv4 \
    --origin-hf-dir /work/models/DeepSeek-V4-Flash-bf16-unpacked --vocab-size 129280

# ② HF bf16 -> dpsk FP8/FP4  (精简镜像)
docker run --rm --gpus all -v $ROOT:/work -w /work/deepseek-v4-flash-training/tools $CONV_IMG \
  python3 hf_bf16_to_megablocks.py \
    --hf-src /work/models/<name>-hf-bf16 --template /work/models/DeepSeek-V4-Flash \
    --dst /work/models/<name>-fp8fp4 --fill-missing-from-template

# ③ 校验(应输出 "OK — 69187 tensors ... identical names/dtype/shape")
docker run --rm -v $ROOT:/work -w /work/deepseek-v4-flash-training/tools $CONV_IMG \
  python3 verify_roundtrip.py struct --a /work/models/<name>-fp8fp4 --b /work/models/DeepSeek-V4-Flash
```

`verify_roundtrip.py` 其它子命令:`keymap`(键集覆盖)、`micro`(量化原语逐位闭环)、
`cmp-dequant`(把新 ckpt 反量化逐位对比 bf16-unpacked)、`cmp-bf16`(两个 HF bf16 目录逐位对比)。

## 注意事项

- **MTP 层**:SFT 通常**不保存** MTP/nextn 层,第①步产物会缺 797 个 `mtp.0.*`。
  默认 `--fill-missing-from-template` 从原始 `DeepSeek-V4-Flash` 拷贝(未训练的)MTP 补全,
  得到完整 46-shard。若不需要 MTP,用 `--no-mtp`,并改用一份 `num_nextn_predict_layers: 0` 的 config。
- **有损**:对微调后的 BF16 再量化是第二次舍入,**无法还原原始字节**,但能得到一个有效、
  数值等价的 FP8/FP4 ckpt(结构与原始逐字段一致)。
- **资源**:第①步把整模型(~531 GB)load 进 CPU 内存;磁盘需 ~700 GB 空闲(540 GB 中间 + 149 GB 产物)。
  第①步 CPU 密集(可省 GPU),第②步用 GPU 更快(纯 CPU 也能跑,见 `--cpu`)。
- **清理**:中间 HF BF16 默认转换后删除;手动跑时记得自行清理(root 属主,用容器 `rm -rf`)。

## 校验含义

- `struct` 通过 = 新 ckpt 的张量名 / dtype / shape / 分片与官方 `DeepSeek-V4-Flash` **逐字段相同**,可直接当原始格式使用。
- 该转换链路已端到端验证:bf16↔FP8/FP4 round-trip 在全部 69187 个张量上**逐位一致**;
  真实训练 ckpt 走完 ①→② 后 `struct` 全过(详见提交历史与 `tools/verify_roundtrip.py`)。

## 相关文件

- `tools/ckpt_to_fp8fp4.sh` — 一键编排脚本
- `tools/hf_bf16_to_megablocks.py` — 第②步,BF16→dpsk FP8/FP4(`megablocks_to_hf_bf16.py` 的逆)
- `tools/megablocks_to_hf_bf16.py` — 反向参考工具,dpsk FP8/FP4→BF16
- `tools/verify_roundtrip.py` — 校验工具集
- `docker/Dockerfile.convert` — 精简转换镜像
