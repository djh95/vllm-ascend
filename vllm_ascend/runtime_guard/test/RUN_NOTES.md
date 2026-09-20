

## 2026-09-20 (II) C5+C6 本机落地执行(方法论重构后首次完整跑通)

上一章(09-20 I)记录了方法论重构资产 + 162 部署;本章为本机(definitive)执行结果。
环境:test-mrv2-cann91 容器,910B4×2(cards 2,3),DSV2-Lite TP2 eager,
product=03a6ad89d,base=50283947d(rg-perf-t0),runner=v1,纯 Python 无重编。
入口 `perf/scripts/run_c56_ab_local.sh`(顺序臂:guard 臂含 C5 dump 计时,再
stress+leakback;裸 base 臂 stress+leakback;差分判决)。

### 结果(results/c56_local_ab_20260920/,definitive run 07:36-08:09)

- **C5 dump 直接计时**(manual_dump 10 组 = 5 档 × 2 rep × 2 rank,54 层/54 文件/rank):
  | tier(prompt tok) | D2H ms/rank | save ms/rank | bytes/rank |
  |---|---|---|---|
  | 1024 (1036) | 150-219 | 101-112 | 34 MiB |
  | 4096 (4108) | 160-413 | 180-233 | 91-125 MiB |
  | 16384 (16396) | 157-716 | 331-769 | 182-490 MiB |
  | 65536 (65548) | 200-881 | 790-2817 | 547-1948 MiB |
  | 131072 (131083) | 639-1140 | 2692-5359 | 2005-3892 MiB |
  rep 间 bytes 有差(dump 按 block 粒度覆盖该请求已分配 block,第二轮分配更多);
  D2H 有效带宽 ~2-4 GB/s;save(torch.save×54)在 128k 档 ~5s 是主要成本。
- **C6 对照差分 leakback**(settle 30s + 300s × 27 点,全检测器开,guard 臂前置 10 次 dump,共 ~19GB dump 文件):
  - guard 臂:RSS +0.2MB,HBM −0.7MB;stress 15 req / 244,169 tok(1k-128k LongBench,序号前缀防 KV 复用)
  - base 臂:RSS +0.0MB,HBM +1.3MB;stress 17 req / 506,591 tok(裸 base 吞吐更高,符合预期)
  - **差分:RSS +0.1MB,HBM −2.0MB → PASS**(门槛 +10MB / +512MB);dump/检测器零残留
  - 第一轮(run1,07:11,留档 verdict_run1_weakstress.txt)stress driver 有 bug 只发了 5 请求,但 C5 数据完整、C6 差分 +0.2MB 同样 PASS

### 过程中修掉的 4 个测试资产 bug(均已 commit 到 analysis)

1. **PYTHONPATH 覆盖**:boot 脚本 `PYTHONPATH=shim:product` 丢掉容器继承的 CANN
   site-packages(/usr/local/Ascend/*/python/site-packages),camem.py 顶层无守卫的
   `from acl.rt import memcpy` 直接 ModuleNotFoundError,报错位置误导为
   multiproc_executor.py:941(实为 worker_main except 日志点)。修:统一
   `:$PRODUCT:${PYTHONPATH:-}` 追加(b51a10add)。
2. **boot 子 shell 组合异步列表**:`( cd tree && ENV setsid python ... & echo $! )`
   把整个 compound list 丢后台,留下 bash wrapper wait4(server) 且持有 `$()` 管道
   → `PID=$(boot)` 在 server 退出前永不返回(实测卡 15min,server 健康)。修:cd
   独立成句,异步任务为简单命令直接 exec(3aebe66cd)。
3. **stress `dict(LB_TIERS)` 三元组崩溃**:两个 worker 线程在首个 LongBench 档
   同时 ValueError 静默死亡,sent=5 提前结束。修:`{n: t for n, t, _ in LB_TIERS}`(3dc82c588)。
4. **npu-smi 25.5 HBM 解析**:NPU id 与芯片名同格、HBM 在下一行 chip 行行尾
   `X / Y`(与 AICore、Mem 挤同格)。修:col-1 首 token 匹配 + 下一行最后一个
   `X / Y` 对(3dc82c588,实测 {2: 29255, 3: 29032})。

### 遗留

- C5 v2 直接计时未跑:本机 product 树钉在 03a6ad89d(无 d6acae3b6 版本门禁),
  v2 boot 需 1ae55b060 环境或 162(任务 #14,用户暂停)。PR/RFC 仍按已披露项处理。
- dump 产物 19GB 在 /data0/test-mrv2-cann91/rg_c56/run/A/dump(重跑会被
  rm -rf;磁盘 11T 充裕,暂留)。
- 162 侧执行恢复配方见上一章(容器三要件:privileged / shm≥1g / driver 整目录)。
