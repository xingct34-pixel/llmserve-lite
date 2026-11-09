# CPU资源与调度实验

## 目的与范围

沿用现有单模型、单context、单worker架构，对比context_tokens为512/1024/2048、queue_capacity为1/4时的资源和接入行为。每例8个客户端、默认8个测量请求、16个输出token上限，先单独预热一次。固定2个模型CPU线程及12个HTTP线程。请求均为SSE。

队列容量改变等待和拒绝行为，不会产生并行模型解码能力。该实验属于请求级资源配置测量，未实现GPU资源调度或continuous batching。

## 运行

Windows进入WSL时明确使用 `wsl.exe -d Ubuntu-24.04`。下列命令在项目根目录的Linux终端执行。

```bash
bash scripts/check.sh mock
bash scripts/check.sh cpu
python3 bench/resource_matrix.py --engine mock --output results/resource-mock
python3 bench/resource_matrix.py --engine llama --output results/resource-real
```

真实实验仅使用已有且SHA-256匹配的授权Qwen Q4_K_M，不下载模型。输出目录应新建。脚本只启动和停止自身子进程，采样可用内存不足700MiB时请求自身服务退出。采样门槛并不等于系统级内存隔离保证。

每例保存所有请求结果、成功数、429数、其他失败、成功请求TTFT p95、成功吞吐、资源采样与服务指标。成功样本分位必须与拒绝率一起阅读。8个请求不足以估计稳健的尾延迟或生产容量。

## 基准工具的内存优化

原cpu_matrix.py使用read_bytes()将整个约469MiB模型读入客户端内存后计算SHA-256。现在使用bench/run.py中的sha256_file()，每次读取1MiB，哈希值保持相同。

这是基准客户端的内存优化，未声称降低llama.cpp推理服务的权重、KV Cache或进程内存。

```bash
python3 bench/hash_memory.py --mode whole-file --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
python3 bench/hash_memory.py --mode stream --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

分别在新Linux进程中运行，比较ru_maxrss峰值，输出带模型哈希及文件大小。CTest的benchmark_helpers测试验证跨块哈希一致性、禁止整文件读取以及缺失文件报错。

实测摘要见 [本轮结果](CPU资源结果-2026-11-09.md)。Mock与真实模型原始结果分开保存。无新增Python第三方依赖，无新增权重。

## 后续学习顺序

先看配置中context_tokens、queue_capacity与http_threads的关系，再跟踪Scheduler和Job的等待、取消与完成。对照bench/resource_matrix.py理解成功延迟与拒绝率为何一起报告，最后用bench/hash_memory.py学习整文件读取和流式读取的内存差别。所有报告中的客户端资源与服务端资源按各自进程解释。
