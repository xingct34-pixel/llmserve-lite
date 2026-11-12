# 真实CPU容器与监控联调

本机没有Docker/Prometheus/Grafana，本轮不安装系统软件。仓库新增独立GitHub Actions工作流，在Ubuntu隔离runner验证；只有对应提交的工作流成功，才能声称远程路径通过。

## 构建与运行边界

deploy/Dockerfile.cpu从已固定并核验的llama.cpp b6500源码构建，GGML_NATIVE关闭，编译并行最多2。构建上下文用Dockerfile专用ignore文件限制，不包含GGUF、宿主编译产物、Git凭据或私人配置。权重不嵌入镜像，不上传仓库或测试附件。

runner只从已授权的官方固定revision获取同一个Qwen2.5-0.5B-Instruct Q4_K_M，运行前核对完整SHA-256；模型目录只读挂载。镜像使用非root用户，保留回环绑定和SIGTERM协作退出。

在已有合适Linux Docker环境、固定源码及模型已就绪时：

```bash
docker build -f deploy/Dockerfile.cpu -t llmserve-cpu:test .
python3 bench/platform_smoke.py --image llmserve-cpu:test --output results/platform-local.json
```

工具要求8080、9090、3000空闲；忙碌时拒绝执行，不结束其他项目。只清理自己创建并保存ID的容器。该示例依赖Linux host-network，不能推断Windows Docker Desktop默认同样可用。基础镜像及发行版包不是完全字节锁定的供应链。

## 实际检查内容

1. 真实CPU模型就绪、SSE生成成功、容量过载429、无其他失败，服务正常退出码0。
2. Prometheus v3.16.0抓取/metrics/prometheus，验证up、成功计数和直方图样本。
3. Grafana 13.3.1用公开配置自动注册数据源和4个面板，检查健康API、dashboard UID及数据源地址。该检查不等同于对网页截图作视觉验收。
4. 保留生产告警规则，同时额外加载只用于CI的短窗口规则；确认拒绝告警进入firing、停止负载后恢复inactive。没有配置邮件、短信或外部Webhook，不向任何人发送测试通知。
5. 记录官方监控镜像版本/摘要、模型哈希、实际成功/拒绝计数和服务指标；监控不把提示词、路径或请求ID作为标签。

Grafana测试管理密码在进程内临时随机生成，不写入代码、结果或配置。服务、监控均只绑定本机回环；真实模型容器配置默认不启用前缀缓存。

## 结论范围

该路径只验证单机CPU服务、指标采集、看板注册和内部告警状态。没有实现Kubernetes部署、扩缩容、多机调度、GPU、外部告警投递或生产环境安全认证。测试输出不是性能优化对照，不将云端CPU数据与本机i5数据计算加速倍数。
