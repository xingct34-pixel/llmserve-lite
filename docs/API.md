# HTTP API

默认地址 `http://127.0.0.1:8080`，仅本机回环。JSON 编码为 UTF-8，响应含 `Cache-Control: no-store`。keep-alive 最多处理一次请求，简化低并发资源回收。

| 接口 | 返回 |
|---|---|
| GET /healthz | 200，`{"status":"ok"}` |
| GET /readyz | 200，`{"ready":true}`；关闭中为 503 |
| GET /metrics | JSON 累计计数、队列深度与峰值、active、engine |
| POST /v1/completions | 非流式 JSON 或 SSE |

请求示例：

```json
{"prompt":"你好","max_tokens":16,"stream":true}
```

只接受 prompt、max_tokens、stream 三个字段。prompt 必须为非空有效 UTF-8 字符串，不含 NUL，最多 12000 字节；整个请求体最多 16384 字节。max_tokens 为 int32 且在 1 到配置上限之间，默认 128；stream 为布尔值，默认 false。输入长度最终还需通过引擎上下文校验。

非流式成功结果：

```json
{"type":"done","request_id":"1","text":"你好","finish_reason":"length","usage":{"prompt_tokens":10,"completion_tokens":2},"timings":{"queue_ms":0.1,"execution_ms":40.2,"prefill_ms":0,"decode_ms":40.1,"first_token_ms":20.1}}
```

上例数字仅用于说明字段，**不是实测**。mock 的 token 口径是模拟片段与字符估算；真实 llama 引擎才使用实际 tokenizer。计时均为 steady_clock 的毫秒差，不受系统时间校准影响。

SSE 成功示例：

```text
data: {"type":"delta","request_id":"1","text":"你"}

data: {"type":"delta","request_id":"1","text":"好"}

data: {"type":"done","request_id":"1","finish_reason":"length","usage":{"prompt_tokens":10,"completion_tokens":2},"timings":{}}

data: [DONE]

```

一次成功流恰好有一个 `[DONE]`。不能假设一个 delta 对应一个 token，因为一个 UTF-8 字符可能跨多个模型 token。真实输出 token 数以末尾 usage 为准。

SSE 中途失败：

```text
event: error
data: {"type":"error","request_id":"1","code":"generation_timeout","message":"generation_timeout"}

```

错误流不会再发送 `[DONE]`。客户端应检查终态与成功结束标记，不能只看 HTTP 200；缺失正常终态的断流也不能算成功。

| 情况 | HTTP 状态 / 行为 |
|---|---|
| JSON 错误、未知字段、参数越界、上下文超限 | 400 |
| 请求体过大 | 413 |
| 接入在途上限或业务队列满 | 429，`Retry-After: 1` |
| 未知路由 | 404 |
| 关闭中 | 503 |
| 响应头前排队或执行超时 | 504 |
| 响应头前引擎失败 | 500 |
| 已开始 SSE 后失败 | 200 已发出；error 事件终止 |
| 客户端断连 | socket 写失败触发取消，不生成成功结束标记 |

`X-Request-ID` 在任务被接收后提供；请求 ID 进程内递增，不跨重启持久化。应用日志每条一行 JSON，含 timestamp_unix_ms、事件、请求 ID、终态与阶段耗时；llama.cpp 启动/加载诊断可能附带原生文本。`/metrics` 是 JSON 快照，不是 Prometheus exposition 格式；故障、取消、超时分开计数。

默认 MockEngine 支持 `__error__` 和 `__error_after__` 两个特殊 prompt，用于注入开始前错误与第三个 delta 后的错误；真实 llama 引擎不会解释这些标记。


## HTTP 状态计数

`http_requests_started` 在 completions handler 开始时增加；`http_responses_logged` 与 `http_status_429/4xx/5xx` 来自 HTTP 库响应日志。它们与推理调度计数分开，覆盖已记录的接入层拒绝，但不保证覆盖所有底层连接失败。SSE 中途 error 不改变已发送的 HTTP 200。`completed` 是生成成功，不证明客户端收到全部内容；压测以客户端结果为准。
