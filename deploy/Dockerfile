# LLMServe-Lite 本地推理服务 | deploy/Dockerfile
# 功能：构建能独立启动的 Mock 服务镜像。第一阶段编译程序，后一阶段只带上运行所需文件，
# 因此可以检查 HTTP 与队列行为，无需携带模型。真实 GGUF 的镜像入口见 Dockerfile.cpu。
# 目的：得到一份只需模拟引擎即可演示协议和调度行为的运行镜像。


# 模块：构建镜像
# 功能：只构建无权重Mock版，不下载模型或隐式获取llama.cpp
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /source
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY third_party/httplib third_party/httplib
COPY third_party/nlohmann third_party/nlohmann


# 模块：构建阶段
# 编译器和依赖只用于生成可执行文件；后面的 FROM 开始新的运行阶段。
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && cmake --build build -j2


# 模块：运行镜像
# 功能：以非root用户运行，延续仅监听回环地址的边界
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app


# 模块：运行阶段
# 从前一阶段取二进制，再放入 mock 配置。ENTRYPOINT 指定程序，CMD 提供默认配置参数。
# 最后一个 FROM 开始新的文件系统。COPY --from=build 只带入指定产物；容器启动时执行的是 /app/llmserve，用 CMD 里那份配置选择 Mock。
COPY --from=build /source/build/llmserve /app/llmserve
COPY configs/mock.json /app/config.json
COPY third_party/httplib/LICENSE /app/licenses/httplib-LICENSE
COPY third_party/nlohmann/LICENSE /app/licenses/nlohmann-LICENSE
COPY LICENSE /app/licenses/llmserve-LICENSE
USER 65532:65532
STOPSIGNAL SIGTERM
ENTRYPOINT ["/app/llmserve"]
CMD ["/app/config.json"]
