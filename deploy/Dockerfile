# LLMServe-Lite 本地推理服务 | deploy/Dockerfile
# 构建能独立启动的 Mock 服务镜像。第一阶段编译程序，后一阶段只带上运行所需文件，
# 因此可以检查 HTTP 与队列行为，无需携带模型。真实 GGUF 的镜像入口见 Dockerfile.cpu。


# 模块：构建镜像：只构建无权重Mock版，不下载模型或隐式获取llama.cpp


FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /source
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY third_party/httplib third_party/httplib
COPY third_party/nlohmann third_party/nlohmann
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && cmake --build build -j2


# 模块：运行镜像：以非root用户运行，延续仅监听回环地址的边界


FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /source/build/llmserve /app/llmserve
COPY configs/mock.json /app/config.json
COPY third_party/httplib/LICENSE /app/licenses/httplib-LICENSE
COPY third_party/nlohmann/LICENSE /app/licenses/nlohmann-LICENSE
COPY LICENSE /app/licenses/llmserve-LICENSE
USER 65532:65532
STOPSIGNAL SIGTERM
ENTRYPOINT ["/app/llmserve"]
CMD ["/app/config.json"]
