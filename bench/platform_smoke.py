#!/usr/bin/env python3
"""Run only owned Linux containers; verify real inference, monitoring and internal alerts."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import socket
import tempfile
import time
import urllib.parse
import urllib.request
import uuid

from run import ROOT, get, request, sha256_file


# 模块：HTTP轮询：只访问本机，返回JSON或明确超时，不修改系统网络设置


def read_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def wait_for(check, seconds=90):
    deadline = time.monotonic() + seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except Exception as error:
            last_error = str(error)
        time.sleep(0.5)
    raise RuntimeError(f'local container check timed out: {last_error}')


# 模块：平台验收：权重只读挂载，Prometheus/Grafana固定版本且仅绑定回环


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='llmserve-cpu:test')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    model = ROOT / 'models' / manifest['filename']
    if sha256_file(model) != manifest['expected_sha256']:
        raise RuntimeError('official model hash mismatch')
    for port in (8080, 9090, 3000):
        with socket.socket() as check:
            check.bind(('127.0.0.1', port))  # 忙碌端口直接报错，不结束其他项目
    containers = []
    report = {
        'engine': 'llama',
        'model_sha256': manifest['expected_sha256'],
        'prometheus_image': 'prom/prometheus:v3.16.0',
        'grafana_image': 'grafana/grafana:13.3.1',
        'scope': 'Linux host-network CPU containers; internal alert state only; no notification delivery',
    }

    def docker(*parts):
        result = subprocess.run(
            ['docker', *map(str, parts)], capture_output=True, text=True, timeout=300
        )
        if result.returncode:
            raise RuntimeError('docker ' + parts[0] + ': ' + result.stderr[-3000:])
        return result.stdout.strip()

    def start(image, *parts, command=()):
        identifier = docker(
            'run',
            '-d',
            '--network',
            'host',
            '--name',
            'llmserve-validation-' + uuid.uuid4().hex[:12],
            *parts,
            image,
            *command,
        )
        containers.append(identifier)
        return identifier

    try:
        with tempfile.TemporaryDirectory(prefix='llmserve-platform-') as temporary:
            directory = Path(temporary)
            directory.chmod(0o755)
            # 临时文件只有公开的本机配置；不会包含访问令牌或用户私人路径配置。
            prometheus = (ROOT / 'deploy/prometheus.yml').read_text()
            prometheus = prometheus.replace(
                'scrape_interval: 5s', 'scrape_interval: 1s\n  evaluation_interval: 1s'
            )
            prometheus = prometheus.replace(
                'scrape_configs:', '  - /etc/prometheus/ci-alerts.yml\nscrape_configs:'
            )
            (directory / 'prometheus.yml').write_text(prometheus)
            (directory / 'ci-alerts.yml').write_text(
                '''groups:
  - name: local-ci-only
    rules:
      - alert: LLMServeCITestRejection
        expr: increase(llmserve_http_status_429_total[10s]) > 0
        for: 1s
'''
            )
            provisioning = directory / 'provisioning'
            (provisioning / 'datasources').mkdir(parents=True)
            (provisioning / 'dashboards').mkdir()
            (provisioning / 'datasources/local.yml').write_text(
                '''apiVersion: 1
datasources:
  - name: LLMServe Prometheus
    uid: llmserve-prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:9090
    isDefault: true
'''
            )
            (provisioning / 'dashboards/local.yml').write_text(
                '''apiVersion: 1
providers:
  - name: llmserve
    type: file
    options:
      path: /var/lib/grafana/dashboards
'''
            )
            dashboard = json.loads((ROOT / 'deploy/dashboard.json').read_text())
            dashboard['templating']['list'][0]['current'] = {
                'text': 'LLMServe Prometheus',
                'value': 'llmserve-prometheus',
            }
            (directory / 'dashboard.json').write_text(json.dumps(dashboard))


            # 模块：容器启动：只读挂载模型与配置，资源上限只作用于本测试的容器


            service = start(
                args.image,
                '--memory',
                '1g',
                '--cpus',
                '2',
                '-v',
                f'{model.parent.resolve()}:/app/models:ro',
            )
            wait_for(lambda: get(8080, '/readyz'))
            prom = start(
                report['prometheus_image'],
                '--memory',
                '256m',
                '--cpus',
                '1',
                '-v',
                f'{directory}/prometheus.yml:/etc/prometheus/prometheus.yml:ro',
                '-v',
                f'{ROOT}/deploy/alerts.yml:/etc/prometheus/alerts.yml:ro',
                '-v',
                f'{directory}/ci-alerts.yml:/etc/prometheus/ci-alerts.yml:ro',
                command=(
                    '--config.file=/etc/prometheus/prometheus.yml',
                    '--web.listen-address=127.0.0.1:9090',
                ),
            )
            wait_for(
                lambda: read_json('http://127.0.0.1:9090/api/v1/query?query=up')['status']
                == 'success'
            )
            grafana = start(
                report['grafana_image'],
                '--memory',
                '512m',
                '--cpus',
                '1',
                '-e',
                'GF_SERVER_HTTP_ADDR=127.0.0.1',
                '-e',
                'GF_AUTH_ANONYMOUS_ENABLED=true',
                '-e',
                'GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer',
                '-e',
                'GF_SECURITY_ADMIN_PASSWORD=' + uuid.uuid4().hex,
                '-e',
                'GF_ANALYTICS_REPORTING_ENABLED=false',
                '-e',
                'GF_ANALYTICS_CHECK_FOR_UPDATES=false',
                '-v',
                f'{provisioning}:/etc/grafana/provisioning:ro',
                '-v',
                f'{directory}/dashboard.json:/var/lib/grafana/dashboards/llmserve.json:ro',
            )
            report['grafana_health'] = wait_for(
                lambda: read_json('http://127.0.0.1:3000/api/health')
            )
            registered = wait_for(
                lambda: read_json('http://127.0.0.1:3000/api/dashboards/uid/llmserve-lite-cpu')
            )
            report['dashboard_panels'] = len(registered['dashboard']['panels'])
            if report['dashboard_panels'] != 4:
                raise RuntimeError('dashboard provisioning did not load expected panels')
            datasource = read_json('http://127.0.0.1:3000/api/datasources/uid/llmserve-prometheus')
            if datasource['url'] != 'http://127.0.0.1:9090':
                raise RuntimeError('Grafana datasource mismatch')

            def query(expression):
                url = 'http://127.0.0.1:9090/api/v1/query?' + urllib.parse.urlencode(
                    {'query': expression}
                )
                return read_json(url)['data']['result']

            wait_for(lambda: query('up{job="llmserve-lite"}')[0]['value'][1] == '1')
            time.sleep(2)  # 先形成零拒绝计数的抓取基线，再制造过载


            # 模块：真实过载：先确认一个任务正在运行，再发并发请求，拒绝不算成功


            with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
                active = pool.submit(request, 8080, '从1开始依次列出三十个数字，使用逗号分隔。', 64)
                wait_for(lambda: get(8080, '/metrics')['active'] == 1, seconds=15)
                pending = [pool.submit(request, 8080, '请简短介绍可靠队列。', 8) for _ in range(8)]

                def rejection_fired():
                    alerts = read_json('http://127.0.0.1:9090/api/v1/alerts')['data']['alerts']
                    return any(
                        a['labels']['alertname'] == 'LLMServeCITestRejection'
                        and a['state'] == 'firing'
                        for a in alerts
                    )

                report['internal_alert_fired'] = bool(wait_for(rejection_fired, seconds=30))
                rows = [active.result()] + [future.result() for future in pending]
            successes = sum(row['success'] for row in rows)
            rejected = sum(row['status'] == 429 for row in rows)
            other_failures = len(rows) - successes - rejected
            if successes < 1 or rejected < 1 or other_failures:
                raise RuntimeError(
                    f'unexpected real workload outcomes: {successes}/{rejected}/{other_failures}'
                )
            wait_for(lambda: not rejection_fired(), seconds=30)
            report['internal_alert_resolved'] = True
            report['requests'] = {
                'total': len(rows),
                'success': successes,
                'rejected_429': rejected,
                'other_failures': other_failures,
            }
            wait_for(lambda: float(query('llmserve_completed_total')[0]['value'][1]) >= successes)
            report['scraped_completed'] = query('llmserve_completed_total')
            report['scraped_histogram_count'] = query('llmserve_success_execution_seconds_count')
            report['server_metrics'] = get(8080, '/metrics')
            report['image_digests'] = {
                image: json.loads(
                    docker('image', 'inspect', image, '--format', '{{json .RepoDigests}}')
                )
                for image in (args.image, report['prometheus_image'], report['grafana_image'])
            }
            docker('stop', '-t', '15', service)
            report['service_exit_code'] = int(
                docker('inspect', service, '--format', '{{.State.ExitCode}}')
            )
            if report['service_exit_code'] != 0:
                raise RuntimeError('real service did not stop cleanly')
    finally:
        for identifier in reversed(containers):
            subprocess.run(['docker', 'logs', '--tail', '40', identifier], check=False)
            subprocess.run(
                ['docker', 'stop', '-t', '15', identifier], check=False, capture_output=True
            )
            subprocess.run(['docker', 'rm', identifier], check=False, capture_output=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
