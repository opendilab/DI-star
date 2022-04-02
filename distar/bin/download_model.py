import sys
import requests
import os
import argparse

# 屏蔽warning信息
requests.packages.urllib3.disable_warnings()


class Downloader:
    def __init__(self, url, file_path, timeout=60):
        self.url = url
        self.file_path = file_path
        self.timeout = timeout
        self.d_times = 0
        self.total_size = 0
        # 第一次请求是为了得到文件总大小
        r1 = requests.get(self.url, stream=True, verify=False)
        if r1.status_code == 200:
            self.total_size = int(r1.headers['Content-Length'])
        else:
            raise ConnectionError('can not connet %s' % self.url)

    def download(self):
        # 本地文件下载了多少
        if os.path.exists(self.file_path):
            temp_size = os.path.getsize(self.file_path)
        else:
            temp_size = 0
        # 从本地文件已经下载过的后面下载
        headers = {'Range': 'bytes=%d-' % temp_size}

        # 重新请求网址，加入新的请求头的
        r = requests.get(url, stream=True, verify=False,
                         headers=headers, timeout=self.timeout)

        with open(self.file_path, "ab") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    temp_size += len(chunk)
                    f.write(chunk)
                    f.flush()

                    # 下载实现进度显示
                    done = int(50 * temp_size / self.total_size)
                    sys.stdout.write("\r[%s%s] %d kb / %d kb " % ('█' * done,
                        ' ' * (50 - done), temp_size/1000, self.total_size/1000))
                    sys.stdout.flush()

        print()  # 刷新控制台，避免上面\r 回车符


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()
    model_name = args.name + '.pth'
    url = 'http://opendilab.org/download/DI-star/' + model_name
    path = os.path.join(os.path.dirname(__file__), model_name)
    print('download model to {}'.format(path))
    d = Downloader(url, path, 5)
    d.download()
