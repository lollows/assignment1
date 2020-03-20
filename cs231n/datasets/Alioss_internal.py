# -*- coding: utf-8 -*-
import os
import sys
import oss2


class AliOss:
    def __init__(self):
        # 首先初始化AccessKeyId、AccessKeySecret、Endpoint等信息。
        # 首先通过环境变量获取，否则返回default。需将“<你的AccessKeyId>”替换成真实的AccessKeyId等。
        self.access_key_id = os.getenv('OSS_TEST_ACCESS_KEY_ID', 'LTAI4Fipp4y1Br5ok1rk7XQP')
        self.access_key_secret = os.getenv('OSS_TEST_ACCESS_KEY_SECRET', 'hjpRfOVSPIbfevf5Mpi76EBPFziulo')
        self.bucket_name = os.getenv('OSS_TEST_BUCKET', 'deeplearning-dataset')
        self.endpoint = os.getenv('OSS_TEST_ENDPOINT', 'oss-cn-hangzhou-internal.aliyuncs.com')

        # 阿里云主账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM账号进行API访问或日常运维，
        # 请登录 https://ram.console.aliyun.com 创建RAM账号。
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)

        # 创建Bucket对象，所有Object相关的接口都可以通过Bucket对象来进行。
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

    # 当HTTP响应头部没有Content-Length时，total_bytes的值为None。
    def percentage(self, consumed_bytes, total_bytes):
        """进度条回调函数，计算当前完成的百分比
        :param consumed_bytes: 已经上传/下载的数据量
        :param total_bytes: 总数据量
        """
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            print('\r{0}% '.format(rate), end='')

            sys.stdout.flush()

    # 上传文件至oss,如果oss对象文件名已经存在会覆盖，不存在则新建。
    # [Params]
    # from_file: 本地文件名
    # to_oss: oss对象文件名
    def upload(self, from_file, to_oss=None):
        # 删除原文件
        # if self.exists(to_oss):
        #     self.bucket.delete_object(to_oss)

        # 若oss文件名未指定，则默认为当前文件名
        oss_file_name = os.path.basename(from_file) if to_oss is None else to_oss
        self.bucket.put_object_from_file(oss_file_name, from_file, progress_callback=self.percentage)
        print("\rUpload file " + from_file + " to " + oss_file_name)
        return True

    # 下载OSS文件到本地文件。如果指定的本地文件存在会覆盖，不存在则新建。
    # [Params]
    # from_oss: oss对象文件
    # out_path: 本地文件名
    def download(self, from_oss, out_path):
        # oss对象文件不存在返回False
        if not self.exists(from_oss):
            return False

        # 未指定输出路径，默认为当前路径下保存
        out = from_oss if out_path is None else out_path
        self.bucket.get_object_to_file(from_oss, out, progress_callback=self.percentage)
        print("\rDownload file " + from_oss + " to " + out)
        return True

    # 文件是否存在
    # [Params]
    # oss_file_name: oss文件名
    def exists(self, oss_file_name):
        return self.bucket.object_exists(oss_file_name)

    # 删除文件
    # [Params]
    # oss_file_name: oss的文件名
    def delete(self, oss_file_name):
        self.bucket.delete_object(oss_file_name)
        print("Delete file "+oss_file_name)
        return True


if __name__ == "__main__":
    print("===============================================")
    Oss = AliOss()
    # # oss文件名未指定
    Oss.upload('InstallPackage.txt')
    # # 指定oss路径与文件名
    Oss.upload('InstallPackage.txt', 'cifar-10/InstallPackage.txt')
    Oss.upload("/home/lollows/dataset/cifar-10/cifar-10-python.tar.gz", 'cifar-10/cifar-10-python.tar.gz')
    # 删除oss文件
    Oss.delete('cifar-10/InstallPackage.txt')
    # 下载oss文件
    Oss.download('InstallPackage.txt', "/home/lollows/dataset/InstallPackage.txt")
    Oss.download('cifar-10/cifar-10-python.tar.gz',"/home/lollows/dataset/cifar-10/cifar-10-python.tar.gz")
