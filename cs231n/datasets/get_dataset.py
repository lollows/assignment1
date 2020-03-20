import Alioss_internal


# download deeplearning dataset from Aliyun Oss
from_oss = "cifar-10/cifar-10-python.tar.gz"
outpath = "/home/lollows/dataset/cifar-10/cifar-10-python.tar.gz"
Oss = Alioss_internal.AliOss()
Oss.download(from_oss, outpath)
