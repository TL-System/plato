import tarfile

path = './data/C'
tar = tarfile.open(path + '.tar.gz')
tar.extractall(path='./data')
tar.close()