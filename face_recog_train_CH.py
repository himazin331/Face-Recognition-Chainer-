# -*- coding: utf-8 -*-

# 顔画像学習プログラム(Chainer)(開発終了)

import argparse as arg
import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L 
from chainer import training
from chainer.training import extensions

# CNNの定義
class CNN(chainer.Chain):
    
    # 各層定義
    def __init__(self, n_out):
        super(CNN, self).__init__(
            # 畳み込み層の定義
            conv1 = L.Convolution2D(1, 16, 5, 1, 0),  # 1st 畳み込み層
            conv2 = L.Convolution2D(16, 32, 5, 1, 0), # 2nd 畳み込み層
            conv3 = L.Convolution2D(32, 64, 5, 1, 0), # 3rd 畳み込み層

            #全ニューロンの線形結合
            link = L.Linear(None, 1024), # 全結合層
            link_class = L.Linear(None, n_out), # クラス分類用全結合層(n_out:クラス数)
        )
        
    # フォワード処理
    def __call__(self, x):
        
        # 畳み込み層->ReLU関数->最大プーリング層
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2)   # 1st
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2)  # 2nd
        h3 = F.relu(self.conv3(h2))  # 3rd
        
        # 全結合層->ReLU関数
        h4 = F.relu(self.link(h3))
        
        # 予測値返却
        return self.link_class(h4) # クラス分類用全結合層
 
# Trainer
class trainer(object):
    
    # モデル構築,最適化手法セットアップ
    def __init__(self):
        
        # モデル構築
        self.model = L.Classifier(CNN(2))
        
        # 最適化手法のセットアップ
        self.optimizer = chainer.optimizers.Adam() # Adamアルゴリズム
        self.optimizer.setup(self.model) # optimizerにモデルをセット
        
    # 学習
    def train(self, train_set, batch_size, epoch, gpu, out_path):

        # GPU処理に対応付け
        if gpu >= 0:
            chainer.cuda.get_device(gpu).use() # デバイスオブジェクト取得
            self.model.to_gpu()  # 入力データを指定のデバイスにコピー
        
        # データセットイテレータの作成(学習データの繰り返し処理の定義,ループ毎でシャッフル)
        train_iter = chainer.iterators.SerialIterator(train_set, batch_size)

        # updater作成
        updater = training.StandardUpdater(train_iter, self.optimizer, device=gpu)
        # trainer作成
        trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_path)

        # extensionの設定
        # 処理の流れを図式化
        trainer.extend(extensions.dump_graph('main/loss'))
        # 学習毎snapshot(JSON形式)書込み
        trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
        # log(JSON形式)書込み
        trainer.extend(extensions.LogReport())
        # 損失値をグラフにプロット
        trainer.extend(
                extensions.PlotReport('main/loss', 'epoch', file_name='loss.png'))
        # 実値をグラフにプロット
        trainer.extend(
                extensions.PlotReport('main/accuracy', 'epoch', file_name='accuracy.png'))
        # 学習毎「学習回数, 損失値, 実値, 経過時間」を出力
        trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'main/accuracy', 'elapsed_time']))
        # プログレスバー表示
        trainer.extend(extensions.ProgressBar())

        # 学習開始
        trainer.run()

        print("___Training finished\n\n")
        
        # モデルをCPU対応へ
        self.model.to_cpu()
    
        # パラメータ保存
        print("___Saving parameter...")
        param_name = os.path.join(out_path, "face_recog.model") # 学習済みパラメータ保存先
        chainer.serializers.save_npz(param_name, self.model) # NPZ形式で学習済みパラメータ書込み
        print("___Successfully completed\n\n")
    
# データセット作成
def create_dataset(data_dir):
    
    print("\n___Creating a dataset...")
    
    cnt = 0
    prc = ['/', '-', '\\', '|']
    
    # 画像セットの個数
    print("Number of Rough-Dataset: {}".format(len(os.listdir(data_dir))))
    # 画像データの個数
    for c in os.listdir(data_dir):
        d = os.path.join(data_dir, c)
        print("Number of image in a directory \"{}\": {}".format(c, len(os.listdir(d))))
    
    train = []  # 仮データセット[フォルダ名, ラベル]
    label = 0
    
    # 仮データセット作成
    for c in os.listdir(data_dir):
        
        print('\nclass: {}, class id: {}'.format(c, label))   # 画像フォルダ名とクラスIDの出力
       
        d = os.path.join(data_dir, c)                   # フォルダ名と画像フォルダ名の結合
        imgs = os.listdir(d)
        
        # JPEG形式の生データだけを読込
        for i in [f for f in imgs if ('jpg'or'JPG' in f)]:        
            
            # キャッシュファイルをスルー
            if i == 'Thumbs.db':
                continue
            
            train.append([os.path.join(d, i), label])       # 画像フォルダパスと画像パスを結合後、リストに格納->仮データセット

            cnt += 1
            
            print("\r   Loading a images and labels...{}    ({} / {})".format(prc[cnt%4], cnt, len(os.listdir(d))), end='')
            
        print("\r   Loading a images and labels...Done    ({} / {})".format(cnt, len(os.listdir(d))), end='')
        
        label += 1
        cnt = 0

    train_set = chainer.datasets.LabeledImageDataset(train, '.')    # データセット化
    
    print("\n___Successfully completed\n")
    
    return train_set
    
def main():
    
    # プログラム情報
    print("Face Recognition train Program(CH) ver.4")
    print("Last update date:    2020/03/12 (Stop development)\n")
    
    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='Face Recognition train Program(Chainer)')
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                        help='学習フォルダパスの指定(未指定ならエラー)')
    parser.add_argument('--out', '-o', type=str, 
                        default=os.path.dirname(os.path.abspath(__file__))+'/result'.replace('/', os.sep),
                        help='パラメータの保存先指定(デフォルト値=./result)')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='ミニバッチサイズの指定(デフォルト値=32)')
    parser.add_argument('--epoch', '-e', type=int, default=15,
                        help='学習回数の指定(デフォルト値=15)')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU IDの指定(負の値はCPU処理を示す, デフォルト値=-1)')
    args = parser.parse_args()

    # 学習フォルダパス未指定->例外
    if args.data_dir == None:
        print("\nException: Folder not specified.\n")
        sys.exit()
    # 存在しない学習フォルダ指定時->例外
    if os.path.exists(args.data_dir) != True:
        print("\nException: Folder {} is not found.\n".format(args.data_dir))
        sys.exit()

    # 設定情報出力
    print("=== Setting information ===")
    print("# Images folder: {}".format(os.path.abspath(args.data_dir)))
    print("# Output folder: {}".format(args.out))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Epoch: {}".format(args.epoch))
    print("===========================")

    # データセット作成
    train_set = create_dataset(args.data_dir)

    # 学習開始
    print("___Start training...")
    Trainer = trainer()
    Trainer.train(train_set, args.batch_size, args.epoch, args.gpu, args.out)
   
if __name__ == '__main__':
    main()