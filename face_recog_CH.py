from PIL import Image
import numpy as np
import cv2

import sys
import os
import argparse as arg

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.serializers as S
  
# ==================================== face_recog_train_CH.pyと同じネットワーク構成　====================================
class CNN(chainer.Chain):
    def __init__(self, n_out):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 16, 5, 1, 0),  
            conv2=L.Convolution2D(16, 32, 5, 1, 0),  
            conv3=L.Convolution2D(32, 64, 5, 1, 0),  
            link=L.Linear(None, 1024),  
            link_class=L.Linear(None, n_out),  
        )
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2)
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.link(h3))
        return self.link_class(h4)
# ================================================================================================================

def main():

    # コマンドラインオプション引数
    parser = arg.ArgumentParser(description='RealTime Face Recognition Program(Chainer)')
    parser.add_argument('--param', '-p', type=str, default=None,
                        help='学習済みパラメータの指定(未指定ならエラー)')
    parser.add_argument('--cascade', '-c', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/haar_cascade.xml'.replace('/', os.sep),
                        help='Haar-cascadeの指定(デフォルト値=./haar_cascade.xml)')
    parser.add_argument('--device', '-d', type=int, default=0,
                        help='カメラデバイスIDの指定(デフォルト値=0)')
    args = parser.parse_args()

    # パラメータファイル未指定時->例外
    if args.param == None:
        print("\nException: Trained Parameter-File not specified.\n")
        sys.exit()
    # 存在しないパラメータファイル指定時->例外
    if os.path.exists(args.param) != True:
        print("\nException: Trained Parameter-File {} is not found.\n".format(args.param))
        sys.exit()
    # 存在しないHaar-cascade指定時->例外
    if os.path.exists(args.cascade) != True:
        print("\nException: Haar-cascade {} is not found.\n".format(args.cascade))
        sys.exit()

    # 設定情報出力
    print("=== Setting information ===")
    print("# Trained Prameter-File: {}".format(os.path.abspath(args.param)))
    print("# Haar-cascade: {}".format(args.cascade))
    print("# Camera device: {}".format(args.device))
    print("===========================")

    # カメラインスタンス生成
    cap = cv2.VideoCapture(args.device)
    # FPS値の設定
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # 顔検出器のセット
    detector = cv2.CascadeClassifier(args.cascade)

    # 学習済みパラメータの読み込み
    model = L.Classifier(CNN(2))
    S.load_npz(args.param, model)

    red = (0, 0, 255)
    green = (0, 255, 0)
    p = (10, 30)
    
    while True:

        # フレーム取得
        _, frame = cap.read()

        # カメラ認識不可->例外
        if _ == False:
            print("\nException: Camera read failure.\n".format(args.param))
            sys.exit()

        # 顔検出
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray)
 
        # 顔未検出->continue
        if len(faces) == 0:

            cv2.putText(frame, "face is not found",
                    p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, red, thickness=2)
            cv2.imshow("frame", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue
        
        # 顔検出時
        for (x, y, h, w) in faces:
            
            # 顔領域表示
            cv2.rectangle(frame, (x, y), (x+w, y+h), red, thickness=2) 
            
            # 顔が小さすぎればスルー
            if h < 50 and w < 50:
                cv2.putText(frame, "detected face is too small",
                    p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, red, thickness=2)
                cv2.imshow("frame", frame)
                break
                
            # 検出した顔を表示
            cv2.imshow("gray", cv2.resize(gray[y:y + h, x:x + w], (250, 250)))
                    
            # 画像処理
            face = gray[y:y + h, x:x + w]
            face = Image.fromarray(face)
            face = np.asarray(face.resize((32, 32)), dtype=np.float32)
            recog_img = face[np.newaxis, :, :]
                    
            # 顔識別
            y = model.predictor(chainer.Variable(np.array([recog_img])))
            c = F.softmax(y).data.argmax()
            
            if c == 0:
                cv2.putText(frame, "Abe Sinzo",
                    p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, thickness=2)     
            elif c == 1:
                cv2.putText(frame, "Aso Taro",
                    p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, thickness=2)    
                
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    
    # リソース解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()