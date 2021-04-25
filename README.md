# Face-Recognition-Chainer-
Face recognition program by CNN (Chainer implementation)

詳細は[こちら(訓練フェーズ)](https://qiita.com/hima_zin331/items/fa93c7c546da0c3ac31a)と[こちら(推論フェーズ)](https://qiita.com/hima_zin331/items/721a030e7d924340ee27)
*Not supported except in Japanese Language  
 
**Note: You need a camera device to run it.**

___

## face_recog_train_CH.py

**Command**  
```
python face_recog_train_CH.py -d <DIR_NAME> -e <EPOCH_NUM> -b <BATCH_SIZE>
                                                          (-o <OUT_PATH> -g <GPU_ID>)   
							  
EPOCH_NUM  : 15 (Default)  
BATCH_SIZE : 32 (Default)  
OUT_PATH   : ./result (Default)  
GPU_ID     : -1 (Default) *Not GPU  
```

**Output files**
- face_recog.model
	- Parameter file.
- accuracy.png
	- A graph plotting prediction accuracy.
- loss.png
	- A graph plotting loss values.
- cg.dot
	- Structure of the network model (DOT format).
- log
	- History of loss values and prediction accuracy (JSON format).
- snapshot_iter_XXX
	- Parameter and other snapshots.

**How to place training data**
```
train_data --- Class_A --- img1.jpg
            |           |- img2.jpg
	    |           |- ...
	    |
            |- Class_B --- img1.jpg
            |           |- img2.jpg
	    |           |- ...
	    |
	    -- Class_C --- img1.jpg
                        |- img2.jpg
	                |- ...
```
The command line argument (-d) specifies the parent directory (train_data in the example above).  
Labels are assigned to the child directories in order from 0 to name.  
(In the example above, Class_A is 0, Class_B is 1, and Class_C is 2)

**NOTE**
```
    for c in os.listdir(data_dir):
        d = os.path.join(data_dir, c)
        print("Number of image in a directory \"{}\": {}".format(c, len(os.listdir(d))))
```
The above process accounts for how many training data are available.  
If 'Thumbs.db' (thumbnail cache) is included, it is also accounted for. But don't worry, I've made sure it doesn't affect your training.

## face_recog_CH.py

**Command**  
```
python face_recog_train.py -p <PARAM_NAME> -e <CASCADE>
                                            (-d <CAMERA_ID>)
                                                          
CASCADE   : ./haar_cascade.xml (Default)  
CAMERA_ID : 0 (Default)  
```

## face_cut.py

**Command**  
```
python face_cut.py -d <IMG_DIR> -c <CASCADE>
                                (-o <OUT_PATH> -s <OUT_SIZE> -l <Index>)
                                                          
CASCADE   : ./haar_cascade.xml (Default)  
OUT_PATH  : ./result_crop (Default)
OUT_SIZE  : 32 (Default) *Output image size N x N.
Index     : 1 (Default) *Output image name index. Starting from dataN.jpg when you specify `-l N`.
```
