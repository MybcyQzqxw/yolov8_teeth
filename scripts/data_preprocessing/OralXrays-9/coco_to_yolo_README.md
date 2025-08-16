coco2yolo_train，coco2yolo_val这两个是OrayXrays-9的，下载OrayXrays的annotations文件并解压后，把这两个文件放在annotations文件夹下，运行后会创建yolo文件夹并把json文件转换为yolo格式，存储在该文件夹下。然后把yolo文件夹下的labels_train2017重命名为train，labels_val2017重命名为val。再把文件夹结构调整为：

```
yolo/  
├── images/  
│   ├── train/                 # 训练图片（10000张）  
│   └── val/                   # 验证图片（2688张）  
├── labels/                       
│   ├── train/                 # 上面提到的重命名后的train文件夹（10000个标签）  
│   └── val/                   # 上面提到的重命名后的val文件夹（2688个标签）  
└── dataset.yaml               # 在运行coco2yolo_train时会生成这个文件，把它放在这个位置就可以
```
