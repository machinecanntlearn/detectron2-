import os
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data import MetadataCatalog

from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.datasets.pascal_voc import register_pascal_voc


# -------- COCO -------- #
def register_all_coco(root="datasets"):

    # 将划分的数据集的meta信息保存
    SPLITS = [
        ("coco_2014_train","COCO2014/train2014", "coco/annotations/instances_train2014.json"),
        ("coco_2014_val","COCO2014/val2014", "coco/annotations/instances_val2014.json"),
        ("coco_2017_train","COCO2017/train2017", "coco/annotations/instances_train2017.json"),
        ("coco_2017_val","COCO2017/val2017", "coco/annotations/instances_val2017.json"),
        # 数据集的名字， img的路径，annotation的路径
    ]
    for key, image_root, json_file in SPLITS:
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# -------- PASCAL VOC -------- #
def register_all_voc(root="datasets"):

    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
