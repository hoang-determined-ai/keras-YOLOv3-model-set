export DEBIAN_FRONTEND=noninteractive

# configure locale
apt-get update
# make sure that locales package is available
apt-get install --reinstall -y locales
# uncomment chosen locale to enable it's generation
sed -i 's/# pl_PL.UTF-8 UTF-8/pl_PL.UTF-8 UTF-8/' /etc/locale.gen
# generate chosen locale
locale-gen pl_PL.UTF-8
# set system-wide locale settings
export LANGUAGE pl_PL
# verify modified configuration
dpkg-reconfigure locales

apt install python3-opencv -y
apt install unzip
pip install Cython
pip install -r requirements.txt
mkdir -p /data/COCO2017 && \
    wget -O /data/COCO2017/train2017.zip http://images.cocodataset.org/zips/train2017.zip && \
    wget -O /data/COCO2017/val2017.zip http://images.cocodataset.org/zips/val2017.zip && \
    wget -O /data/COCO2017/test2017.zip http://images.cocodataset.org/zips/test2017.zip && \
    wget -O /data/COCO2017/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip && \
    wget -O /data/COCO2017/image_info_test2017.zip http://images.cocodataset.org/annotations/image_info_test2017.zip && \
    wget -O /data/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
cd /data/COCO2017 && \
    unzip -e train2017.zip && unzip -e val2017.zip && unzip -e test2017.zip && \
    unzip -e annotations_trainval2017.zip && unzip -e image_info_test2017.zip
cp -rf /data/COCO2017/train2017.txt /data/COCO2017/trainval.txt && cat /data/COCO2017/val2017.txt >> /data/COCO2017/trainval.txt

cd /run/determined/workdir/tools/dataset_converter && \
    python coco_annotation.py --dataset_path=/data/COCO2017/ --output_path=/data/COCO2017
cd /run/determined/workdir/tools/model_converter && \
    python convert.py /run/determined/workdir/cfg/yolov3.cfg /data/yolov3.weights /data/yolov3.h5

cd /run/determined/workdir/