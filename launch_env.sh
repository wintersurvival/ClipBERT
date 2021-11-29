PATH_TO_STORAGE=/fsx/home/fangliangs/ClipBERT/data
# docker image should be automatically pulled
source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/vis_db \
    $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained

