
BUCKET_NAME=directed-radius-213506.appspot.com
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=celeb_gan_$DATE
export JOB_DIR=gs://$BUCKET_NAME/jobs/$JOB_NAME
export SAVE_DIR=gs://$BUCKET_NAME/generated
export REGION=us-central1

gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --runtime-version 1.12 \
    --job-dir=$JOB_DIR \
    --python-version 3.5 \
    --package-path=trainer \
    --module-name trainer.task \
    --region $REGION -- \
    --save-dir $SAVE_DIR
