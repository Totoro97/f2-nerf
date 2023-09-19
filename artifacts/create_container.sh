IMG=zebincai/det-dev-llm:v01
REPO_NAME=f2nerf
NAME=zebin_${REPO_NAME}


docker run -it -d --name $NAME \
  --gpus all \
  --privileged \
  --hostname in_docker \
  --add-host in_docker:127.0.0.1 \
  --add-host $(hostname):127.0.0.1 \
  --shm-size 2G \
  -e DISPLAY \
  -p 6005:22 \
  -v /etc/localtime:/etc/localtime:ro \
  -v /media:/media \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/zebin/work/f2-nerf:/${REPO_NAME} \
  -w /${REPO_NAME} \
  $IMG \
  /bin/bash

