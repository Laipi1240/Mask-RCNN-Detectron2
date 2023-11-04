#!/usr/bin/env bash

ARGS=("$@")
USER_NAME="user"
# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
    xauth_list=$(xauth nlist $DISPLAY)
    xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
    if [ ! -z "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

# Prevent executing "docker run" when xauth failed.
if [ ! -f $XAUTH ]; then
    echo "[$XAUTH] was not properly created. Exiting..."
    exit 1
fi

BASH_OPTION=bash
xhost +
docker run \
    -it \
    --rm \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v "$XAUTH:$XAUTH" \
    -v "/home/$USER/unity-example-1:/home/$USER/unity-example-1" \
    -v "/home/$USER/Mask-RCNN-Detectron2:/home/$USER/Mask-RCNN-Detectron2" \
    -v "/home/$USER/Downloads:/home/$USER/Downloads" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -w "/home/$USER" \
    --user "${USER_NAME}:root" \
    --network host \
    --privileged \
    --security-opt seccomp=unconfined \
    yutingk/unitydataset \
    $BASH_OPTION
xhost -