host='mobix'

port=9999

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

LONGOPTS=port
OPTIONS=p:
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2
fi
eval set -- "$PARSED"
while true; do
    case "$1" in
        -p|--port)
            port="$2"
            shift 2
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

ssh $host "cd ~/Workspace/Pandia && ./scripts/stop_webrtc_receiver.sh -p $port"