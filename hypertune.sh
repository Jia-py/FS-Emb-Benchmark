nnictl create --config nni-experiments/config/DCN.yaml

while true
do
  status=$(nnictl experiment status | grep status | awk -F'"' '{print $4}')
  if [ $status = 'DONE' ]; then
    echo "Experiment finished with status DONE"
    break
  elif [ $status = 'ERROR' ]; then
    echo "Experiment finished with status ERROR"
    exit 1
  fi
  sleep 5
done
nnictl stop
sleep 10

nnictl create --config nni-experiments/config/pnn.yaml

while true
do
  status=$(nnictl experiment status | grep status | awk -F'"' '{print $4}')
  if [ $status = 'DONE' ]; then
    echo "Experiment finished with status DONE"
    break
  elif [ $status = 'ERROR' ]; then
    echo "Experiment finished with status ERROR"
    exit 1
  fi
  sleep 5
done
nnictl stop
sleep 10

nnictl create --config nni-experiments/config/autoint.yaml
while true
do
  status=$(nnictl experiment status | grep status | awk -F'"' '{print $4}')
  if [ $status = 'DONE' ]; then
    echo "Experiment finished with status DONE"
    break
  elif [ $status = 'ERROR' ]; then
    echo "Experiment finished with status ERROR"
    exit 1
  fi
  sleep 5
done
nnictl stop
sleep 10

nnictl create --config nni-experiments/config/fm.yaml
while true
do
  status=$(nnictl experiment status | grep status | awk -F'"' '{print $4}')
  if [ $status = 'DONE' ]; then
    echo "Experiment finished with status DONE"
    break
  elif [ $status = 'ERROR' ]; then
    echo "Experiment finished with status ERROR"
    exit 1
  fi
  sleep 5
done
nnictl stop
sleep 10

nnictl create --config nni-experiments/config/deepfm.yaml
while true
do
  status=$(nnictl experiment status | grep status | awk -F'"' '{print $4}')
  if [ $status = 'DONE' ]; then
    echo "Experiment finished with status DONE"
    break
  elif [ $status = 'ERROR' ]; then
    echo "Experiment finished with status ERROR"
    exit 1
  fi
  sleep 5
done
nnictl stop
sleep 10

nnictl create --config nni-experiments/config/widedeep.yaml
while true
do
  status=$(nnictl experiment status | grep status | awk -F'"' '{print $4}')
  if [ $status = 'DONE' ]; then
    echo "Experiment finished with status DONE"
    break
  elif [ $status = 'ERROR' ]; then
    echo "Experiment finished with status ERROR"
    exit 1
  fi
  sleep 5
done
nnictl stop
sleep 10

nnictl create --config nni-experiments/config/xdeepfm.yaml
while true
do
  status=$(nnictl experiment status | grep status | awk -F'"' '{print $4}')
  if [ $status = 'DONE' ]; then
    echo "Experiment finished with status DONE"
    break
  elif [ $status = 'ERROR' ]; then
    echo "Experiment finished with status ERROR"
    exit 1
  fi
  sleep 5
done
nnictl stop
sleep 10