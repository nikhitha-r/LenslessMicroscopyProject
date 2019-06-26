#!/bin/bash
echo 'This script is used to copy data (e.g. result directories) periodically from the local storage to the mounted storage'
echo 'Please edit this script and insert the right paths for you.'
echo 'After that, remove this message and the following exit statement'
exit
echo 'Welcome to the copy_results_script!'
read -p "Please enter your preferred copying interval in minutes: " COP_INT
SEC_IN_MIN=60
COP_INT_SEC=$((COP_INT*SEC_IN_MIN))
echo 'starting the automatic copying'
timestamp() {
  date +"%T"
}

while true; do
  echo '-------------------------------------------------'
  echo 'start copying at:' 
  timestamp
  cp -r -u /ssdtemp/[LOCAL_PATH] /ssdtemp/[MOUNTED_PATH];
  echo '------------------------------------------------'
  echo 'done copying at:' 
  timestamp
  echo  'now wait for ' $COP_INT ' min'
  echo '------------------------------------------------'
  sleep $COP_INT_SEC
done

