#!/bin/bash
echo 'This script is used to install sshfs, mount the DSS storage, and copy data to the local storage'
echo 'Please edit this script and insert the right paths for you.'
echo 'After that, remove this message and the following exit statement'
exit

echo '-----------------------------------------------------------'
echo 'Welcome to the get data script'
echo 'This script installs sshfs, mounts the DSS storage and copies it to the GPU server ssd'
echo '-----------------------------------------------------------'
echo 'Have you already started byobu? It is recommended to start byobu before continuing'
echo 'It is also helpful to open a new byobu window so that you can do other things while this script is running'
echo 'To open a new window, press F2, to change between existing windows, press F3'
echo '-----------------------------------------------------------'
while true; do
    read -p "Continue? (y/n)" yn
    case $yn in
        [Yy]* ) echo '-----------------------------------------------------------'
                read -p "Please enter your LRZ id: " LRZ_ID
                echo '-----------------------------------------------------------'
                echo 'Installing sshfs';
		        sudo apt install sshfs; 
                y
                echo '-----------------------------------------------------------'
		        echo 'preparing new directories';
		        mkdir -p -v /ssdtemp/mounted; 
		        mkdir -p -v /ssdtemp/local;
                echo '-----------------------------------------------------------'
		        echo 'mounting storage via sshfs, you may be asked for your password';
                sshfs $LRZ_ID@lxlogin6.lrz.de:/dss/[INSERT_HERE_PATH_ON_DSS]/$LRZ_ID /ssdtemp/mounted;
		        echo 'mounting completed';
                echo '-----------------------------------------------------------'
		        echo 'copying data from the mounted  directory, this will take some time';
        		cp -r /ssdtemp/mounted/[INSERT_HERE_DIRECTORY_TO_BE_COPIED] /ssdtemp/local;
		        echo 'data copied';
                echo '-----------------------------------------------------------'
                echo 'The get data script ends here. Goodbye'
                echo '-----------------------------------------------------------'
		        break;;
		
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done 
