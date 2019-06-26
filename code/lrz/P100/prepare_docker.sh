#!/bin/bash
echo 'This script is used to install additional modules inside the docker container'
echo 'Please edit this script and insert the right paths for you.' 
echo 'Note that the paths here are the paths in the docker container'
echo 'After that, remove this message and the following exit statement'
exit
echo '-----------------------------------------------------------'
echo 'Welcome to the prepare_docker script!'
echo 'This script installs the requirements, and runs tests.'
echo '-----------------------------------------------------------'
echo 'install requirements'
echo '-----------------------------------------------------------'
pip install -r '[INSERT_HERE_LOCAL_PATH_TO_REQU]/requirements.txt'
echo '-----------------------------------------------------------'
echo 'run tests'
python [INSERT_HERE_LOCAL_PATH_TO_TEST_M]/test_modules.py
echo '-----------------------------------------------------------'
echo 'The prepare_docker script ends here. Goodbye'
echo '-----------------------------------------------------------'

