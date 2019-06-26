############# READ ME to the server_scripts folder ########################
Contents:
-> Overall information
-> Where to place the files
-> Data storage options
-> Additional information, help, tutorials

################## Overall information ################################
When using the LRZ GPU respources, you have to options: the single P100s and the DGXs
Sometimes a user only gets access to the P100s at first and to the DGXs later on.
For both, the process is as follows: 
-> make a reservation at datalab.srv.lrz.de (need to be in eduroam or mwn to access)
-> When time slot starts, log into the server via ssh (IP is sent via email)
-> During the slot, you can work on the server 
   It is HIGHLY recommended to use byobu (text-based window manager and terminal multiplexer)
   (for P100: create docker container, on the DGX you already start inside of the container)
   
The major difference in working on the P100 compared to the DGX is the data storage and mounting:

P100: you start with nothing, and have to get your data and code from somwhere. 
One option (probably not the best), is to mount your DSS directory via sshfs, and copy everything you need to the SSD local to the P100 for fast access
To get your results, you again have to copy them to the mounted storage (the local storage is lost after your time is up)
For how to automate some of this, see the scripts in the server_scripts folder, usage is explained in instructionsP100VirtualServer.txt

DGX: you start already in  a docker container with fast access to your DSS directory. So no initial copying of data and final copying of results needed

For more information on how to work on the P100 and the DGX and how to use the scripts, see the instructionsP100VirtualServer.txt and instructionsDGX.txt files

#################### Where to place the files ##############################
It is recommended to have a dedicated server_scripts directory in your project, where the files and scripts are stored.
When working on the DGX, that's enough. When working on the P100, you have get access to the get_data.sh and get_docker.sh scripts BEFORE you can mount the DSS directory.
On way to solve this is to put them into a cloud storage (did this with dropbox) and get them via typing wget 'URL' 

##################### Data storage options #################################
These instructions assume the access to the LRZ DSS storage. You can adapt it to work with the LRZ Linux Cluster Storage as well (Not for DGX)

###################### Additional Information ##############################
See https://github.com/stefan-it/lrz-gpu-tutorial for more information and tutorials (courtesy of Stefan Schweter https://schweter.eu/)






