#########################################################
About the Intel AI DevCloud

This document contains cluster usage basics:

* How to get started with using the cluster, 

* Where to find machine learning frameworks,

* Jupyter Notebook. 

We highly recommend that you read this document first.


If you have further questions, you can post them at:
https://colfaxresearch.com/forums/colfax-cluster/

Sincerely,

AI DevCloud Team
Colfax Research


#########################################################


Table of Contents
  
1. Computation on the Cluster
  
2. Basic Job Submission
  
3. Running Multiple Jobs
  
4. Data Management
  
5. Python and Other Tools
  
6. Conda environments

7. Jupyter Notebook




#########################################################

1. Computation on the Cluster

Eight-word summary: do not run jobs on the login node.


When you log in, you will find yourself on the host c009,
 which is your login node. 
This node is intended only for 
code development and compilation, but NOT for computation.
That is because it does not have much compute power, 
and, 
additionally, there are limitations on CPU time and RAM 
usage on the login node; your workload will be killed if
 it exceeds the limit.



To run computational workloads on powerful compute nodes,
you must submit a job through the Torque job queue using 
qsub directives. See Section 2 for a sample job script.


You can find more detailed information about jobs at
https://access.colfaxresearch.com/?p=compute

(If the link does not work, go to the original welcome 
email and then click on the instruction link.
Then go to the 'compute' page.)



#########################################################


2. Basic Job Submission : 

Submitting a job can be done through a job script file. 
Suppose that you have a Python application,
'my_application.py'. 
In the same folder, use your favorite text editor and 
create a file "myjob". Then add the following three lines.
   

#PBS -l nodes=1:ppn=2

   cd $PBS_O_WORKDIR
   
python my_application.py


The first line is a special command that requests one 
Intel Xeon processor, and all processing slots on the 
node (see the access page). The second line ensures that
the script runs in the same directory as where you have 
submitted it. And the third line runs the Python 
application.

You can submit this job with:
   
[u100@c001 ~]# qsub myjob



This command will return a Job ID, which is the tracking 
number for your job.

You can track the job with:
   
[u100@c001 ~]# qstat


Once it is complete, the output will be in the files:
   
[u100@c001 ~]# cat myjob.oXXXXXX

   [u100@c001 ~]# cat myjob.eXXXXXX


Here 'XXXXXX' is the Job ID. The .o file contains the 
standard output stream, and .e file contains the error
stream.


For more information on job scripts, see:
https://access.colfaxresearch.com/?p=compute


#########################################################


3. Running Multiple Jobs

This cluster gives you access to over 80 Intel Xeon 
Scalable Processors. Together, you have a theoretical 
peak over 200 TFLOP/s of machine learning performance.

However, to get this performance, you need to correctly 
use the cluster as discussed in this section.


For most machine learning workloads, reserve 1 node per 
job (this is the default). If you reserve more nodes,
your application will not take advantage of them, unless 
you explicitly use a distributed training library such as 
MLSL. 
Most people do not. Reserving extra nodes, whether
your application uses them or not, reduces the queue
priority of your future jobs.


Instead, to take advantage of multiple nodes available to 
you, submit multiple single-node jobs with different 
parameters. 

For example, you can submit several jobs with 
different values of the learning rate like this:
  

* Your application “my_application.py” should use the 
command-line arguments to set the learning rate:


  import sys
print("Running with learning rate %s"%sys.argv[1])
learning_rate=float(sys.argv[1])


* Your job file “myjob” may contain the following:
  

#PBS -l nodes=1:ppn=2
cd $PBS_O_WORKDIR
  
python my_application.py $1


* You will submit several jobs like this:
  

[u100@c001 ~]# qsub myjob -F “0.001”
[u100@c001 ~]# qsub myjob -F “0.002”
[u100@c001 ~]# qsub myjob -F “0.005”
[u100@c001 ~]# qsub myjob -F “0.010”


If resources are available, all 4 jobs will start at the 
same time on different compute nodes. 
This workflow will produce results up to 4 times faster 
than if you had only one compute node.

If you do have a distributed workload (e.g. with MPI) then
you can find information on requesting multiple nodes at:
https://access.colfaxresearch.com/?p=compute

#########################################################


4. Data Management

The quota for your home folder is 200 GB. Home folder is 
NFS-shared between the login node and the compute nodes.


Some machine learning datasets can be found in /data/
You may use /tmp directory, but note that this directory
is cleaned every 2 hours.


#########################################################

#########################################################


5. Python, TensorFlow and Other Tools: 

For best performance, use Intel Distribution for Python 
from /glob/intel-python/python2/bin/python and 
/glob/intel-python/python2/bin/python These paths are 
included in your environment by default. 

All frameworks and tools, such as TensorFlow and Intel 
Compiler, are located in the /glob/ directory. 


If you need to install some additional Python modules, 
use a local Conda environment in your home directory. 
Same for non-Python tools: put them in your home folder.

You can also use the --user switch for pip to install in 
your local or home directory

	pip install --user opencv


#########################################################

#########################################################


6. Conda environments

Conda is available for users that want to easily manage 
their environments. For best performance, create new 
environments using Intel Distribution for Python. 

To do this, first add Intel's channel into Conda:
	conda config --add channels intel

When creating your environment, you have the option of
picking between core or full versions of Python 2 and 3.

To create an environment with core Python 3:
	conda create -n <nameofyourenv> intelpython3_core python=3

For core Python 2:
	conda create -n <nameofyourenv> intelpython2_core python=2

To use the full version instead of core, replace "core" with
"full".
	conda create -n <nameofyourenv> intelpython2_full python=2

In order to use the newly created environment, users will need 
to activate it.
	source activate <nameofyourenv>

To leave the environment:
	source deactivate

The Intel channel provides a variety of Python versions. If
a particular version is required, you can use the search
option to see what is available. Intel distributed packages 
are tagged "[intel]".
	conda search -f python

#########################################################

#########################################################
7. Jupyter Notebook

If you would like to use Jupyter Notebook on Intel AI 
DevCloud, you can find the instructions here:
https://access.colfaxresearch.com/?p=connect#sec-jup

HOwever, note that JupyterHub will only run on one node,
so you will not be able to take advantage multi-node
computation.

#########################################################
