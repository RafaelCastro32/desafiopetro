ls -d "$PWD"/cfg/task3_qr_version1/test/* > listOfFilesTEST.list
ls -d "$PWD"/cfg/task3_qr_version1/train/* > listOfFilesTRAIN.list
./darknet detector train cfg/digitos.data cfg/my_tiny.cfg darknet53.conv.74 -dont_show
