import student_success_model as ssm
import numpy as np

model = ssm.SSM()

arr = [x for x in range(0, 33)]

# (categorical cols) catcols = "school","address","famsize","Pstatus,"Mjob","Fjob","reason","guardian"catcols= [0,3,4,5,8,9,10,11]
model.read_csv("./include/student.csv",
               sep=";",
               # usecols=np.asarray(list(set(arr) - set(arr[:30]))))
               usecols=np.asarray(arr))

model.split_data(.25, 1)
#
model.svc_linear()
