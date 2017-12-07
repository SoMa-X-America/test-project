import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


##############
# 파일 로드
##############
def load_data_file(fname, num_flds=0):
  data = []
  with open(fname) as fp:
    for line in fp:
      flds = line.split()
      for j in range(0,11):
        data.append(flds[j:j+1])
    return np.matrix(data=data, dtype=np.float32, copy=False)
degreedata = load_data_file('C:/Users/YeoChunghyun/Desktop/txttest/result_Degree.txt', num_flds=10 + 1)
print('DEG', degreedata.shape)
degreedata=degreedata.reshape((54465,11,1))
print('DEG', degreedata.shape)
print(degreedata[0,10])
#print(len(degreedata))

mat_for_plot=[0]

def cal_data(data,num=0):
  count=0
  sum=[num,0,0,0,0,0,0,0,0,0,0]
  result=0
  for i in range(0,len(data)-num):
    if data[i+num,0]-data[i,0]-num==0:
      count=count+1
      print(count)
      for j in range(1,11):
        sum[j]=sum[j]+ abs(data[i+num,j]-data[i,j])
  for i in range(1,11):
    sum[i]=sum[i]/count
    result+=sum[i]
  #print(sum)
  #print(result/10)
  return result
  '''
  if result/10<0:
    return -result/10
  else:
    return result/10
'''
min=99
min_num=0
for i in range(1,251):
  temp=cal_data(degreedata,i)
  if temp<min:
    min_num=i
    min=temp
  mat_for_plot.append(temp)
print(min_num,"개씩")
print('그때의 값',min,'\n')
print(mat_for_plot)
plt.figure()
plt.plot(mat_for_plot)
plt.show()




