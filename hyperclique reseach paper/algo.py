#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
mydata=[]
with open('kosarak.dat') as f_d, open('kosarak.csv','w') as f_csv:
    csv_writer=csv.writer(f_csv)
    
    for i in f_d:
        line=[col.strip() for col in i.split(' ')]
        mydata.append(line)
        csv_writer.writerow(line)


# In[2]:


for i in range(0,len(mydata)):
    for j in range(0,len(mydata[i])):
        mydata[i][j]=int(mydata[i][j])
        
for i in range(10):
    print(mydata[i])


# In[3]:


noOfTransaction=len(mydata)


# In[4]:


import time
s_time=time.time()
from collections import Counter

freq_of_candidates = Counter()
for i in mydata:
    # print(Counter(d))
    freq_of_candidates.update(i)
# print(freq_of_candidates)


# In[5]:


result = list(x for l in mydata for x in l)
result=set(result)
unique_list=list(result)
unique_list.sort()
# print(unique_list)
 


# In[6]:


dict1={}
min_sup=0.2
h_con=0.03
for i in freq_of_candidates:
    count=freq_of_candidates[i]/noOfTransaction
    #print(count)
    if(count>min_sup):
        dict1[i]=count
# print(dict1)

import collections
sorted_dict = collections.OrderedDict(dict1)
# print(sorted_dict)
dict2={}
global_list2=[]
dict2=dict(sorted_dict)
support={}
support=dict2
# print(dict2)
global_list2=list(dict2.keys())
global_list2.sort()

# print(global_list2)
# print(global_list2)
# global_list1=to_be_list
global_list1 = [] 
global_list1= list(map(lambda x:[x], global_list2))
# print(global_list1)


# In[7]:


# #global_list=[]
# def gen_set(list2,k):
#     list3=[]
#     if(k==2):
#         for i in range(len(list2)+1):
#              for j in range(i+1,len(list2)):
#                 list3.append([list2[i],list2[j]])
#     if(k>2):
#         for i in range(len(list2)):
#             for j in range(i+1,len(list2)):
#                 x=set(list2[i])
#                 y=set(list2[j])
#                 list3.append(list((x).union(y)))
                       
                                     
# #                 #print(i,j,l,to_be_used[i][l])
# #                     elif(l==k-3):
# # #                         list3.append(list(set(list2[i]).union(set(list2[j]))))
                       
                        
#     print(list3)                    
#     return list3             
def gen_set(list2,k):
    list3=[]
    if(k==2):
        for i in range(len(list2)+1):
             for j in range(i+1,len(list2)):
                list3.append([list2[i],list2[j]])
        #print(list3)
    if(k>2):
        for i in range(len(list2)):
            for j in range(i+1,len(list2)):
                for l in range(k-2):
                    if(list2[i][l]!=list2[j][l]):
                        break
                #print(i,j,l,to_be_used[i][l])
                    elif(l==k-3):
                        list3.append(list(set(list2[i]).union(list2[j])))
#     print(list3)                    
    return (list3)             


# In[8]:


# global_list2=gen_set(global_list2,2)
# print(global_list2)


# In[9]:


# global_list2=gen_set(global_list2,3)
# print(global_list2)


# In[10]:


def supp_prun(list3):
    #support={}
    global global_list2
    list5=[]
    global_list=[]
    for i in list3:
        c=0
        count1=0
        for j in mydata:
            if(set(i).issubset(set(j))):
                c+=1
        count1=(c/len(mydata))
        if(count1<min_sup):
            #print(count1)
            list5.append(i)
#     for i in list5:
#         i=set(i)
#         for k in global_list2:
#             k=set(k)
#             if(i.issubset(k)):
    #print(list5)            # global_list.append(list(k))
    x=set(frozenset(i) for i in list5)
    y=set(frozenset(j) for j in global_list2)
    z=((y).difference(x) )  
    global_list2=[list(x) for x in z]
  
 


# In[11]:


# supp_prun(global_list2)
# print(global_list2)


# In[12]:


def gen_cross():
    global global_list2
    global support
    global_list=[]
    list4=[]
   
    for i in support:
        for j in support:
            pro=support.get(i)*h_con
            if((i!=j) and (pro>support.get(j))):
                i=set([i])
                list4.append(i)
                break
        
    for i in list4:
        for k in global_list2:
            k=set(k)
            if(i.issubset(k)):
                global_list.append(list(k))
    x=set(frozenset(i) for i in global_list)
    y=set(frozenset(j) for j in global_list2)
    z=((y).difference(x) )  
    global_list2=[list(x) for x in z]
  


# In[13]:



def gen_hcon_cross(support1):
    global global_list2
    global global_list1
    for i in list(dict2.keys()):
        h_c=0
        for j in list(support1.keys()):
            if(set([i]).issubset(set(j))):
                h_c=support1.get(j)/dict2.get(i)
                if(h_c<h_con):
                    global_list2.remove(list(j))
                    del support1[j]
                break
    return support1


# In[14]:


k=2
while(len(global_list2)):
    
    support1={}
    global_list2=gen_set(global_list2,k)
    supp_prun(global_list2)
    gen_cross()
    for i in global_list2:
        c=0
        count1=0
        for j in mydata:
            if(set(i).issubset(set(j))):
                c+=1
        count1=(c/len(mydata))
        support1[tuple(i)]=count1

    support1=gen_hcon_cross(support1)
    for l in global_list2:
         global_list1.append(l)
    
    k=k+1
    support=support1
    #print(global_list2)
    #print(gen_hcon_cross())
    
# print(global_list2)
# print(support)
print(global_list1)

print("--%s seconds--" %(time.time()-s_time))
   


# In[15]:


hyper_p=[]
hyper_p.append(len(global_list1))
print(hyper_p)


# In[16]:


#https://swcarpentry.github.io/python-novice-gapminder/09-plotting/
#https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
import matplotlib.pyplot as plt
h_conf=[0.1,0.2,0.3,0.4,0.6,0.8]
exe_time=[449.171,434.192,432.653,447.96,431.9706,437.884]
plt.plot(h_conf, exe_time, color='blue', linestyle='dashed', markersize=12)
plt.title('Minimum h-confidence thresholds vs execution time')

plt.xlabel('Minimum h-confidence threshold')
plt.ylabel('execution time in seconds')
plt.show()


h_conf=[0.1,0.2,0.3,0.4,0.6,0.8]
hyper=[33,29,27,27,27,27]
plt.plot(h_conf, hyper, color='blue', linestyle='dashed', markersize=12)
plt.title('Minimum h-confidence thresholds vs no. of hyperclique patterns')

plt.xlabel('Minimum h-confidence threshold')
plt.ylabel('No. of hyperclique patterns')
plt.show()

sup=[0.05,0.1,0.15,0.2,0.5]
exe_time=[104.06,25.049,12.279,18.834,6.687]
plt.plot(h_conf, exe_time, color='blue', linestyle='dashed', markersize=12)
plt.title('Minimum support thresholds vs execution time')

plt.xlabel('Minimum support threshold')
plt.ylabel('execution time in seconds')
plt.show()

sup=[0.05,0.1,0.15,0.2,0.5]
hyper1=[33,9,7,5,1]
plt.plot(h_conf, hyper1, color='blue', linestyle='dashed', markersize=12)
plt.title('Minimum support thresholds vs No. of hyperclique patterns')

plt.xlabel('Minimum support threshold')
plt.ylabel('No. of hyperclique patterns')
plt.show()


# In[ ]:




