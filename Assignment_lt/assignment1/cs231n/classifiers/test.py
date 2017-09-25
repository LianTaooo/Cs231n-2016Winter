#1.test quicksort
#定义函数，使用递归调用
#块注释：ctrl+1

#method1:快速排序
#def quicksort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivot = arr[len(arr)//2]
#     left = [x for x in arr if x<pivot]
#     middle = [x for x in arr if x== pivot]
#     right = [x for x in arr if x> pivot]
#     return quicksort((left))+middle+quicksort(right)
#print(quicksort([3,6,8,10,1,2,1]))

#def qsort(arr):
#    if len(arr) <= 1:
#        return arr
#    else:
#        pivot = arr[0]
#        return qsort([x for x in arr[1:] if x < pivot]) + \
#               [pivot] + \
#               qsort([x for x in arr[1:] if x >= pivot])
#print(qsort([3,6,8,10,1,2,1,45,1,7]))


import numpy as np
import time
# =============================================================================
# #数组转矩阵；向量化计算
# X=np.ones((3,4))
# Z=np.ones((2,4))*3
# 
# XZ = np.dot(X,Z.T)
# X_squaresum=np.sum(X**2, axis=1)
# Z_squaresum = np.sum(Z.T **2, axis =0)
# dists = np.matrix(X_squaresum).T +np.matrix(Z_squaresum) - 2*np.matrix(XZ)
# dists = np.array(dists)
# =============================================================================

# =============================================================================
# #numPy 生成矩阵
# a=np.array([[1,5],[2,4]])   #a 为2维数组
# b=a[1]    #b 也为数组 array([2,4])
# =============================================================================

# =============================================================================
# #返回一维数组中的出现次数最多的值；没有，返回第一个数
# closest_y=np.array([0, 1, 4, 3, 2, 6, 7, 7])
# y_pred = np.argmax(np.bincount(closest_y))
# =============================================================================


# =============================================================================
# W = np.random.randn(3073, 10) * 0.0001 
# =============================================================================


# =============================================================================
# time_start=time.time()
# 
# time_finish=time.time()
# print('Time: %f' %(time_finish-time_start))
# =============================================================================


# =============================================================================
# #softmax 优化计算
# f = np.array([123, 456, 789]) # 例子中有3个分类，每个评分的数值都很大
# p = np.exp(f) / np.sum(np.exp(f)) # 不妙：数值问题，可能导致数值爆炸
# 
# # 那么将f中的值平移到最大值为0：
# f -= np.max(f) # f becomes [-666, -333, 0]
# p = np.exp(f) / np.sum(np.exp(f)) # 现在OK了，将给出正确结果
# =============================================================================


hidden_size=5
a=np.zeros(hidden_size) 

























