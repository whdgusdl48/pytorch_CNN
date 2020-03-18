import numpy as np
import matplotlib.pyplot as plt

def pprint(arr):
    print("type:{}".format(type(arr)))
    print("shape: {}, dimension: {}, dtype:{}".format(arr.shape, arr.ndim, arr.dtype))
    print("Array's Data:\n", arr)

newarr = np.array([1,2,3])
pprint(newarr)
newarr2 = np.array([[1,2,3],[4,5,6]])
pprint(newarr2)
newarr3 = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
pprint(newarr3)

# 배열 생성 및 초기화

zerosarr = np.zeros((3,4))
pprint(zerosarr)
onearr = np.ones((2,3,4))
pprint(onearr)
fullarr = np.full((1,3,5),3)
pprint(fullarr)
eyearr = np.eye(4)
pprint(eyearr)
linearr = np.linspace(0,1,5)
pprint(linearr)

# plt.figure()
# plt.plot(linearr,'o')
# plt.show()

arrangearr = np.arange(0,10,2,np.float)
pprint(arrangearr)

# plt.figure()
# plt.plot(arrangearr,'o')
# plt.show()

logarr = np.logspace(0.1,1,20,endpoint=True)
pprint(logarr)

# plt.figure()
# plt.plot(logarr,'o')
# plt.show()

random1 = np.random.normal(0,1,10000)

plt.figure()
plt.hist(random1,bins=100)
plt.show()

random2 = np.random.rand(3,4)
pprint(random2)

random3 = np.random.randint(5,10,size=(2,4))
pprint(random3)