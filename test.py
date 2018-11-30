# import math
# list1 = ["nima", "daue", "zhangsan"]
# for index, item in enumerate(list1):
#  print index, item
# # dict={"nima":1,"daue333":3}

# print math.log(4,2)



#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time

def procedure():
    time.sleep(2.5)

t0 = time.clock()
procedure()
t1 = time.clock()
print "time.clock(): ", t1 - t0

t0 = time.time()
procedure()
t1 = time.time()
print "time.time()", t1 - t0
a=[0,2,3]
print a[0:-1]
print a[1:]



